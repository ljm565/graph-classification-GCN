import gc
import sys
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from torch import distributed as dist

from tools import TrainingLogger
from trainer.build import get_model, get_data_loader
from utils import RANK, LOGGER, colorstr, init_seeds
from utils.filesys_utils import *
from utils.training_utils import *




class Trainer:
    def __init__(
            self, 
            config,
            mode: str,
            device,
            is_ddp=False,
            resume_path=None,
        ):
        init_seeds(config.seed + 1 + RANK, config.deterministic)

        # init
        self.mode = mode
        self.is_training_mode = self.mode in ['train', 'resume']
        self.device = torch.device(device)
        self.is_ddp = is_ddp
        self.is_rank_zero = True if not self.is_ddp or (self.is_ddp and device == 0) else False
        self.config = config
        self.world_size = len(self.config.device) if self.is_ddp else 1
        if self.is_training_mode:
            self.save_dir = make_project_dir(self.config, self.is_rank_zero)
            self.wdir = self.save_dir / 'weights'

        # path, data params
        self.config.is_rank_zero = self.is_rank_zero
        self.resume_path = resume_path

        # init model, dataloader, etc.
        self.modes = ['train', 'validation'] if self.is_training_mode else ['train', 'validation', 'test']
        self.dataloaders = get_data_loader(self.config, self.modes, self.is_ddp)
        self.model = self._init_model(self.config, self.mode)
        self.training_logger = TrainingLogger(self.config, self.is_training_mode)

        # save the yaml config
        if self.is_rank_zero and self.is_training_mode:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.config.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', self.config)  # save run args
        
        # init criterion, optimizer, etc.
        self.epochs = self.config.epochs
        self.criterion = nn.CrossEntropyLoss()
        if self.is_training_mode:
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=5e-4)

            # init scheduler
            total_steps = len(self.dataloaders['train']) * self.epochs
            pct_start = 10 / total_steps
            final_div_factor = self.lr / 25 / 1e-6
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=pct_start, final_div_factor=final_div_factor)


    def _init_model(self, config, mode):
        def _resume_model(resume_path, device, is_rank_zero):
            try:
                checkpoints = torch.load(resume_path, map_location=device)
            except RuntimeError:
                LOGGER.warning(colorstr('yellow', 'cannot be loaded to MPS, loaded to CPU'))
                checkpoints = torch.load(resume_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoints['model'])
            del checkpoints
            torch.cuda.empty_cache()
            gc.collect()
            if is_rank_zero:
                LOGGER.info(f'Resumed model: {colorstr(resume_path)}')
            return model

        # init models
        do_resume = mode == 'resume' or (mode == 'validation' and self.resume_path)
        model = get_model(config, self.device)

        # resume model
        if do_resume:
            model = _resume_model(self.resume_path, self.device, config.is_rank_zero)

        # init ddp
        if self.is_ddp:
            torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device])
        
        return model


    def do_train(self):
        self.train_cur_step = -1
        self.train_time_start = time.time()
        
        if self.is_rank_zero:
            LOGGER.info(f'\nUsing {self.dataloaders["train"].num_workers * (self.world_size or 1)} dataloader workers\n'
                        f"Logging results to {colorstr('bold', self.save_dir)}\n"
                        f'Starting training for {self.epochs} epochs...\n')
        
        if self.is_ddp:
            dist.barrier()

        for epoch in range(self.epochs):
            start = time.time()
            self.epoch = epoch

            if self.is_rank_zero:
                LOGGER.info('-'*100)

            for phase in self.modes:
                if self.is_rank_zero:
                    LOGGER.info('Phase: {}'.format(phase))

                if phase == 'train':
                    self.epoch_train(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()
                else:
                    self.epoch_validate(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()
            
            # clears GPU vRAM at end of epoch, can help with out of memory errors
            torch.cuda.empty_cache()
            gc.collect()

            # Early Stopping
            if self.is_ddp:  # if DDP training
                broadcast_list = [self.stop if self.is_rank_zero else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if not self.is_rank_zero:
                    self.stop = broadcast_list[0]
            
            if self.stop:
                break  # must break all DDP ranks

            if self.is_rank_zero:
                LOGGER.info(f"\nepoch {epoch+1} time: {time.time() - start} s\n\n\n")

        if RANK in (-1, 0) and self.is_rank_zero:
            LOGGER.info(f'\n{epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            

    def epoch_train(
            self,
            phase: str,
            epoch: int
        ):
        self.encoder.train()
        self.decoder.train()
        train_loader = self.dataloaders[phase]
        nb = len(train_loader)

        if self.is_ddp:
            train_loader.sampler.set_epoch(epoch)

        # init progress bar
        if RANK in (-1, 0):
            logging_header = ['CE Loss']
            pbar = init_progress_bar(train_loader, self.is_rank_zero, logging_header, nb)

        for i, (src, trg, mask) in pbar:
            self.train_cur_step += 1
            batch_size = src.size(0)
            src, trg = src.to(self.device), trg.to(self.device)
            if self.config.use_attention:
                mask = mask.to(self.device)
            
            teacher_forcing = True if random.random() < self.config.teacher_forcing_ratio else False
            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()

            enc_output, hidden = self.encoder(src)
            
            # iteration one by one due to Badanau attention mechanism
            decoder_all_output = []
            for j in range(self.max_len):
                if teacher_forcing or j == 0:
                    trg_word = trg[:, j].unsqueeze(1)
                    dec_output, hidden, _ = self.decoder(trg_word, hidden, enc_output, mask)
                    decoder_all_output.append(dec_output)
                else:
                    trg_word = torch.argmax(dec_output, dim=-1)
                    dec_output, hidden, _ = self.decoder(trg_word.detach(), hidden, enc_output, mask)
                    decoder_all_output.append(dec_output)

            decoder_all_output = torch.cat(decoder_all_output, dim=1)
            loss = self.criterion(decoder_all_output[:, :-1, :].reshape(-1, decoder_all_output.size(-1)), trg[:, 1:].reshape(-1))

            loss.backward()
            self.enc_optimizer.step()
            self.dec_optimizer.step()

            if self.is_rank_zero:
                self.training_logger.update(
                    phase, 
                    epoch + 1,
                    self.train_cur_step,
                    batch_size, 
                    **{'train_loss': loss.item()},
                )
                loss_log = [loss.item()]
                msg = tuple([f'{epoch + 1}/{self.epochs}'] + loss_log)
                pbar.set_description(('%15s' * 1 + '%15.4g' * len(loss_log)) % msg)
            
        # upadate logs
        if self.is_rank_zero:
            self.training_logger.update_phase_end(phase, printing=True)
        
        
    def epoch_validate(
            self,
            phase: str,
            epoch: int,
            is_training_now=True
        ):
        def _init_log_data_for_vis():
            data4vis = {'src': [], 'trg': [], 'pred': []}
            if self.config.use_attention:
                data4vis.update({'score': []})
            return data4vis

        def _append_data_for_vis(**kwargs):
            for k, v in kwargs.items():
                if isinstance(v, list):
                    self.data4vis[k].extend(v)
                else: 
                    self.data4vis[k].append(v)

        with torch.no_grad():
            if self.is_rank_zero:
                if not is_training_now:
                    self.data4vis = _init_log_data_for_vis()

                val_loader = self.dataloaders[phase]
                nb = len(val_loader)
                logging_header = ['CE Loss'] + self.config.metrics
                pbar = init_progress_bar(val_loader, self.is_rank_zero, logging_header, nb)

                self.encoder.eval()
                self.decoder.eval()

                for i, (src, trg, mask) in pbar:
                    batch_size = src.size(0)
                    src, trg = src.to(self.device), trg.to(self.device)
                    if self.config.use_attention:
                        mask = mask.to(self.device)

                    sources = [self.tokenizers[0].decode(s.tolist()) for s in src]
                    targets = [self.tokenizers[1].decode(t.tolist()) for t in trg]
                    targets4metrics = [self.tokenizers[1].decode(t[1:].tolist()) for t in trg]
                    
                    enc_output, hidden = self.encoder(src)
                    predictions, score, loss = self.decoder.batch_inference(
                        start_tokens=trg[:, 0], 
                        enc_output=enc_output,
                        hidden=hidden,
                        mask=mask,
                        max_len=self.max_len,
                        tokenizer=self.tokenizers[1],
                        loss_func=self.criterion,
                        target=trg
                    )
                
                    metric_results = self.metric_evaluation(loss, predictions, targets4metrics)

                    self.training_logger.update(
                        phase, 
                        epoch, 
                        self.train_cur_step if is_training_now else 0, 
                        batch_size, 
                        **{'validation_loss': loss.item()},
                        **metric_results
                    )

                    # logging
                    loss_log = [loss.item()]
                    msg = tuple([f'{epoch+1}/{self.epochs}'] + loss_log + [metric_results[k] for k in self.metrics])
                    pbar.set_description(('%15s' + '%15.4g' * (len(loss_log) + len(self.metrics))) % msg)

                    ids = random.sample(range(batch_size), min(batch_size, self.config.prediction_print_n))
                    for id in ids:
                        print_samples(' '.join(sources[id].split()[1:]), targets4metrics[id], predictions[id])

                    if not is_training_now:
                        _append_data_for_vis(
                            **{'src': sources,
                               'trg': targets,
                               'pred': predictions}
                        )
                        if self.config.use_attention:
                            _append_data_for_vis(**{'score': score.detach().cpu()})

                # upadate logs and save model
                self.training_logger.update_phase_end(phase, printing=True)
                if is_training_now:
                    self.training_logger.save_model(self.wdir, {'encoder': self.encoder, 'decoder': self.decoder})
                    self.training_logger.save_logs(self.save_dir)

                    high_fitness = self.training_logger.model_manager.best_higher
                    low_fitness = self.training_logger.model_manager.best_lower
                    self.stop = self.stopper(epoch + 1, high=high_fitness, low=low_fitness)

    
    def metric_evaluation(self, loss, response_pred, response_gt):
        metric_results = {k: 0 for k in self.metrics}
        for m in self.metrics:
            if m == 'ppl':
                metric_results[m] = self.evaluator.cal_ppl(loss.item())
            elif m == 'bleu2':
                metric_results[m] = self.evaluator.cal_bleu_score(response_pred, response_gt, n=2)
            elif m == 'bleu4':
                metric_results[m] = self.evaluator.cal_bleu_score(response_pred, response_gt, n=4)
            elif m == 'nist2':
                metric_results[m] = self.evaluator.cal_nist_score(response_pred, response_gt, n=2)
            elif m == 'nist4':
                metric_results[m] = self.evaluator.cal_nist_score(response_pred, response_gt, n=4)
            else:
                LOGGER.warning(f'{colorstr("red", "Invalid key")}: {m}')
        
        return metric_results
    

    def vis_attention(self, phase, result_num):
        if result_num > len(self.dataloaders[phase].dataset):
            LOGGER.info(colorstr('red', 'The number of results that you want to see are larger than total test set'))
            sys.exit()

        # validation
        self.epoch_validate(phase, 0, False)
        if self.config.use_attention:
            vis_save_dir = os.path.join(self.config.save_dir, 'vis_outputs') 
            os.makedirs(vis_save_dir, exist_ok=True)
            visualize_attn(self.data4vis, self.tokenizers, result_num, vis_save_dir)
        else:
            LOGGER.warning(colorstr('yellow', 'Your model does not have attention module..'))


    def inference(self, query):
        query, mask = make_inference_data(query, self.tokenizers[0], self.max_len)

        with torch.no_grad():
            query = query.to(self.device)
            if self.config.use_attention:
                mask = mask.to(self.device)
            self.encoder.eval()
            self.decoder.eval()

            enc_output, hidden = self.encoder(query)
            decoder_all_output, decoder_bos = [], torch.LongTensor([[self.tokenizers[1].bos_token_id]]).to(self.device)
            for j in range(self.max_len):
                if j == 0:
                    dec_output, hidden, _ = self.decoder(decoder_bos, hidden, enc_output, mask)
                    decoder_all_output.append(dec_output)
                else:
                    trg_word = torch.argmax(dec_output, dim=-1)
                    dec_output, hidden, _ = self.decoder(trg_word.detach(), hidden, enc_output, mask)
                    decoder_all_output.append(dec_output)
            decoder_all_output = torch.cat(decoder_all_output, dim=1)
            output = self.tokenizers[1].decode(torch.argmax(decoder_all_output.detach().cpu(), dim=-1)[0].tolist())
        
        if output.split()[-1] == self.tokenizers[1].eos_token:
            return ' '.join(output.split()[:-1])
        return output  
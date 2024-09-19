import gc
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
            final_div_factor = self.config.lr / 25 / 1e-6
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.config.lr, total_steps=total_steps, pct_start=pct_start, final_div_factor=final_div_factor)


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
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device])
        
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
        self.model.train()
        train_loader = self.dataloaders[phase]
        nb = len(train_loader)

        if self.is_ddp:
            train_loader.sampler.set_epoch(epoch)

        # init progress bar
        if RANK in (-1, 0):
            logging_header = ['CE Loss', 'Accuracy', 'lr']
            pbar = init_progress_bar(train_loader, self.is_rank_zero, logging_header, nb)

        for i, batch in pbar:
            self.train_cur_step += 1
            cur_lr = self.optimizer.param_groups[0]['lr']
            batch_size = batch.y.size(0)
            batch = batch.to(self.device)
            output, feature = self.model(batch)
            loss = self.criterion(output, batch.y)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_acc = torch.sum(torch.argmax(output, dim=-1).detach().cpu() == batch.y.detach().cpu()) / batch_size

            if self.is_rank_zero:
                self.training_logger.update(
                    phase, 
                    epoch + 1,
                    self.train_cur_step,
                    batch_size, 
                    **{'train_loss': loss.item(), 'lr': cur_lr},
                    **{'train_acc': train_acc.item()}
                )
                loss_log = [loss.item(), train_acc.item(), cur_lr]
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
            data4vis = {'feature': [], 'pred': [], 'label': []}
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
                logging_header = ['CE Loss', 'Accuracy']
                pbar = init_progress_bar(val_loader, self.is_rank_zero, logging_header, nb)

                self.model.eval()

                for i, batch in pbar:
                    batch_size = batch.y.size(0)
                    batch = batch.to(self.device)
                    output, feature = self.model(batch)
                    loss = self.criterion(output, batch.y)

                    pred = torch.argmax(output, dim=-1).detach().cpu()
                    label = batch.y.detach().cpu()
                    val_acc = torch.sum(pred == label) / batch_size

                    self.training_logger.update(
                        phase, 
                        epoch + 1,
                        self.train_cur_step if is_training_now else 0, 
                        batch_size, 
                        **{'validation_loss': loss.item()},
                        **{'validation_acc': val_acc.item()}
                    )

                    # logging
                    loss_log = [loss.item(), val_acc.item()]
                    msg = tuple([f'{epoch + 1}/{self.epochs}'] + loss_log)
                    pbar.set_description(('%15s' * 1 + '%15.4g' * len(loss_log)) % msg)

                    if not is_training_now:
                        _append_data_for_vis(
                            **{'feature': feature.detach().cpu(),
                               'pred': pred,
                               'label': label}
                        )

                # upadate logs and save model
                self.training_logger.update_phase_end(phase, printing=True)
                if is_training_now:
                    model = self.model.module if self.is_ddp else self.model
                    self.training_logger.save_model(self.wdir, model)
                    self.training_logger.save_logs(self.save_dir)


    def tsne_visualization(self, phase):
        self.epoch_validate(phase, 0, False)
            
        # feature visualization
        vis_save_dir = os.path.join(self.config.save_dir, 'vis_outputs') 
        os.makedirs(vis_save_dir, exist_ok=True)

        pred = torch.cat(self.data4vis['pred']).numpy()
        label = torch.cat(self.data4vis['label']).numpy()
        output = torch.cat(self.data4vis['feature']).numpy()
        
        tsne = TSNE()
        x_test_2D = tsne.fit_transform(output)
        x_test_2D = (x_test_2D - x_test_2D.min())/(x_test_2D.max() - x_test_2D.min())

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plt.setp(ax, xticks=[], yticks=[])
        
        ax[0].scatter(x_test_2D[:, 0], x_test_2D[:, 1], s=10, cmap='tab10', c=label)
        ax[0].set_title("GT visualization")

        ax[1].scatter(x_test_2D[:, 0], x_test_2D[:, 1], s=10, cmap='tab10', c=pred)
        ax[1].set_title("Pred visualization")

        fig.tight_layout()
        plt.savefig(os.path.join(vis_save_dir, 'tsne_vis.png'))
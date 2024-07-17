import os

import torch
from torch.utils.data import distributed
from torch_geometric.loader import DataLoader

from models import GCN
from utils import LOGGER, RANK, colorstr
from utils.data_utils import load_data, seed_worker

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def get_model(config, device):
    model = GCN(config).to(device)
    return model


def build_dataset(config, modes):
    if config.tu_dataset_train:
        dataset = load_data(config)
        config.num_node_features = dataset.num_node_features
        config.cls_num = dataset.num_classes
        tmp_dsets = {
            'train': dataset[:int(len(dataset) * 0.7)],
            'validation': dataset[int(len(dataset) * 0.7):int(len(dataset) * 0.85)],
            'test': dataset[int(len(dataset) * 0.85):]
        }
        dataset_dict = {mode: tmp_dsets[mode] for mode in modes}
    else:
        LOGGER.warning(colorstr('yellow', 'You have to implement data pre-processing code..'))
        raise NotImplementedError
    return dataset_dict


def build_dataloader(dataset, batch, workers, shuffle=True, is_ddp=False):
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if not is_ddp else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)


def get_data_loader(config, modes, is_ddp=False):
    datasets = build_dataset(config, modes)
    dataloaders = {m: build_dataloader(datasets[m], 
                                       config.batch_size, 
                                       config.workers, 
                                       shuffle=(m == 'train'), 
                                       is_ddp=is_ddp) for m in modes}

    return dataloaders
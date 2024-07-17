import random
import numpy as np

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset

from utils import LOGGER, colorstr



def load_data(config):
    if config.tu_dataset_train:
        dataset = TUDataset(root=config.tu_dataset.path, name='COLLAB')
        dataset.transform = make_onehot_node_feature(dataset)
        dataset = dataset.shuffle()
    else:
        LOGGER.error(colorstr('red', 'You have to implement data pre-processing code..'))
        raise NotImplementedError
    return dataset


def make_onehot_node_feature(dataset):
    max_degree = 0                     
    for data in dataset:
        M_deg = degree(data.edge_index[0], dtype=torch.long).max().item()
        max_degree = max(max_degree, M_deg)

    return T.OneHotDegree(max_degree)


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

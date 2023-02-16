import torch
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset




"""
common utils
"""
def save_checkpoint(file, model, optimizer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')


def load_data(base_path):
    torch.manual_seed(999)
    dataset = TUDataset(root=base_path + 'data/', name='COLLAB')
    dataset.transform = make_onehot_node_feature(dataset)
    dataset = dataset.shuffle()
    return dataset


def make_onehot_node_feature(dataset):
    max_degree = 0                     
    for data in dataset:
        M_deg = degree(data.edge_index[0], dtype=torch.long).max().item()
        max_degree = max(max_degree, M_deg)

    return T.OneHotDegree(max_degree)
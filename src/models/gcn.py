import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool



class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout
        self.num_node_features = config.num_node_features
        self.cls_num = config.cls_num

        self.gcn1 = GCNConv(self.num_node_features, self.hidden_dim)
        self.gcn2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.cls_num)

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)


    def forward(self, data):
        node_feature, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN1
        node_feature = self.gcn1(node_feature, edge_index)
        node_feature = self.relu(node_feature)
        node_feature = self.dropout_layer(node_feature)

        node_feature = self.gcn2(node_feature, edge_index)
        node_feature = self.relu(node_feature)
        node_feature = self.dropout_layer(node_feature)

        # readout
        node_feature = global_mean_pool(node_feature, batch)

        # fc
        output = self.fc(node_feature)

        return output, node_feature
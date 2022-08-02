import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features=114,n_filters=32, embed_dim=128, output_dim=64, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        D1_nn1 = Sequential(Linear(num_features, 198), ReLU(), Linear(198, 198))
        self.D1_conv1 = GINConv(D1_nn1)
        self.D1_bn1 = torch.nn.BatchNorm1d(198)

        D1_nn2 = Sequential(Linear(198, 64), ReLU(), Linear(64, 64))
        self.D1_conv2 = GINConv(D1_nn2)
        self.D1_bn2 = torch.nn.BatchNorm1d(64)

        D1_nn3 = Sequential(Linear(64, 32), ReLU(), Linear(32, 32))
        self.D1_conv3 = GINConv(D1_nn3)
        self.D1_bn3 = torch.nn.BatchNorm1d(32)


        # combined layers
        self.fc1 = nn.Linear(dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x1, edge_index_1, batch1 = data.x.float(), data.edge_index, data.batch
        x1 = F.relu(self.D1_conv1(x1, edge_index_1))
        x1 = self.D1_bn1(x1)
        x1 = F.relu(self.D1_conv2(x1, edge_index_1))
        x1 = self.D1_bn2(x1)
        x1 = F.relu(self.D1_conv3(x1, edge_index_1))
        x1 = self.D1_bn3(x1)
        x1 = global_add_pool(x1, batch1)
       
        # add some dense layers
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.fc2(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        out = self.out(x1)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LeakyReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model

class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features=114, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=64, dropout=0.3):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.D1_conv1 = GATConv(num_features, num_features, heads=10)
        self.D1_conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.D1_fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1000)
        self.D1_fc_g2 = torch.nn.Linear(1000, output_dim)
        
        
        self.D2_conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.D2_conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.D2_fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1000)
        self.D2_fc_g2 = torch.nn.Linear(1000, output_dim)
        
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.fc1_xt = nn.Linear(1000,128)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        
        self.out = nn.Linear(32, self.n_output)       # n_output = 1 for regression task

    def forward(self, data1,data2):
        x1, edge_index_1, batch1 = data1.x, data1.edge_index, data1.batch
        #Forward feed for drug1
        x1 = self.D1_conv1(x1, edge_index_1)
        x1 = self.relu(x1)
        x1 = self.D1_conv2(x1, edge_index_1)
        x1 = self.relu(x1)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x1 = torch.cat([gmp(x1, batch1), gap(x1, batch1)], dim=1)
        x1 = self.relu(self.D1_fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.D1_fc_g2(x1)
        
        x2, edge_index_2, batch2 = data2.x, data2.edge_index, data2.batch
        #Forward feed for drug2
        x2 = self.D2_conv1(x2, edge_index_2)
        x2 = self.relu(x2)
        x2 = self.D2_conv2(x2, edge_index_2)
        x2 = self.relu(x2)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x2 = torch.cat([gmp(x2, batch2), gap(x2, batch2)], dim=1)
        x2 = self.relu(self.D2_fc_g1(x2))
        x2 = self.dropout(x2)
        x2 = self.D2_fc_g2(x2)
        xd = torch.cat((x1, x2), 1)

        # get protein input
        xt = data1.target
        xt=xt.view(-1,1000)
        xt = self.fc1_xt(xt)
        
        # concat
        xc = torch.cat((xd, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
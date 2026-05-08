# Code for computing the GAT model in a single graph

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels)
        self.gat2 = GATConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch):

        # First GAT layer
        x = self.gat1(x, edge_index)
        x = F.relu(x)

        # Second GAT layer
        x = self.gat2(x, edge_index)
        x = F.relu(x)

        # Global mean pooling
        return global_mean_pool(x, batch) # [batch_size = 1, hidden_channels]
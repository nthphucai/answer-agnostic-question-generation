import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_conv import TransformerConv


# define a GCN model with edge labels
class GCNWithEdgeLabels(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNWithEdgeLabels, self).__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels)

        self.Vx = torch.nn.Linear(in_channels, out_channels)
        self.Vh = torch.nn.Linear(hidden_channels, out_channels)

        self.act = nn.GELU()

    def forward(self, x, edge_index, edge_attr):
        # apply first GCN layer
        x_graph = self.conv1(x, edge_index, edge_attr)
        x_graph = self.act(x_graph)
        x_graph = F.dropout(x_graph, training=self.training)

        # # apply second GCN layer
        x_graph = self.conv2(x_graph, edge_index, edge_attr)
        x_graph = self.act(x_graph)
        x_graph = self.act(x_graph)

        x_graph = F.dropout(x_graph, training=self.training)

        x = self.Vx(x)  # [64, 512] -> [64, 64]
        x_graph = self.Vh(x_graph)  # [64, 128] -> [64, 64]
        out = F.gelu(x_graph + x)
        return out

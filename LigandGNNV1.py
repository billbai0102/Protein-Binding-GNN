import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class LigandGNNV1(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.conv1 = GCNConv(in_features, in_features * 2)
        self.conv2 = GCNConv(in_features * 2, out_classes)

    def forward(self, data):
        x = data.x
        x = self.conv1(x, data.edge_index)
        x = F.relu(x)
        out = self.conv2(x, data.edge_index)

        return out

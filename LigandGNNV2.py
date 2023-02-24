from torch.nn import functional as F
import torch
from torch_geometric.nn import Linear, GENConv, LayerNorm, DeepGCNLayer
from torch.nn import ReLU


class LigandGNNV2(torch.nn.Module):
    """
    Implementation of DeeperGCN
    """
    def __init__(self, hidden_channels, num_layers):
        super().__init__()
        self.node_encoder = Linear(1070, hidden_channels)
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.node_encoder(x)
        x = self.layers[0].conv(x, edge_index)
        for layer in self.layers[1:]:
            x = layer(x, edge_index)
        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.2, training=self.training)

        return self.lin(x)
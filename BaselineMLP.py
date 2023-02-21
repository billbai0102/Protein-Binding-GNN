import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """
    Classifies nodes, no edges or graph structure
    """
    def __int__(self, in_size, out_size):
        super.__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, in_size * 4),
            nn.ReLU(),
            nn.Linear(in_size * 4, in_size * 2),
            nn.ReLU(),
            nn.Linear(in_size * 2, out_size)
        )

    def forward(self, data):
        x = data.x
        out = self.layers(x)
        return out

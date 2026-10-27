import torch.nn as nn
from .layers import PixelNorm


class MappingNetwork(nn.Module):
    """f: Z -> W. Original paper uses 8 layers; 6 here for a faster run."""
    def __init__(self, z_dim=256, w_dim=256, n_layers=6):
        super().__init__()
        layers = [PixelNorm()]
        in_dim = z_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = w_dim
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)

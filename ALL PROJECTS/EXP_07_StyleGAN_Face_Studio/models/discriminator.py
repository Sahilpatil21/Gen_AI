import torch.nn as nn


class Discriminator(nn.Module):
    """Plain CNN discriminator (close to a DCGAN discriminator; only the generator is style-based)."""
    def __init__(self, img_channels=3, features=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, features, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),                                    # 64 -> 32
            nn.Conv2d(features, features * 2, 4, 2, 1), nn.BatchNorm2d(features * 2), nn.LeakyReLU(0.2, inplace=True),      # 32 -> 16
            nn.Conv2d(features * 2, features * 4, 4, 2, 1), nn.BatchNorm2d(features * 4), nn.LeakyReLU(0.2, inplace=True),  # 16 -> 8
            nn.Conv2d(features * 4, features * 8, 4, 2, 1), nn.BatchNorm2d(features * 8), nn.LeakyReLU(0.2, inplace=True),  # 8 -> 4
            nn.Conv2d(features * 8, 1, 4, 1, 0),                                                                             # 4 -> 1
        )

    def forward(self, x):
        return self.model(x).view(-1)

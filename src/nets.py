# Source: https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial9/AE_CIFAR10.ipynb

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, base_channel_size: int = 64) -> None:
        """
        Args:
           latent_dim : Dimensionality of latent representation z
           base_channel_size : Number of channels we use in the first 
           convolutional layers. Deeper layers might use a duplicate of it.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base_channel_size, kernel_size=3,
                      padding=1, stride=2),  # 32x32 => 16x16
            nn.GELU(),
            nn.Conv2d(base_channel_size, base_channel_size,
                      kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channel_size, 2 * base_channel_size, kernel_size=3,
                      padding=1, stride=2),  # 16x16 => 8x8
            nn.GELU(),
            nn.Conv2d(2 * base_channel_size, 2 * base_channel_size,
                      kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * base_channel_size, 2 * base_channel_size,
                      kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            nn.GELU(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * base_channel_size, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, base_channel_size: int = 64) -> None:
        """
        Args:
           latent_dim : Dimensionality of latent representation z
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
        """
        super().__init__()
        base_channel_size = 64
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * base_channel_size), nn.GELU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * base_channel_size, 2 * base_channel_size, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            nn.GELU(),
            nn.Conv2d(2 * base_channel_size, 2 * \
                      base_channel_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(2 * base_channel_size, base_channel_size, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            nn.GELU(),
            nn.Conv2d(base_channel_size, base_channel_size,
                      kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(
                base_channel_size, 3, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=2, features=64):
        super().__init__()

        self.model = nn.Sequential(
            self._block(in_channels, features, 4, 2, 1),
            self._block(features, features * 2, 4, 2, 1),
            self._block(features * 2, features * 4, 4, 2, 1),
            self._block(features * 4, features * 8, 4, 1, 1),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))


if __name__ == "__main__":
    x = torch.randn(8, 2, 512, 512)
    y = PatchGANDiscriminator(2, 32)(x)
    print(y.size())

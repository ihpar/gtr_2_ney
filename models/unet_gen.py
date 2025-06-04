import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super().__init__()

        self.encoder = nn.ModuleList([
            self._block(in_channels, features,
                        kernel_size=4, stride=2, padding=1),
            self._block(features, features * 2, 4, 2, 1),
            self._block(features * 2, features * 4, 4, 2, 1),
            self._block(features * 4, features * 8, 4, 2,  1),
            self._block(features * 8, features * 8, 4, 2, 1),
        ])

        self.decoder = nn.ModuleList([
            self._upblock(features * 8, features * 8, 4, 2, 1),
            self._upblock(features * 8 * 2, features * 4, 4, 2, 1),
            self._upblock(features * 4 * 2, features * 2, 4, 2, 1),
            self._upblock(features * 2 * 2, features, 4, 2, 1),
        ])

        self.final_layer = nn.ConvTranspose2d(
            features * 2, out_channels, kernel_size=4, stride=2, padding=1)
        self.act = nn.Sigmoid()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def _upblock(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        encodings = []
        for layer in self.encoder:
            x = layer(x)
            encodings.append(x)

        encodings = encodings[:-1][::-1]
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            x = torch.cat([x, encodings[i]], dim=1)

        return self.act(self.final_layer(x))


if __name__ == "__main__":
    x = torch.randn(8, 1, 512, 512)
    y = UNetGenerator(1, 1, 32)(x)
    print(y.size())

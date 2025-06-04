import torch
import torch.nn as nn
import torch.nn.functional as F


class SubpixelUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels *
                              (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        return self.pixel_shuffle(x)


class UNetWithSubpixel(nn.Module):
    def __init__(self, in_features=1, features=32):
        super().__init__()

        # Contracting Path
        self.enc1 = self.conv_block(in_features, features)
        self.enc2 = self.conv_block(features, features * 2)
        self.enc3 = self.conv_block(features * 2, features * 4)
        self.enc4 = self.conv_block(features * 4, features * 8)

        # Bottleneck
        self.bottleneck = self.conv_block(features * 8, features * 16)

        # Expanding Path with Subpixel Convolutions
        self.upconv4 = SubpixelUpsample(features * 16, features * 8)
        self.dec4 = self.conv_block(features * 16, features * 8)
        self.upconv3 = SubpixelUpsample(features * 8, features * 4)
        self.dec3 = self.conv_block(features * 8, features * 4)
        self.upconv2 = SubpixelUpsample(features * 4, features * 2)
        self.dec2 = self.conv_block(features * 4, features * 2)
        self.upconv1 = SubpixelUpsample(features * 2, features)
        self.dec1 = self.conv_block(features * 2, features)

        # Output layer
        self.final = nn.Conv2d(features, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat((dec4, enc4), dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1))

        # Output
        return torch.sigmoid(self.final(dec1))


if __name__ == "__main__":
    model = UNetWithSubpixel(features=32)
    input_tensor = torch.rand((8, 1, 512, 512))
    output = model(input_tensor)
    print(output.shape)

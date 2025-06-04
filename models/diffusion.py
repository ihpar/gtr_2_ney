import torch
import torch.nn as nn

import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalDiffusionUNet1D(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, base_channels=64, time_channels=128):
        super().__init__()

        # Time Embedding Layer (for conditioning on time step)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_channels),
            nn.ReLU(),
            nn.Linear(time_channels, time_channels)
        )

        # Encoder (Downsampling)
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, base_channels,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(base_channels, base_channels,
                          kernel_size=3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(base_channels, base_channels * 2,
                          kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(base_channels * 2, base_channels * 4,
                          kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
        ])

        # Decoder (Upsampling)
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(
                    base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d(
                    base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            )
        ])

        # Skip connection layer to match channels
        self.skip_layers = nn.ModuleList([
            nn.Conv1d(base_channels * 4, base_channels * 2, kernel_size=1),
            nn.Conv1d(base_channels * 2, base_channels, kernel_size=1)
        ])

        # Final Convolution to match output channels (flute sound)
        self.conv_final = nn.Conv1d(
            base_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        Forward pass for the Conditional Diffusion Model.

        Args:
            x: Noisy guitar input audio, shape: [batch_size, 1, 4800].
            t: Time step conditioning, shape: [batch_size].

        Returns:
            Predicted flute audio, shape: [batch_size, 1, 4800].
        """
        # Time Embedding
        t_emb = self.time_mlp(t.unsqueeze(1).float())  # Embedding time step t
        t_emb = t_emb.unsqueeze(-1)  # Shape: [batch_size, time_channels, 1]

        # Encoder pass
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        # Decoder pass with skip connections
        for layer, skip, skip_layer in zip(self.decoder, reversed(skips), self.skip_layers):
            # Apply a convolution to the skip connection to match channel size
            skip = skip_layer(skip)

            # Upsample time embedding to match the current feature map size
            t_emb_up = F.interpolate(t_emb, size=skip.size()[
                                     2:], mode='linear', align_corners=False)
            print(x.size(), t_emb_up.size(), skip.size())
            x = layer(x + t_emb_up + skip)

        # Final output layer
        return self.conv_final(x)


if __name__ == "__main__":
    model = ConditionalDiffusionUNet1D(1, 1, 64)
    input = torch.rand(32, 1, 4800) * 2 - 1
    t = torch.randint(0, 1000, (input.size(0),))
    output = model(input, t)
    print(output.size())

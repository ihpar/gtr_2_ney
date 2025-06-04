import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3,
                      stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(channels),
            nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=2,
                      stride=1, padding=1, dilation=2),
            nn.BatchNorm1d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator_HiFi_Dil(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7),
            nn.Tanh(),

            # Adding Multi-Scale Residual Blocks
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),

            # Dilated Convolutions for Long-Range Dependencies
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.BatchNorm1d(128),
            nn.Tanh(),

            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.BatchNorm1d(64),
            nn.Tanh(),

            # Final layer to project back to 1-channel audio
            nn.Conv1d(64, 1, kernel_size=11, stride=1, padding=7)
        )

    def forward(self, x):
        return self.net(x)


class Discriminator_HiFi_Dil(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.2),

            # Adding Residual and Multi-Scale Convolutional Blocks
            ResidualBlock(64),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            # Final convolutional layer to project to a single scalar
            nn.Conv1d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # Define adversarial loss
    adversarial_criterion = nn.BCEWithLogitsLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate models and move them to the device
    generator = Generator_HiFi_Dil().to(device)
    discriminator = Discriminator_HiFi_Dil().to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    # Dummy input (audio) tensor
    batch_size = 4
    n_samples = 32704
    real_audio = torch.randn(batch_size, 1, n_samples).to(device)

    # Training loop placeholder
    for epoch in range(1):
        optimizer_d.zero_grad()

        generated_audio = generator(real_audio)
        real_out = discriminator(real_audio)
        fake_out = discriminator(generated_audio.detach())
        # print(generated_audio.size(), real_out.size(), fake_out.size())

        d_loss = adversarial_criterion(
            real_out - fake_out, torch.ones_like(real_out))
        d_loss.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()

        gen_out = discriminator(generated_audio)
        g_loss = adversarial_criterion(gen_out, torch.ones_like(gen_out))

        g_loss.backward()
        optimizer_g.step()

        print(f"Epoch [{epoch}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

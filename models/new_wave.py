import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Causal convolutions
        self.conv_filter = nn.Conv1d(
            residual_channels, residual_channels, kernel_size, dilation=dilation
        )
        self.conv_gate = nn.Conv1d(
            residual_channels, residual_channels, kernel_size, dilation=dilation
        )
        self.conv_residual = nn.Conv1d(
            residual_channels, residual_channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(
            residual_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        # Causal padding to ensure the output length matches input length
        padding = (self.kernel_size - 1) * self.dilation
        # Padding only on the left for causality
        x_padded = F.pad(x, (padding, 0))

        # Dilated convolutions
        filter_output = torch.tanh(self.conv_filter(x_padded))
        gate_output = torch.sigmoid(self.conv_gate(x_padded))
        gated_output = filter_output * gate_output

        # Residual and skip connections
        residual = self.conv_residual(gated_output)
        skip = self.conv_skip(gated_output)

        # Add the residual to the input (residual connection)
        return (x + residual), skip


class WaveNet(nn.Module):
    def __init__(self, residual_channels, skip_channels, num_blocks, num_layers, kernel_size):
        super().__init__()
        self.input_conv = nn.Conv1d(1, residual_channels, kernel_size=1)

        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList()
        for b in range(num_blocks):
            for l in range(num_layers):
                dilation = 2 ** l
                self.residual_blocks.append(
                    ResidualBlock(residual_channels,
                                  skip_channels,
                                  kernel_size,
                                  dilation)
                )

        # Final layers
        self.conv_out1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_out2 = nn.Conv1d(skip_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.input_conv(x)
        skip_connections = []

        # Pass through residual blocks
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        # Sum skip connections (all tensors now have consistent sizes)
        x = torch.sum(torch.stack(skip_connections), dim=0)
        x = F.relu(x)
        x = F.relu(self.conv_out1(x))
        x = torch.tanh(self.conv_out2(x))  # Output in [-1, 1]

        return x

# Sample Dataset


class AudioDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs  # Shape: [N, 32704]
        self.targets = targets  # Shape: [N, 32704]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Training Function


def train_wavenet(model, dataloader, optimizer, criterion, epochs, device):
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs = inputs.to(device).unsqueeze(1)  # Add channel dimension
            targets = targets.to(device).unsqueeze(1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    # Dummy Data for Training
    num_samples = 4
    input_audio = torch.rand(num_samples, 32704) * 2 - \
        1  # Random audio in [-1, 1]
    output_audio = torch.rand(num_samples, 32704) * 2 - 1

    # Dataset and DataLoader
    dataset = AudioDataset(input_audio, output_audio)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize Model, Loss, Optimizer
    model = WaveNet(residual_channels=64, skip_channels=128,
                    num_blocks=2, num_layers=10, kernel_size=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_wavenet(model, dataloader, optimizer,
                  criterion, epochs=1, device=device)

import torch
import torch.nn as nn


class PhaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        y_hat = y_hat * 2.0 * torch.pi - torch.pi
        y = y * 2.0 * torch.pi - torch.pi
        phase_diff = y_hat - y
        phase_diff = torch.atan2(torch.sin(phase_diff),
                                 torch.cos(phase_diff))
        return torch.mean(phase_diff ** 2)


if __name__ == "__main__":
    predicted = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)
    criterion = PhaseLoss()
    loss = criterion(predicted, target)
    print(loss)

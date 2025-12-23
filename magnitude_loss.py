import torch
import torch.nn as nn


class MagnitudeLoss(nn.Module):
    def __init__(self, mini, maxi):
        super().__init__()
        self.mini = mini
        self.maxi = maxi

    def forward(self, y_hat, y):
        y_hat = y_hat * (self.maxi - self.mini) + self.mini
        y = y * (self.maxi - self.mini) + self.mini
        magnitude_diff = (y_hat - y) ** 2
        loss = torch.mean(magnitude_diff)

        return loss


if __name__ == "__main__":
    predicted = torch.rand(8, 1, 512, 512)
    target = torch.rand(8, 1, 512, 512)
    criterion = MagnitudeLoss(1, 10)
    loss = criterion(predicted, target)
    print(loss)

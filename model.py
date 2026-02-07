import torch
import torch.nn as nn

class MySuperResolutionModel(nn.Module):
    def __init__(self, upscale_factor=2):
        super(MySuperResolutionModel, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3 * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

# Usage example
model = MySuperResolutionModel(upscale_factor=2)
print(model)

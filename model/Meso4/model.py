import torch
import torch.nn as nn


class Meso4(nn.Module):
    """Meso-4 CNN (Afchar et al., 2018) reimplementation.

    Outputs raw logits of shape (B, 1); apply torch.sigmoid for probabilities.
    Logit convention: P(fake) = sigmoid(logit).
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=1)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=1)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, padding=1)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=4, padding=1)

        self.flatten = nn.Flatten()

        dummy_input = torch.zeros(1, 3, 256, 256)
        dummy_output = self._forward_conv(dummy_input)
        flattened_size = dummy_output.view(dummy_output.size(0), -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 16)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 1)

    def _forward_conv(self, x):
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv2(x)))
        x = self.pool3(self.bn3(self.conv3(x)))
        x = self.pool4(self.bn4(self.conv4(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = self.flatten(x)
        x = self.dropout(self.leaky_relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def build_model():
    return Meso4()

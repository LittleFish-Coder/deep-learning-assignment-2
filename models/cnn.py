import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary


class CNN(nn.Module):
    """
    The input image size is of shape (batch_size, 3, 256, 256)
    """

    def __init__(self, in_channels=3, num_classes=50):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(num_features=32)
        self.norm2 = nn.BatchNorm2d(num_features=64)
        self.norm3 = nn.BatchNorm2d(num_features=128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=128, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 1st convolutional layer
        x = self.conv1(x)  # x.shape = (batch_size, 32, 256, 256)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.pool(x)  # x.shape = (batch_size, 32, 128, 128)

        # 2nd convolutional layer
        x = self.conv2(x)  # x.shape = (batch_size, 64, 126, 126)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.pool(x)  # x.shape = (batch_size, 64, 63, 63)

        # 3rd convolutional layer
        x = self.conv3(x)  # x.shape = (batch_size, 128, 61, 61)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.pool(x)  # x.shape = (batch_size, 128, 30, 30)

        # Global average pooling
        x = self.global_avg_pool(x)  # x.shape = (batch_size, 128, 1, 1)

        # Flatten
        x = x.view(x.size(0), -1)  # x.shape = (batch_size, 128)

        # Fully connected layer
        x = self.fc1(x)  # x.shape = (batch_size, 256)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)  # x.shape = (batch_size, num_classes)

        # We don't apply softmax here because PyTorch's cross-entropy loss function implicitly applies softmax
        return x


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)

    summary(model, input_size=(3, 256, 256))

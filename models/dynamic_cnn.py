import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary


class DYNAMIC_CNN(nn.Module):
    """
    The input image size is of shape (batch_size, 3, 256, 256)
    """

    def __init__(self, in_channels=3, num_classes=50):
        super(DYNAMIC_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.norm1 = nn.BatchNorm2d(num_features=32)
        self.norm2 = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64 * 62 * 62, out_features=num_classes)

    def forward(self, x):
        # 1st convolutional layer
        x = self.conv1(x)  # x.shape = (batch_size, 32, 254, 254)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.pool(x)  # x.shape = (batch_size, 32, 127, 127)
        # 2nd convolutional layer
        x = self.conv2(x)  # x.shape = (batch_size, 64, 125, 125)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.pool(x)  # x.shape = (batch_size, 64, 62, 62)
        # Flatten P.S. x = x.view(x.size(0), -1) also works
        x = x.reshape(x.shape[0], -1)  # x.shape = (batch_size, 64 * 62 * 62)
        # Fully connected layer
        x = self.fc1(x)
        # we don't apply softmax here because pytorch's cross-entropy loss function implicitly applies softmax
        return x


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DYNAMIC_CNN().to(device)

    summary(model, input_size=(3, 256, 256))

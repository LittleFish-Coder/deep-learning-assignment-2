import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary


# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         batch_size, C, width, height = x.size()
#         proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(batch_size, -1, width * height)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = F.softmax(energy, dim=-1)
#         proj_value = self.value_conv(x).view(batch_size, -1, width * height)
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(batch_size, C, width, height)
#         out = self.gamma * out + x
#         return out


# class ATTENTION_CNN(nn.Module):
#     """
#     The input image size is of shape (batch_size, 3, 256, 256)
#     """

#     def __init__(self, method="self", in_channels=3, num_classes=50):
#         super(ATTENTION_CNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=0)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
#         self.norm1 = nn.BatchNorm2d(num_features=32)
#         self.norm2 = nn.BatchNorm2d(num_features=64)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(in_features=64 * 62 * 62, out_features=num_classes)
#         self.attention1 = SelfAttention(in_channels=32)
#         self.attention2 = SelfAttention(in_channels=64)

#     def forward(self, x):
#         # 1st convolutional layer
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = F.relu(x)
#         x = self.pool(x)
#         x = self.attention1(x)
#         # 2nd convolutional layer
#         x = self.conv2(x)
#         x = self.norm2(x)
#         x = F.relu(x)
#         x = self.pool(x)
#         x = self.attention2(x)
#         # Flatten P.S. x = x.view(x.size(0), -1) also works
#         x = x.reshape(x.shape[0], -1)
#         # Fully connected layer
#         x = self.fc1(x)
#         # we don't apply softmax here because pytorch's cross-entropy loss function implicitly applies softmax
#         return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual block
        self.use_identity = stride == 1 and in_channels == out_channels
        if not self.use_identity:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, 1, stride, padding=0)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        if not self.use_identity:
            identity = self.shortcut_conv(identity)
            identity = self.shortcut_bn(identity)

        x = x + identity
        x = self.relu(x)

        return x


class ConvNet(nn.Module):
    """
    The input image size is of shape (batch_size, 3, 256, 256)
    """

    def __init__(self, method="self", in_channels=3, num_classes=50):
        super(ConvNet, self).__init__()

        self.block1 = Block(in_channels, 32)
        self.block2 = Block(32, 64)
        self.block3 = Block(64, 128)
        self.block4 = Block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.sa = SpatialAttention()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=0)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        # self.bn1 = nn.BatchNorm2d(num_features=32)
        # self.bn2 = nn.BatchNorm2d(num_features=64)
        # self.sa = SpatialAttention()
        # self.relu = nn.ReLU(inplace=True)
        # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # 1st block
        x = self.block1(x)
        x = self.pool(x)

        # 2nd block
        x = self.block2(x)
        x = self.pool(x)

        # 3rd block
        x = self.block3(x)
        x = self.pool(x)

        # 4th block
        x = self.block4(x)
        x = self.pool(x)

        # attention mechanism
        x = self.sa(x) * x
        x = self.dropout(x)
        x = self.global_avg_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        # # 1st convolutional layer
        # x = self.conv1(x)
        # x = self.bn1(x)
        # ## attention mechanism
        # x = self.sa(x) * x
        # x = self.relu(x)

        # # 2nd convolutional layer
        # x = self.conv2(x)
        # x = self.bn2(x)
        # ## attention mechanism
        # x = self.sa(x) * x
        # x = self.relu(x)

        # # Global average pooling
        # x = self.global_avg_pool(x)

        # # Flatten
        # x = x.view(x.size(0), -1)

        # # Fully connected layer
        # x = self.fc(x)
        return x

import torch.nn.functional
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    """The Residual block of ResNet."""

    def __init__(self, input_channels, output_channels,
                 use_1x1conv=False):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels,
                               kernel_size=3, stride=2 if use_1x1conv else 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(output_channels, output_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels,
                                   kernel_size=1, stride=2, padding=0, bias=False)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class resnet34(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(resnet34, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            Residual(64, 64, use_1x1conv=True),
            Residual(64, 64),
            Residual(64, 64)
        )
        self.conv3 = nn.Sequential(
            Residual(64, 128, use_1x1conv=True),
            Residual(128, 128),
            Residual(128, 128),
            Residual(128, 128)
        )
        self.conv4 = nn.Sequential(
            Residual(128, 256, use_1x1conv=True),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256)
        )
        self.conv5 = nn.Sequential(
            Residual(256, 512, use_1x1conv=True),
            Residual(512, 512),
            Residual(512, 512)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def init_weights(self):
        for m in self.modules():  # 遍历每层
            if isinstance(m, nn.Conv2d):  # 若是卷积层
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):  # 批量归一化层
                nn.init.constant_(m.weight, 1)  # w初始为1
                nn.init.constant_(m.bias, 0)    # b初始为0

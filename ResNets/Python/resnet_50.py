# Imports
import torch.nn as nn

# Model
class conv_block(nn.Module):
    def __init__(self, in_channels, int_channels, stride, expand_ratio=4, stride_flag=False):
        super(conv_block, self).__init__()
        self.Stride2_Flag = stride_flag
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=int_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(int_channels)
        self.conv2 = nn.Conv2d(in_channels=int_channels, out_channels=int_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int_channels)
        self.conv3 = nn.Conv2d(in_channels=int_channels, out_channels=int_channels*expand_ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(int_channels*expand_ratio)
        self.relu = nn.ReLU()
        self.identity_downsample1 = nn.Conv2d(in_channels=in_channels, out_channels=int_channels * expand_ratio, kernel_size=1, stride=2, padding=0, bias=False)
        self.identity_downsample2 = nn.Conv2d(in_channels=in_channels, out_channels=int_channels * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):

        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.Stride2_Flag != False:
            shortcut = self.identity_downsample2(shortcut)
        else:
            shortcut = self.identity_downsample1(shortcut)

        x = x + shortcut
        x = self.relu(x)
        return x



class ResNet_50(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet_50, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

        self.layer1 = nn.Sequential(
            conv_block(in_channels=64, int_channels=64, stride=1, stride_flag=True),
            conv_block(in_channels=256, int_channels=64, stride=1, stride_flag=True),
            conv_block(in_channels=256, int_channels=64, stride=1, stride_flag=True),
        )

        self.layer2 = nn.Sequential(
            conv_block(in_channels=256, int_channels=128, stride=2, stride_flag=False),
            conv_block(in_channels=512, int_channels=128, stride=1, stride_flag=True),
            conv_block(in_channels=512, int_channels=128, stride=1, stride_flag=True),
            conv_block(in_channels=512, int_channels=128, stride=1, stride_flag=True),
        )

        # 6 x Convolutional Block
        self.layer3 = nn.Sequential(
            conv_block(in_channels=512, int_channels=256, stride=2, stride_flag=False),
            conv_block(in_channels=1024, int_channels=256, stride=1, stride_flag=True),
            conv_block(in_channels=1024, int_channels=256, stride=1, stride_flag=True),
            conv_block(in_channels=1024, int_channels=256, stride=1, stride_flag=True),
            conv_block(in_channels=1024, int_channels=256, stride=1, stride_flag=True),
            conv_block(in_channels=1024, int_channels=256, stride=1, stride_flag=True),
        )

        self.layer4 = nn.Sequential(
            conv_block(in_channels=1024, int_channels=512, stride=2, stride_flag=False),
            conv_block(in_channels=2048, int_channels=512, stride=1, stride_flag=True),
            conv_block(in_channels=2048, int_channels=512, stride=1, stride_flag=True),
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x





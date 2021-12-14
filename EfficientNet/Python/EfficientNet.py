"""
EfficientNet Model Architecture
Reference: https://arxiv.org/abs/1905.11946
"""

import torch
import torch.nn as nn
from math import ceil
from config import model_configs, phi_values

class cnn_bn(nn.Module):
    """
    Basic convolution operations together with Batch Normalization and ReLU non-linearity.
    If we set groups=1 as we did by default, then this is a normal Conv.
    If we set it to groups=in_channels, then it is a Depthwise Conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(cnn_bn, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.cnn(x)))


class sne_conv(nn.Module):
    """
    For each channel, its going to be multiplied with output of the sequential layer (Squeeze_and_Excitation) that
    gives us a value for each channel how much that channel is prioritized.
    """
    def __init__(self, in_channels, reduced_dim):
        super(sne_conv, self).__init__()
        self.Squeeze_and_Excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.ReLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x * self.Squeeze_and_Excitation(x)
        return x


class mb_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction=4):
        """
        Mobile Inverted Bottleneck + Squeeze and Excitation
        Structure:
            1x1 Expansion Layer => 3x3 Depth-wise Conv => Squeeze-Excitation => 1x1 Projection Layer

        expand ratio (int): the amount of expansion in the Expansion Layer
        reduction (int): Need for reduction on channels, during Squeeze and Excitation operation
        """
        super(mb_conv, self).__init__()
        self.flag_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.flag_expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)  

        if self.flag_expand:
            self.expand_conv = cnn_bn(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, )

        self.conv = nn.Sequential(
            cnn_bn(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, ),
            sne_conv(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )


    def forward(self, x):
        residual = x
        x = self.expand_conv(x) if self.flag_expand else x

        if self.flag_residual:
            return self.conv(x) + residual
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.compound_scaling(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.backbone = self.model_layers(width_factor, depth_factor, last_channels)
        self.fc = nn.Linear(last_channels, num_classes) 


    def compound_scaling(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate


    def model_layers(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        layers = []
        in_channels = channels

        layers.append(cnn_bn(3, channels, 3, stride=2, padding=1, ))

        for expand_ratio, channels, kernel_size, stride, repeats in model_configs:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                layers.append(
                    mb_conv(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2, 
                    )
                )
                in_channels = out_channels

        layers.append(cnn_bn(in_channels, last_channels, kernel_size=1, stride=1, padding=0, ))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

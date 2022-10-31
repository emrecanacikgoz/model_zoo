# Imports
import torch
import torch.nn as nn

# MobileNet-v1
class MobileNetV1(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, outp, stride):
            """
            Convolutional Layer with Batch Normalization
            """
            return nn.Sequential(
                nn.Conv2d(in_channels=inp, out_channels=outp, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(outp, affine=False),
                nn.ReLU(inplace=True)
            )

        def conv_dsc(inp, outp, stride):
            """
            conv_dsc: Depthwise Separable Convolutional Layer
            If we set groups=1 as we did by default, then this is a normal conv.
            If we set it to groups=inp, then it is a Depthwise conv.
            """
            return nn.Sequential(
                # Depthwise Layer
                nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                nn.ReLU(inplace=True),

                # Pointwise Layer
                nn.Conv2d(in_channels=inp, out_channels=outp, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outp, affine=False),
                nn.ReLU(inplace=True),
            )

        # MobileNet Body Architecture
        self.model = nn.Sequential(
            conv_bn(in_channel, 8, 2),
            conv_dsc(8, 16, 1),
            conv_dsc(16, 32, 2),
            conv_dsc(32, 32, 1),
            conv_dsc(32, 64, 2),
            conv_dsc(64, 64, 1),
            conv_dsc(64, 128, 2),
            conv_dsc(128, 128, 1),
            conv_dsc(128, 128, 1),
            conv_dsc(128, 128, 1),
            conv_dsc(128, 128, 1),
            conv_dsc(128, 128, 1),
            conv_dsc(128, 256, 2),
            conv_dsc(256, 256, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x
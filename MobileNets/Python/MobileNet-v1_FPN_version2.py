# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F



# MobileNet-v1 + FPN
class MobileNetV1_FPN(nn.Module):
    def __init__(self, in_channel=3, n_classes=1):
        super(MobileNetV1_FPN, self).__init__()
        self.num_features_out = 32

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
            groups = inp: Each input channel is convolved with its own set of filters
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
    
        
        self.fpn1 = nn.Sequential(
            conv_bn(in_channel, 8, 2),
            conv_dsc(8, 16, 1),
            conv_dsc(16, 32, 2),
            conv_dsc(32, 32, 1)
        )
        
        self.fpn2 = nn.Sequential(
            conv_dsc(32, 64, 2),
            conv_dsc(64, 64, 1),
            conv_dsc(64, 128, 2),
            conv_dsc(128, 128, 1),
            conv_dsc(128, 128, 1),
            conv_dsc(128, 128, 1),
            conv_dsc(128, 128, 1),
            conv_dsc(128, 128, 1)
        )
        
        self.fpn3 = nn.Sequential(
            conv_dsc(128, 256, 2),
            conv_dsc(256, 256, 2)
        )
        
        # Lateral 1x1 Convolutions
        self.lateral_fpn1 = nn.Conv2d(in_channels=32, out_channels=self.num_features_out, kernel_size=1)
        self.lateral_fpn2 = nn.Conv2d(in_channels=128, out_channels=self.num_features_out, kernel_size=1)
        self.lateral_fpn3 = nn.Conv2d(in_channels=256, out_channels=self.num_features_out, kernel_size=1)
        
        # Upsampling
        self.upsample_fpn1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=self.num_features_out, kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(in_channels=32, out_channels=self.num_features_out, kernel_size=2, stride=2, padding=0)
        )
        self.upsample_fpn2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=self.num_features_out, kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(in_channels=32, out_channels=self.num_features_out, kernel_size=2, stride=2, padding=0)
        )
        
        # 1x1 Convolutions for Outputs 
        self.output_fpn1 = nn.Conv2d(in_channels=self.num_features_out, out_channels=12, kernel_size=1, stride=1)
        self.output_fpn2 = nn.Conv2d(in_channels=self.num_features_out, out_channels=12, kernel_size=1, stride=1)
        self.output_fpn3 = nn.Conv2d(in_channels=self.num_features_out, out_channels=12, kernel_size=1, stride=1)
        
        
    def forward(self, x):
        
        # Bottom-up pathway
        outp1 = self.fpn1(x)
        outp2 = self.fpn2(outp1)
        outp3 = self.fpn3(outp2)
        
        # Top-down pathway and lateral connections
        p3 = self.lateral_fpn3(outp3)
        p2 = self.lateral_fpn2(outp2) + self.upsample_fpn1(p3)
        p1 = self.lateral_fpn1(outp1) + self.upsample_fpn2(p2)
        
        # 1x1 Conv Outputs
        p3 = self.output_fpn2(p3)
        p2 = self.output_fpn2(p2)
        p1 = self.output_fpn1(p1)
        
        return p1, p2, p3
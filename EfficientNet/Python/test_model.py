"""
Test script for testing EfficientNet.py model.
Please check the output summary.

Model Ouput of EfficientNet-B0 for given sample (3, 224, 224):
    ===============================
    Total params: 4,008,829
    Trainable params: 4,008,829
    Non-trainable params: 0
    -------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 219.01
    Params size (MB): 15.29
    Estimated Total Size (MB): 234.88
    -------------------------------
"""

import torch
from config import phi_values
from EfficientNet import  EfficientNet
from torchsummary import summary

def test():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VERSION = "b0"
    BATCH_SIZE = 64
    NUM_CLASSES = 1
    phi, res, drop_rate = phi_values[VERSION]
    x = torch.randn((BATCH_SIZE, 3, res, res)).to(DEVICE)
    model = EfficientNet(version=VERSION, num_classes=NUM_CLASSES).to(DEVICE)

    print(model(x).shape) # (num_examples, num_classes)
    summary(model, input_size=(3, res, res), device=DEVICE)

test()

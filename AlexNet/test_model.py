import torch
from torchsummary import summary
from AlexNet import alex_net

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = alex_net().to(DEVICE)

# 227 Spatial Dim.s are necessary which came from Data Augmentations defined in paper.
summary(model=model, input_size=(3, 227, 227), device=DEVICE)
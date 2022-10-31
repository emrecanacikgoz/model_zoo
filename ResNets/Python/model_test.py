import torch
from torchsummary import summary
from resnet_50 import ResNet_50
from resnet_101 import ResNet_101
from resnet_152 import ResNet_152

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 1000
model = ResNet_152(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
summary(model, input_size=(3, 224, 224), device=DEVICE)
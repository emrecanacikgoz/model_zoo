import torch
from torchsummary import summary
from LeNet_5 import lenet_5

device = "cuda" if torch.cuda.is_available() else "cpu"
model = lenet_5().to(device)
summary(model, input_size=(1, 32, 32), device=device)
x = torch.randn(64, 1, 32, 32).to(device)
print(model(x).shape)
"""
Reference Paper: Krizhevsky, Alex; Sutskever, Ilya; Hinton, Geoffrey E. (2017-05-24).
"ImageNet classification with deep convolutional neural networks".
Communications of the ACM. 60 (6): 84–90. doi:10.1145/3065386.
ISSN 0001-0782.
S2CID 195908774.
"""
# Imports
import torch.nn as nn

# Define Model
class alex_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(alex_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(9216, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Convolutions
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool3(x)

        # Fully Connecteds
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        out = self.softmax(x)

        return out





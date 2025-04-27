import torch
import torch.nn as nn
from _config import resize_x, resize_y, input_channels, num_classes

class TrafficSignCNN(nn.Module):
    def __init__(self):
        super(TrafficSignCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        
        # Calculate the size of the flattened features
        self._to_linear = None
        self._get_conv_output((input_channels, resize_x, resize_y))
        
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def _get_conv_output(self, shape):
        input = torch.rand(1, *shape)
        output = self.pool(self.batchnorm1(nn.functional.relu(self.conv1(input))))
        output = self.pool(self.batchnorm2(nn.functional.relu(self.conv2(output))))
        output = self.pool(self.batchnorm3(nn.functional.relu(self.conv3(output))))
        self._to_linear = output.view(output.size(0), -1).size(1)
        
    def forward(self, x):
        x = self.pool(self.batchnorm1(nn.functional.relu(self.conv1(x))))
        x = self.pool(self.batchnorm2(nn.functional.relu(self.conv2(x))))
        x = self.pool(self.batchnorm3(nn.functional.relu(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
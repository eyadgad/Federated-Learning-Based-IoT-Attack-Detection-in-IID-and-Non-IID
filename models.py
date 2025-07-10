import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------
# CNN Model
# -----------------------------------------------------------
class CNN(nn.Module):
    """
    Args:
        num_classes (int): Number of output classes.
        input_size (tuple): Input dimensions (channels, height, width).
    """
    def __init__(self, num_classes=34, input_size=(1, 28, 28)):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 7 * 7)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------------------------------------
# ANN Model
# -----------------------------------------------------------
class ANN(nn.Module):
    def __init__(self, num_classes=10, input_size=(1, 28, 28)):
        super(ANN, self).__init__()
        self.input_size = input_size
        input_features = np.prod(self.input_size) if isinstance(self.input_size, tuple) else self.input_size
        self.fc1 = nn.Linear(input_features, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1) if isinstance(self.input_size, tuple) else x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        return self.fc5(x)
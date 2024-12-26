import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------
# CNN Model
# -----------------------------------------------------------
class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for image classification.

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
        x = x.view(-1, 32 * 7 * 7)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------------------------------------
# ANN Model
# -----------------------------------------------------------
class ANN(nn.Module):
    """
    A Fully Connected Artificial Neural Network (ANN) for classification.

    Args:
        name (str): Name of the model.
        num_classes (int): Number of output classes.
        input_size (int): Size of the input feature vector.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
    """
    def __init__(self, name, num_classes=34, input_size=43, batch_size=64, epochs=10, lr=1e-2):
        super(ANN, self).__init__()
        self.name = name
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        # Model performance metrics
        self.accuracy = 0
        self.loss = 0
        self.train_loss_hist = []
        self.train_acc_hist = []
        self.val_loss_hist = []
        self.val_acc_hist = []
        self.mae_hist = []
        self.rmse_hist = []

        # Additional control structures
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}

        # Define layers
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)  # Regularization

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  # Final layer, no activation applied here
        return x


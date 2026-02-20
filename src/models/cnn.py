import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture with 4 convolutional layers followed by a fully connected layer.
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        # input is 3 channels, output channels increase with each layer
        # self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(4, 4, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # linear head
        self.fc_input_dim = 4 * 3 * 3

        self.fc = nn.Linear(self.fc_input_dim, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_cam_target_layers(self):
        """
        Utility function to return the target layers for CAM generation.
        For this simple CNN, we will use the last convolutional layer (conv4).
        """
        return [self.conv4]

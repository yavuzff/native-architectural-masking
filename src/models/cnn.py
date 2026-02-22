import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Low-capacity CNN architecture that perfectly shrinks a 28x28 image down to a 4x4
    spatial resolution to isolate corner patches.
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        # Block 1: 28x28 -> 26x26 -> 24x24
        # we use padding=0 (default) to slowly shrink the spatial dimensions
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=0)

        # Pool 1: 24x24 -> 12x12
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 12x12 -> 10x10 -> 8x8
        self.conv3 = nn.Conv2d(4, 4, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(4, 4, kernel_size=3, padding=0)

        # Pool 2: 8x8 -> 4x4
        self.pool2 = nn.MaxPool2d(2, 2)

        # linear head
        # we now have 4 channels, and the spatial dimensions are exactly 4x4.
        self.fc_input_dim = 4 * 4 * 4

        self.fc = nn.Linear(self.fc_input_dim, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_cam_target_layers(self):
        """
        Target the last convolutional layer (conv4) before the final pooling.
        At this layer, the spatial dimension is 8x8.
        """
        return [self.conv4]

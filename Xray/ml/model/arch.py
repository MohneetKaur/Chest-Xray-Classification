import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        """
        Creating custom CNN architecture for Image classification
        """
        super(Net, self).__init__()

        # First convolutional block: 
        # 3 input channels (RGB), 8 output channels, kernel size 3x3, no padding
        self.convolution_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=8, kernel_size=(3, 3), padding=0, bias=True
            ),
            nn.ReLU(),  # Activation function
            nn.BatchNorm2d(8),  # Normalization for 8 output channels
        )

        # First max pooling layer: reduces the spatial dimensions by half (2x2)
        self.pooling11 = nn.MaxPool2d(2, 2)

        # Second convolutional block: 
        # 8 input channels, 20 output channels, kernel size 3x3, no padding
        self.convolution_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8, out_channels=20, kernel_size=(3, 3), padding=0, bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(20),
        )

        # Second max pooling layer: reduces the spatial dimensions by half (2x2)
        self.pooling22 = nn.MaxPool2d(2, 2)

        # Third convolutional block: 
        # 20 input channels, 10 output channels, kernel size 1x1 (point-wise convolution)
        self.convolution_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )

        # Third max pooling layer: reduces the spatial dimensions by half (2x2)
        self.pooling33 = nn.MaxPool2d(2, 2)

        # Fourth convolutional block: 
        # 10 input channels, 20 output channels, kernel size 3x3, no padding
        self.convolution_block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=(3, 3),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(20),
        )

        # Fifth convolutional block: 
        # 20 input channels, 32 output channels, kernel size 1x1
        self.convolution_block5 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=32,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        # Sixth convolutional block: 
        # 32 input channels, 10 output channels, kernel size 3x3
        self.convolution_block6 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )

        # Seventh convolutional block: 
        # 10 input channels, 10 output channels, kernel size 1x1
        self.convolution_block7 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )

        # Eighth convolutional block: 
        # 10 input channels, 14 output channels, kernel size 3x3
        self.convolution_block8 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=14,
                kernel_size=(3, 3),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(14),
        )

        # Ninth convolutional block: 
        # 14 input channels, 16 output channels, kernel size 3x3
        self.convolution_block9 = nn.Sequential(
            nn.Conv2d(
                in_channels=14,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        # Global Average Pooling: reduces each feature map to a single value
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=4))

        # Output convolutional block:
        # 16 input channels, 2 output channels, kernel size 4x4, no padding
        self.convolution_block_out = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=2, kernel_size=(4, 4), padding=0, bias=True
            ),
        )

    def forward(self, x) -> float:
        # Forward pass through each block

        # First Convolution + ReLU + BatchNorm + MaxPool
        x = self.convolution_block1(x)
        x = self.pooling11(x)

        # Second Convolution + ReLU + BatchNorm + MaxPool
        x = self.convolution_block2(x)
        x = self.pooling22(x)

        # Third Convolution + ReLU + BatchNorm + MaxPool
        x = self.convolution_block3(x)
        x = self.pooling33(x)

        # Fourth Convolution + ReLU + BatchNorm
        x = self.convolution_block4(x)

        # Fifth Convolution + ReLU + BatchNorm
        x = self.convolution_block5(x)

        # Sixth Convolution + ReLU + BatchNorm
        x = self.convolution_block6(x)

        # Seventh Convolution + ReLU + BatchNorm
        x = self.convolution_block7(x)

        # Eighth Convolution + ReLU + BatchNorm
        x = self.convolution_block8(x)

        # Ninth Convolution + ReLU + BatchNorm
        x = self.convolution_block9(x)

        # Global Average Pooling
        x = self.gap(x)

        # Output Convolution
        x = self.convolution_block_out(x)

        # Flattening the tensor for fully connected layer
        x = x.view(-1, 2)

        # Applying log softmax for probability scores
        return F.log_softmax(x, dim=-1)

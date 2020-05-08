import torch.nn as nn                        # Import neural net module from pytorch
import torch.nn.functional as F              # Import functional interface from pytorch


# Subclassing nn.Module for neural networks
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        ##############################################################################################################
        # CONVOLUTION BLOCK 1 - DILATED CONVOLUTION
        ##############################################################################################################
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout(0.05)
        )
        ##############################################################################################################
        # TRANSITION BLOCK 1
        ##############################################################################################################
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout(0.05)
        )
        ##############################################################################################################
        # CONVOLUTION BLOCK 2
        ##############################################################################################################
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(2, 2), bias=False, dilation=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout(0.05)
        )
        ##############################################################################################################
        # TRANSITION BLOCK 2
        ##############################################################################################################
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Dropout(0.05)
        )
        ##############################################################################################################
        # CONVOLUTION BLOCK 3 - DEPTH-WISE SEPERABLE
        ##############################################################################################################
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), groups=256, bias=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Dropout(0.05)
        )
        ##############################################################################################################
        # TRANSITION BLOCK 3
        ##############################################################################################################
        self.pool3 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Dropout(0.05)
        )
        ##############################################################################################################
        # CONVOLUTION BLOCK 4
        ##############################################################################################################
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), groups=256, bias=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Dropout(0.05)
        )
        ##############################################################################################################
        # OUTPUT BLOCK
        ##############################################################################################################
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Dropout(0.05)
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=(0, 0), bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool2(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool3(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = self.convblock9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
import torch
import torchvision
import numpy as np
import pandas as pd
import cv2

import torch                                 # Import pytorch library
import torch.nn as nn                        # Import neural net module from pytorch
import torch.nn.functional as F              # Import functional interface from pytorch
import torch.optim as optim                  # Import optimizer module from pytorch

from torchvision import datasets, transforms # Import datasets and augmentation functionality from vision module within pytorch
from torchsummary import summary             # Import summary with pytorch
from torchviz import make_dot

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR

# Subclassing nn.Module for neural networks
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ############################################################################################################
        # PREP LAYER
        ############################################################################################################
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU()
        )
        ############################################################################################################
        # LAYER 1
        ############################################################################################################
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        ############################################################################################################
        # RESNET BLOCK 1
        ############################################################################################################
        self.resblock1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        ############################################################################################################
        # LAYER 2
        ############################################################################################################
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        ############################################################################################################
        # LAYER 3
        ############################################################################################################
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        ############################################################################################################
        # RESNET BLOCK 2
        ############################################################################################################
        self.resblock2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        ############################################################################################################
        # MAX POOL
        ############################################################################################################
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(4, 4)
        )
        ############################################################################################################
        # OUTPUT BLOCK
        ############################################################################################################
        self.linear = nn.Sequential(
            nn.Linear(in_features=512, out_features=10)
        )
        

    def blocks(self, x):

        '''
        PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        Layer1 -
        X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
        Add(X, R1)
        Layer 2 -
        Conv 3x3 [256k]
        MaxPooling2D
        BN
        ReLU
        Layer 3 -
        X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        Add(X, R2)
        MaxPooling with Kernel Size 4
        FC Layer 
        SoftMax
        '''

        # PREP LAYER
        x = self.input_layer(x)

        # LAYER 1
        x = self.layer1(x)
        # RESNET BLOCK 1
        r1 = self.resblock1(x)
        x = x + r1

        # LAYER 2
        x = self.layer2(x)

        # LAYER 2
        x = self.layer3(x)
        # RESNET BLOCK 2
        r2 = self.resblock2(x)
        x = x + r2

        # MAX POOL
        x = self.pool1(x)
        x = torch.flatten(x, 1)

        # LINEAR
        x = self.linear(x)

        return x

    def forward(self, x):
        x = self.blocks(x)
        return F.log_softmax(x, dim=-1)
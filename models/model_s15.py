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
        # _droput = 0.00
        #########################################################################################
        # INPUT BLOCK
        #########################################################################################
        self.double_conv_down0 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU()
            # nn.Dropout(_droput)
        )
        self.pool0 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        
        #########################################################################################
        # DOUBLE CONVOLUTION BLOCK DOWN 1
        #########################################################################################
        self.double_conv_down1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU()
            # nn.Dropout(_droput)
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        
        #########################################################################################
        # DOUBLE CONVOLUTION BLOCK DOWN 2
        #########################################################################################
        self.double_conv_down2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU()
            # nn.Dropout(_droput)
        )
        
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        
        #########################################################################################
        # DOUBLE CONVOLUTION BLOCK DOWN 3
        #########################################################################################
        self.double_conv_down3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU()
            # nn.Dropout(_droput)
        )
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )

        #########################################################################################
        # DOUBLE CONVOLUTION BLOCK DOWN 4
        #########################################################################################
        self.double_conv_down4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256), 
            nn.ReLU()
            # nn.Dropout(_droput)
        )
        self.pool4 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )

        #########################################################################################
        # DOUBLE CONVOLUTION BLOCK DOWN 5
        #########################################################################################
        self.double_conv_down5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256), 
            nn.ReLU()
            # nn.Dropout(_droput)
        )

        #########################################################################################
        # UPSAMPLE BLOCK 4
        #########################################################################################
        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)  
            # nn.Dropout(_droput)
        )

        #########################################################################################
        # DOUBLE CONVOLUTION BLOCK UP 4
        #########################################################################################
        self.double_conv_up4 = nn.Sequential(
            nn.Conv2d(in_channels=(256 + 256), out_channels=256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256), 
            nn.ReLU()
            # nn.Dropout(_droput)
        )

        #########################################################################################
        # UPSAMPLE BLOCK 3
        #########################################################################################
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)  
            # nn.Dropout(_droput)
        )

        #########################################################################################
        # DOUBLE CONVOLUTION BLOCK UP 3
        #########################################################################################
        self.double_conv_up3 = nn.Sequential(
            nn.Conv2d(in_channels=(256 + 128), out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU()
            # nn.Dropout(_droput)
        )

        #########################################################################################
        # UPSAMPLE BLOCK 2
        #########################################################################################
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)  
            # nn.Dropout(_droput)
        )

        #########################################################################################
        # DOUBLE CONVOLUTION BLOCK UP 2
        #########################################################################################
        self.double_conv_up2 = nn.Sequential(
            nn.Conv2d(in_channels=(128 + 128), out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU()
            # nn.Dropout(_droput)
        )

        #########################################################################################
        # UPSAMPLE BLOCK 1
        #########################################################################################
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)  
            # nn.Dropout(_droput)
        )

        #########################################################################################
        # DOUBLE CONVOLUTION BLOCK UP 1
        #########################################################################################
        self.double_conv_up1 = nn.Sequential(
            nn.Conv2d(in_channels=(128 + 64), out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU()
            # nn.Dropout(_droput)
        )

        #########################################################################################
        # MASK HEAD
        #########################################################################################
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), bias=False),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.Dropout(_droput)
        )

        #########################################################################################
        # DEPTH HEAD
        #########################################################################################
        self.depth_predictor = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), bias=False),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.Dropout(_droput)
        )

    def blocks(self, x):

        conv0 = self.double_conv_down0(x)
        x = self.pool0(conv0)

        conv1 = self.double_conv_down1(x)
        x = self.pool1(conv1) 

        conv2 = self.double_conv_down2(x)
        x = self.pool2(conv2) 

        conv3 = self.double_conv_down3(x)
        x = self.pool3(conv3) 

        conv4 = self.double_conv_down4(x)
        x = self.pool4(conv4) 

        x = self.double_conv_down5(x)

        x = self.upsample4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.double_conv_up4(x)

        x = self.upsample3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.double_conv_up3(x)

        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.double_conv_up2(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.double_conv_up1(x)

        # Mask prediction
        x1 = self.mask_predictor(x)
        x1 = torch.flatten(x1, 1, 2)

        # Depth map prediction        
        x2 = self.depth_predictor(x) 
        x2 = torch.flatten(x2, 1, 2)
        
        return x1, x2

    def forward(self, x):
        
        x1, x2 = self.blocks(x)
        
        return x1, x2
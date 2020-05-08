# This file contains the memeNet class (resnet)

import torch
import torch.nn as nn
import torchvision
import numpy as np
import copy


class resnetFineTuning(torch.nn.Module):
    def __init__(self, model):
        super(resnetFineTuning, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model.forward(x)
        x = torch.sigmoid(x)
        return x

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1, num_convolutions=3):
        super(block, self).__init__()
        self.num_convolutions = num_convolutions
        if num_convolutions == 3:
            self.expansion = 4

            # First convolutional layer
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu1 = nn.ReLU()

            # Second convolutional layer
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.ReLU()

            #Third convolutional layer
            self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
            self.relu3 = nn.ReLU()

        else:
            self.expansion = 1

            # First convolutional layer
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu1 = nn.ReLU()

            # Second convolutional layer
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.ReLU()

        self.identity_downsample = identity_downsample


    def forward(self, input):
        identity = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.num_convolutions == 3:
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            if self.identity_downsample is not None:
                identity = self.identity_downsample(identity)
            x += identity
            x = self.relu3(x)
            return x

        else:
            if self.identity_downsample is not None:
                identity = self.identity_downsample(identity)
            x += identity
            x = self.relu2(x)
            return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes, num_convolutions):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # Initial Layers
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Creating ResNet layers
        self.layer1 = self.make_layer(block, layers[0], out_channels=64, stride=1, num_convolutions=num_convolutions)
        self.layer2 = self.make_layer(block, layers[1], out_channels=128, stride=2, num_convolutions=num_convolutions)
        self.layer3 = self.make_layer(block, layers[2], out_channels=256, stride=2, num_convolutions=num_convolutions)
        self.layer4 = self.make_layer(block, layers[3], out_channels=512, stride=2, num_convolutions=num_convolutions)

        self.avgpool = nn.AvgPool2d(7, stride=1) # It does an average pooling to obtain an output of the desired size, in this case 1x1
        if num_convolutions ==3:
            self.fc = nn.Linear(512 * 4, num_classes)
        else:
            self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) # we reshape so we can input it in the FC layer
        x = self.fc(x)
        return x

    def make_layer(self, block, num_residual_blocks, out_channels, stride, num_convolutions=3): # num_residual_blocks: how many times we aer using the block
        identity_downsample = None
        layers = []

        if num_convolutions==3:
            expansion = 4
        else:
            expansion = 1

        if stride != 1 or self.in_channels != out_channels*expansion:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels*expansion))
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride, num_convolutions=num_convolutions)) # This is the first (residual) block. For instance in Resnet 152 this is going to be the first three layers (in the first block)

        self.in_channels = out_channels * expansion

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels, num_convolutions=num_convolutions)) # Here we don't use identity_downsample because we ar enot modifying the number of out channels

        return nn.Sequential(*layers) # With the * command we unpack the layers list in sucha a way that pytorch understands that each element of the list is one layer applied after the other


def ResNet18(img_channels=3, num_classes=15):
    layers = [2, 2, 2, 2]
    return ResNet(block, layers, img_channels, num_classes, num_convolutions=2)

def ResNet34(img_channels=3, num_classes=15):
    layers = [3, 4, 6, 3]
    return ResNet(block, layers, img_channels, num_classes, num_convolutions=2)

def ResNet50(img_channels=3, num_classes=15):
    layers = [3, 4, 6, 3]
    return ResNet(block, layers, img_channels, num_classes, num_convolutions=3)

def ResNet101(img_channels=3, num_classes=15):
    layers = [3, 4, 23, 3]
    return ResNet(block, layers, img_channels, num_classes, num_convolutions=3)

def ResNet152(img_channels=3, num_classes=15):
    layers = [3, 8, 36, 3]
    return ResNet(block, layers, img_channels, num_classes, num_convolutions=3)

def test():
    net = ResNet18()
    x = torch.randn(2, 1, 224, 224)
    y = net(x).to('cuda')



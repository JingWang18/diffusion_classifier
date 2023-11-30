import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import math


class ResBase101(nn.Module):
    def __init__(self):
        super(ResBase101, self).__init__()
        model_resnet101 = models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool

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
        x = x.view(x.size(0), -1)
        return x

class ResBase50(nn.Module):
    def __init__(self):
        super(ResBase50, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool

        # self.time_conv = nn.Conv2d(64, 64, kernel_size=1)
        # self.time_bn = nn.BatchNorm2d(64)
        # self.time_relu = nn.ReLU()

        # self.density_conv = nn.Conv2d(64, 64, kernel_size=1)
        # self.density_bn = nn.BatchNorm2d(64)
        # self.density_relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #########################
        ####### ResBlock ########
        #########################
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def forward_diffusion(x, step, mu, std):
    """ Apply forward diffusion to data x at time step t with noise level beta """
    noise = mu + std*torch.randn_like(std)
    return x * math.sqrt(1 - step) + math.sqrt(step) * noise.cuda()


class ResClassifier(nn.Module):
    def __init__(self, class_num=12, extract=False):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout()
            )
        self.fc2 = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout()
            )
        self.fc3 = nn.Linear(1000, class_num)
        self.extract = extract

        self.density_conv = nn.Linear(2048, 2048)
        self.density_bn = nn.BatchNorm1d(2048)
        self.density_relu = nn.ReLU()

    def forward(self, x, step, mu, std, reverse=False):

        noisy_x = x + forward_diffusion(x, step, mu, std)

        # diffusion backward
        estimated_noise = self.density_relu(self.density_bn(self.density_conv(noisy_x)))

        if reverse:
            noisy_x = x + estimated_noise

        fc1_emb = self.fc1(noisy_x)
        if self.training:
            fc1_emb.mul_(math.sqrt(0.5))
        fc2_emb = self.fc2(fc1_emb)
        if self.training:
            fc2_emb.mul_(math.sqrt(0.5))    

        logit = self.fc3(fc2_emb)


        if self.extract:
            return fc2_emb, logit
        return logit
        
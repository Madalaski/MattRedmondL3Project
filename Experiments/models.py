import torch
from torch.utils import data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchsummary import summary
import numpy

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,96,11,4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96,256,5,1,2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216,4098),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4098,4098),
            nn.ReLU(inplace=True),
            nn.Linear(4098, 17),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(-1, 9216)
        x = self.linear(x)
        return x

class Comparator(torch.nn.Module):
    def __init__(self):
        super(Comparator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,64, 5, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256,128,2)
        )
        #self.linear = nn.Sequential(nn.Linear(128,3))


    def forward(self, a):
        a = self.conv(a)
        a = a.view(-1, 128)
        #a = self.linear(a)
        return a

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier().to(device)

    summary(model, (1, 227, 227))

    model2 = Comparator().to(device)

    summary(model2, (1, 96, 96))

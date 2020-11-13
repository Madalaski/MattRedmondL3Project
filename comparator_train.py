import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import glob
import numpy
import json
from PIL import Image
from models import Comparator
import time
from datetime import datetime

with open("values.json", "r") as v:
    values = json.load(v)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Triplet Dataloader for first training iteration

class PreTrainDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.classLengths = []
        self.modelPaths = []
        self.root = root
        
        self.transform = transform
        folders = glob.glob(root+"\*")
        for idx in range(len(values["classes"])):
            mPaths = glob.glob(folders[idx]+"\*")
            self.classLengths.append(len(mPaths))
            self.modelPaths.append(mPaths)
        
    def __len__(self):
        return sum(self.classLengths) * 20
    
    def __getitem__(self, idx):
        # Feed Class-Specific Triplets

        i = 0
        x = idx
        while x >= self.classLengths[i] * 20:
            x -= self.classLengths[i] * 20
            i += 1
        
        y = (x + (x*31)) % (self.classLengths[i]*20)
        if x//20 == y//20:
            y += 20
        
        zi = (i + (x*31)) % len(self.classLengths)
        while (zi == i):
            zi += 31
            zi %= len(self.classLengths) 
        
        z = (x + (x*31)) % (self.classLengths[zi])

        pa = self.modelPaths[i][x//20]
        pp = self.modelPaths[i][y//20]
        pn = self.modelPaths[zi][z]
        a = self.transform(Image.open(pa).convert("RGB"))
        p = self.transform(Image.open(pp).convert("RGB"))
        n = self.transform(Image.open(pn).convert("RGB"))

        return a,p,n
        
# Triplet Data loading for second training iteration

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, root, root2, transform):
        self.classLengths = []
        self.modelPaths = []
        self.sketchLengths = []
        self.sketchPaths = []
        self.root = root
        self.root2 = root2
        self.transform = transform
        # Set up model and sketch paths to improve performance
        folders = glob.glob(root+"\*")
        for idx in range(len(values["classes"])):
            mPaths = glob.glob(folders[idx]+"\*")
            self.classLengths.append(len(mPaths))
            self.modelPaths.append(mPaths)
        sketchfolders = glob.glob(root2+"\*")
        for folder in sketchfolders:
            sPaths = glob.glob(folder+"\*")
            self.sketchLengths.append(len(sPaths))
            self.sketchPaths.append(sPaths)
                
    
    def __len__(self):
        return sum(self.classLengths) * 20 + sum(self.sketchLengths) * 20
    
    def __getitem__(self, idx):
        if (idx < sum(self.classLengths) * 10):
            # Feed Comparative Triplets

            i = 0
            x = idx
            while x >= self.classLengths[i] * 10:
                x -= self.classLengths[i] * 10
                i += 1

            modelX = x // 200
            indexX = (x % 200) // 20
            indexY = (indexX + (x*23)) % 20
            while indexY == indexX:
                indexY += 23
                indexY %= 20
            models = self.classLengths[i] // 20
            modelZ = (modelX + (x*31)) % models
            while modelZ == modelX:
                modelZ += 31
                modelZ %= models
            pa = self.modelPaths[i][x//20]
            pp = self.modelPaths[i][(modelX*20)+indexY]
            pn = self.modelPaths[i][(modelZ*20)+indexY]
            a = self.transform(Image.open(pa).convert("L"))
            p = self.transform(Image.open(pp).convert("L"))
            n = self.transform(Image.open(pn).convert("L"))

            return a,p,n
        elif idx < sum(self.classLengths) * 20:
            # Feed Class-Specific Triplets

            i = 0
            x = idx - (sum(self.classLengths) * 10)
            while x >= self.classLengths[i] * 10:
                x -= self.classLengths[i] * 10
                i += 1
            
            y = (x + (x*31)) % (self.classLengths[i]*10)
            if x//20 == y//20:
                y += 20
            
            zi = (i + (x*31)) % len(self.classLengths)
            while (zi == i):
                zi += 31
                zi %= len(self.classLengths) 
            
            z = (x + (x*31)) % (self.classLengths[zi])

            pa = self.modelPaths[i][x//20]
            pp = self.modelPaths[i][y//20]
            pn = self.modelPaths[zi][z]
            a = self.transform(Image.open(pa).convert("L"))
            p = self.transform(Image.open(pp).convert("L"))
            n = self.transform(Image.open(pn).convert("L"))

            return a,p,n
        else:
            # Feed Sketch -> Model Triplets

            i = 0
            x = idx - (sum(self.classLengths) * 20)
            while x >= self.sketchLengths[i] * 20:
                x -= self.sketchLengths[i] * 20
                i += 1

            xi = x // 20

            y = (x + (x*31)) % (self.classLengths[i])

            zi = (i + (x*31)) % len(self.classLengths)
            while (zi == i):
                zi += 31
                zi %= len(self.classLengths) 
            
            z = (x + (x*31)) % (self.classLengths[zi])

            pa = self.sketchPaths[i][xi]
            pp = self.modelPaths[i][y]
            pn = self.modelPaths[zi][z]
            a = self.transform(Image.open(pa).convert("L"))
            p = self.transform(Image.open(pp).convert("L"))
            n = self.transform(Image.open(pn).convert("L"))

            return a,p,n

if __name__ == '__main__':

    startTime = datetime.now()

    transform = transforms.Compose([transforms.Grayscale(), transforms.Resize(96), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    path = values["datasetFolder"]+"\MODELS\images"
    path2 = values["datasetFolder"]+"\SKETCHES_TRAINING"

    pretrainset = PreTrainDataset(root=path, transform=transform)

    pretrainloader = torch.utils.data.DataLoader(pretrainset, batch_size=4, shuffle=True, num_workers=4)

    trainset = TripletDataset(root=path,  root2=path2, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

    net = Comparator()
    net.to(device)
    if use_cuda:
        net.cuda()

    PATH = values["comparator"]

    # Set Up Loss and Optimiser

    criterion = torch.nn.TripletMarginLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    # Initial Training Phase: Class-specific triplets

    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(pretrainloader):
            a = data[0].to(device)
            p = data[1].to(device)
            n = data[2].to(device)

            optimizer.zero_grad()

            anchor = net(a)
            positive = net(p)
            negative = net(n)
            loss = criterion(anchor, positive, negative)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.10f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
                
    torch.save(net.state_dict(), PATH)

    last_loss = 1.0000
    epoch = 0

    # Second training phase: Comparative triplets

    for epoch in range(7):
        running_loss = 0.0
        for i, data in enumerate(trainloader):

            a = data[0].to(device)
            p = data[1].to(device)
            n = data[2].to(device)

            optimizer.zero_grad()

            anchor = net(a)
            positive = net(p)
            negative = net(n)
            loss = criterion(anchor, positive, negative)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.10f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                last_loss = running_loss / 1000
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), PATH)

    print(datetime.now() - startTime)




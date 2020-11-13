import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy
import json
from PIL import Image
from models import Classifier

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


if __name__ == '__main__':

    with open("values.json", "r") as v:
        values = json.load(v)

    PATH = values["classifier"]

    transform = transforms.Compose([transforms.RandomAffine(degrees=15, scale=(0.9, 1.0), fillcolor=256), transforms.Grayscale(),  transforms.Resize(227), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    path = values["datasetFolder"]+"/SKETCHES_TRAINING"

    trainset = torchvision.datasets.ImageFolder(root=path, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

    classes = set(values["classes"])

    net = Classifier()
    net.to(device)
    if use_cuda:
        net.cuda()

    # Set up loss function and optimiser

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    epoch = 0

    running_loss = 1.0

    # Simple training for 500 epochs

    for epoch in range(500):
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        epoch += 1

    print('Finished Training')
    # saves Classifier
    torch.save(net.state_dict(), PATH)
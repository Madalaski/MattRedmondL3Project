import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
import glob
import json
import numpy
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import random
from models import Classifier, Comparator



if __name__ == '__main__':

    with open(os.path.abspath("..")+"/values.json", "r") as v:
        values = json.load(v)

    results = open("./results"+values["resultsName"]+".txt", "w")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    random.seed()

    classes = values["classes"]

    PATH = "../"+values["classifier"]
    classifier = Classifier()
    classifier.to(device)
    if use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(PATH, map_location=device))
    PATH = "../"+values["comparator"]
    comparator = Comparator()
    comparator.to(device)
    if use_cuda:
        comparator.cuda()
    comparator.load_state_dict(torch.load(PATH, map_location=device))
    transform = transforms.Compose([transforms.Grayscale(), transforms.Resize(227),transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform2 = transforms.Compose([transforms.Grayscale(), transforms.Resize(96),transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    iPath = os.path.abspath("..")+"/"+values["datasetFolder"]+"/MODELS\images"
    cPath = os.path.abspath("..")+"/"+values["datasetFolder"]+"/SKETCHES_TESTING"
    

    testset = torchvision.datasets.ImageFolder(root=cPath, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=4)

    database = []

    with torch.no_grad():

        print("||||||| Classifier Testing ||||||||||||||||||||||||||||||")

        correct = 0
        total = 0
        for data in testloader:
            inputs = data[0].to(device)
            labels = data[1].to(device)
            outputs = classifier(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("Accuracy of the Classifier on the test images: {} %".format(100.0 * float(correct) / float(total)))
        results.write("Accuracy of the Classifier on the test images: {} %\n".format(100.0 * float(correct) / float(total)))

        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))
        for data in testloader:
            inputs = data[0].to(device)
            labels = data[1].to(device)
            outputs = classifier(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


        for i in range(17):
            print("Accuracy of {} : {}%".format(classes[i], 100.0 * float(class_correct[i]) / float(class_total[i])))
            results.write("{}\t{}%\n".format(classes[i], 100.0 * float(class_correct[i]) / float(class_total[i])))

        print("||||||| Comparator Training ||||||||||||||||||||||||||||||")
        
        index = 0
        folders = glob.glob(iPath + "\*")
        classIndexes = [4,6,7,8,15,18,21,24,29,44,45,50,53,76,77,83,85]
        classSizes = []
        for cIndex in classIndexes:
            total = 0
            for image in glob.glob(folders[cIndex] + "\*")[::20]:
                totalPoint = torch.zeros((1,128)).to(device)
                for i in range(20):    
                    inpImage = Image.open(image).convert("L")
                    inp = transform2(inpImage).to(device)
                    point = comparator(inp.unsqueeze(0))
                    totalPoint += point
                totalPoint /= 20
                new = {"class":index,"point":totalPoint}
                database.append(new)
                total += 1
            classSizes.append(total)
            index += 1
        
        index = 0
        size = len(database)

        print(size)


        prcurve = []
        precision = []
        recall = []
        tfolders = glob.glob(cPath + "\*")
        total = 0
        for cIndex in range(17):
            for image in glob.glob(tfolders[cIndex] + "\*"):
                inpImage = Image.open(image).convert("L")
                inp = transform2(inpImage).to(device)
                point = comparator(inp.unsqueeze(0))
                database.sort(key=lambda x: torch.norm(x["point"] - point))
                found = 0
                rIndex = 0
                i = 0
                diff = 0
                while (found < classSizes[index]):
                    if cIndex == database[i]["class"]:
                        found += 1
                        prcurve.append({"p": found / (i+1), "r": found / classSizes[index]})
                    i += 1
                total += 1
            index += 1

        prcurve.sort(key=lambda x: x["r"])

        i = 0
        total = 0

        results.write("PR Curve\n")
        
        r = 0.05
        k = r
        for x in prcurve[1:]:
            if x["r"] > k:
                precision.append(total / i)
                recall.append(k)
                results.write("{}\t{}\n".format(k, total / i))
                k += r
                i = 0
                total = 0
            total += x["p"]
            i += 1

        precision.append(total / i)
        recall.append(1.0)

        results.write("PR Curve With MPA\n")

        size = 4
        
        mpa = []

        for i in range(len(recall)):
            value = numpy.mean(precision[max(0, i-(size//2)) : min(i+(size//2), len(recall))])
            mpa.append(value)
            results.write("{}\t{}\n".format(recall[i], value))

        results.close()
        
        plt.plot(recall, mpa)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()


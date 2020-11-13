import sys
import json
import glob
import random
import torch
import numpy
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from tkinter import Tk, Canvas, NW, CENTER, TOP, StringVar, ROUND, TRUE, messagebox
import tkinter.ttk as ttk
from tkinter.ttk import Button, Frame, Label
from ttkthemes import ThemedStyle
from PIL import Image, ImageDraw, ImageTk, ImageChops
from vpython import canvas, vec, vertex, triangle, box, compound, curve
from models import Classifier, Comparator

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

with open("values.json", "r") as v:
    values = json.load(v)

random.seed()


roomWidth = 9
roomHeight = 6

classes = values["classes"]

models = []

PATH = values["classifier"]
classifier = Classifier()
classifier.load_state_dict(torch.load(PATH, map_location=device))
PATH = values["comparator"]
identifier = Comparator()
identifier.load_state_dict(torch.load(PATH, map_location=device))
transform = transforms.Compose([transforms.Grayscale(), transforms.Resize(227), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
transform2 = transforms.Compose([transforms.Grayscale(), transforms.Resize((96,96)),transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Define a Layer

class Layer():
    def __init__(self, i, root, parent):
        self.image = Image.new('RGB', (1080, 720), (255,255,255))
        self.d = ImageDraw.Draw(self.image)
        self.i = i
        self.root = root
        self.button = Button(root, text="Layer {}".format(i), command=lambda: parent.changeLayer(i))
        self.button.grid(row=i+2, column=6)
        self.lines = []
        
# Set Up The Main Tkinter Window

class Drawer():
    def __init__(self):
        # Set up root
        self.root = Tk()
        self.root.title("Sketch Program")
        self.root.configure(bg="white")
        self.style = ThemedStyle(self.root)
        self.style.theme_use("arc")
        self.root.protocol("WM_DELETE_WINDOW", lambda: close(self.root))

        # Set up buttons at the top
        
        size = 26

        self.show = Button(self.root, text="Show", command=self.showImage, width=size)
        self.show.grid(row=0, column=0)

        self.rButton = Button(self.root, text="Reconstruct", command=self.reconstruct, width=size)
        self.rButton.grid(row=0, column=1)

        self.classifyButton = Button(self.root, text="Classify", command=lambda: self.classify(self.layerList[self.currentLayer].image), width=size)
        self.classifyButton.grid(row=0, column=2)

        self.clearButton = Button(self.root, text="Clear", command=self.clear, width=size)
        self.clearButton.grid(row=0, column=3)

        self.randomTestButton = Button(self.root, text="Random Test", command=self.test, width=size)
        self.randomTestButton.grid(row=0, column=4)

        self.cls = -1
        self.layer_name = StringVar()
        self.layer_name.set("Layer 0")
        self.cls_lbl = Label(self.root, textvariable=self.layer_name, width=size, anchor=CENTER, background="white")
        self.cls_lbl.grid(row=0,column=5)

        # Set up Canvas

        self.c = Canvas(self.root, bg='white', width=1080, height=720)
        self.room = Image.open("room.png")
        self.photo = ImageTk.PhotoImage(self.room)
        self.c.create_image(0, 0, anchor=NW, image=self.photo, tags="background")
        self.c.grid(row=1, column=0, columnspan=6, rowspan=15)

        # Set up Layer Panel

        self.addLayer = Button(self.root, text="+", command=self.add, width=9)
        self.addLayer.grid(row=1, column=6)

        self.layers = Frame(self.root, width=400, height=720)
        self.layers.grid(row=2, column=6, rowspan=14)

        for i in range(14):
            self.layers.rowconfigure(i, weight=1)

        self.layerLabel = Label(self.root, text="Layers", anchor=CENTER, background="white")
        self.layerLabel.grid(row=0, column=6)

        self.layerList = []
        newLayer = Layer(0, self.layers, self)
        self.layerList.append(newLayer)
        self.currentLayer = 0

        self.testImages = []

        # Set up Painting System

        self.pre_x = None
        self.pre_y = None
        self.line_width = 3.0
        self.color = 'black'
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.root.mainloop()

    # Show Image for debugging

    def showImage(self):
        self.layerList[self.currentLayer].image.show()

    # Clear the current layer
    
    def clear(self):
        self.layerList[self.currentLayer].image = Image.new('RGB', (1080, 720), (255,255,255))
        self.layerList[self.currentLayer].d = ImageDraw.Draw(self.layerList[self.currentLayer].image)
        for line in self.layerList[self.currentLayer].lines:
            self.c.delete(line)
        self.layerList[self.currentLayer].lines.clear()

    # Add a new layer
    
    def add(self):
        i = len(self.layerList)
        newLayer = Layer(i, self.layers, self)
        self.layerList.append(newLayer)
        self.changeLayer(i)

    # Change the current layer
    
    def changeLayer(self, i):
        self.currentLayer = i
        self.layer_name.set("Layer {}".format(self.currentLayer))

    # Reconstruct the scene

    def reconstruct(self):
        clearModels()
        for layer in self.layerList:
            clss = self.classify(layer.image)
            if (clss != -1):
                self.identify(clss, layer.image)

    # Classifying the current image

    def classify(self, im):
        

        with torch.no_grad():

            # Crops the image and gives it a white border

            bg = Image.new(im.mode, im.size, (255,255,255))
            diff = ImageChops.difference(im, bg)
            diff = ImageChops.add(diff, diff, 2.0, -100)
            bbox = diff.getbbox()

            if (bbox == None):
                 return -1

            cropped = im.crop(bbox)
            border = max(cropped.size) // 6
            final = Image.new("RGB", (max(cropped.size) + 2*border,max(cropped.size) + 2*border), (255,255,255))
            final.paste(cropped, (((final.size[0] - cropped.size[0] - 2*border) // 2) + border, ((final.size[1] - cropped.size[1] - 2*border) // 2) + border))

            # Transforms image and runs it through the Classifier
            
            inp = transform(final)
            outputs = classifier(inp.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            print(classes[predicted.item()])
            return predicted.item()

    
    # Use the Comparator on the current image

    def identify(self, clss, im):
        if (clss >= 0):
            with torch.no_grad():
                # Crop and border image
                path = values["datasetFolder"]+"\MODELS\images"
                folder = path+"/"+classes[clss]
                bg = Image.new(im.mode, im.size, (255,255,255))
                diff = ImageChops.difference(im, bg)
                diff = ImageChops.add(diff, diff, 2.0, -100)
                bbox = diff.getbbox()
                x = (bbox[0] + bbox[2]) / 2
                y = (0.2*bbox[1]) + (0.8*bbox[3])
                x -= 540
                y -= 360
                x /= 540
                y /= -360
                size = (abs(bbox[2] - bbox[0]) / 720)
                
                cropped = im.crop(bbox)
                
                final = Image.new("RGB", (max(cropped.size) + 40,max(cropped.size) + 40), (255,255,255))
                final.paste(cropped, (((final.size[0] - cropped.size[0] - 40) // 2) + 40, ((final.size[1] - cropped.size[1] - 40) // 2) + 40))

                # Run the image through the Comparator

                inp = transform2(final)
                point = identifier(inp.unsqueeze(0))
                i = 0
                total = 0.0
                averages = []
                # Run through each image in the class
                imagePaths =  glob.glob(folder+"\*")
                for imagePath in imagePaths:
                    if i > 19:
                        averages.append(total / 20)
                        i -= 20
                        total = 0
                    otherImage = transform2(Image.open(imagePath).convert("RGB"))
                    otherPoint = identifier(otherImage.unsqueeze(0))
                    local = otherPoint - point
                    distance = torch.norm(otherPoint - point)
                    total += distance
                    i += 1
                averages.append(total / 20)
                

                i = numpy.argmin(averages)

                minDist = 100
                minJ = -1

                # Calculate Orientation

                for j in range(10):
                    imagePath = imagePaths[(i*20)+j]
                    otherImage = transform2(Image.open(imagePath).convert("RGB"))
                    otherPoint = identifier(otherImage.unsqueeze(0))
                    local = otherPoint - point
                    distance = torch.norm(otherPoint - point)

                    if (distance < minDist):
                        minJ = j
                        minDist = distance

                path = values["datasetFolder"]+"\MODELS\models-obj"

                modelName = glob.glob(folder+"\*")[i*20].split("\\")[-1].split("_")[0]

                # Re-integrate the Model

                show(path+"\\"+modelName+".obj", x, y, minJ*36, size)

    # Apply Test image
    
    def test(self):
        path = values["datasetFolder"]+"\SKETCHES_TESTING"

        loc = random.choice(values["classes"])
        imageFolder = path+"/"+loc
        imageDest = random.choice(glob.glob(imageFolder+"\*"))
        test = Image.open(imageDest)
        size = 300

        test = test.resize((size,size), Image.LANCZOS)

        arr = numpy.asarray(test.convert('RGBA')).copy()
        arr[:, :, 3] = (255 * (arr[:, :, :3] != 255).any(axis=2)).astype(numpy.uint8)
        test = Image.fromarray(arr)
        testPhoto = ImageTk.PhotoImage(test)
        self.testImages.append(testPhoto)
        x = random.randrange(size//2, 1080-(size//2))
        y = random.randrange(size//2, 720-(size//2))
        
        layer = self.layerList[self.currentLayer]
        layer.lines.append(self.c.create_image(x, y, image=testPhoto))
        layer.image.paste(test, (x-(size//2), y-(size//2)))
             
    # Every to occur when the Mouse is pressed down on the Canvas

    def paint(self, event):
        paint_color = self.color
        if self.pre_x and self.pre_y:
            self.layerList[self.currentLayer].d.line([(int(self.pre_x), int(self.pre_y)), (int(event.x), int(event.y))], fill=paint_color, width=int(self.line_width))
            line = self.c.create_line(self.pre_x, self.pre_y, event.x, event.y,
                                width=self.line_width, fill=paint_color,
                                capstyle=ROUND, smooth=TRUE, splinesteps=36, tags=str(self.currentLayer+2))
            self.layerList[self.currentLayer].lines.append(line)
        self.pre_x = event.x
        self.pre_y = event.y

    def reset(self, event):
        self.pre_x, self.pre_y = None, None

s = canvas(title="Reconstruction", width=1080, height=720, autoscale=False, centre=vec(0,0,-roomHeight/2), range=roomWidth/2)

# Close the program

def close(root):
    if messagebox.askokcancel("Quit", "Are You Sure You Want To Quit?"):
        exit()
        root.destroy()
        s.delete()
        sys.exit()

# Show the model
    
def show(modelPath, posX, posY, angle, size):
    global models
    global s

    if (len(s.objects) >= 65535):
        return 0
    f = []

    model = open(modelPath, "r")

    lines = model.read().split("\n")

    v = []
    n = []
    f = []

    minV = [0, 0, 0]
    maxV = [0, 0, 0]

    vertexCount = 0

    # Parse the OBJ file

    for line in lines:
        e = line.split(" ")
        if (e[0] == "v"):
            v.append(vec(float(e[1]), -float(e[3]), -float(e[2])))
            minV, maxV = minmaxVertices(minV, maxV, [float(e[1]), -float(e[3]), -float(e[2])])
        elif (e[0] == "vn"):
            n.append(vec(float(e[1]), -float(e[3]), -float(e[2])))
        elif (e[0] == "f"):
            if vertexCount < 65533:
                t0 = e[1].split("/")
                t1 = e[2].split("/")
                t2 = e[3].split("/")
                vert0 = vertex(pos=v[int(t0[0])-1], normal=n[int(t0[2])-1])
                vert1 = vertex(pos=v[int(t1[0])-1], normal=n[int(t1[2])-1])
                vert2 = vertex(pos=v[int(t2[0])-1], normal=n[int(t2[2])-1])
                f.append(triangle(vs=[vert0, vert1, vert2]))
                vertexCount += 3

    modelSize = max(maxV[0]-minV[0], maxV[1]-minV[1], maxV[2]-minV[2])

    models.append(compound(f))

    origin = numpy.array([0, 0, 7.79423])
    fov = numpy.pi / 2.0
    direction = numpy.array([posX*numpy.tan(fov/2.0), posY*numpy.tan(fov/2.0)*(2.0/3.0), -1])
    direction = direction / numpy.linalg.norm(direction)

    point = numpy.array([0,0,0])
    normal = numpy.array([0,0,0])

    # Determine the portion of the screen the object is on.

    if (abs(posX) < 0.475 and abs(posY) < 0.475):
        point = numpy.array([0, 0, -roomHeight/2])
        normal = numpy.array([0, 0, 1])
    else:
        if posY > posX:
            if posY > -posX:
                point = numpy.array([0, roomHeight/2, 0])
                normal = numpy.array([0, -1, 0])
            else:
                point = numpy.array([-roomWidth/2, 0, 0])
                normal = numpy.array([1, 0, 0])
        else:
            if posY < -posX:
                point = numpy.array([0, -roomHeight/2, 0])
                normal = numpy.array([0, 1, 0])
            else:
                point = numpy.array([roomWidth/2, 0, 0])
                normal = numpy.array([-1, 0, 0])
    
    # Linear interpolation to find the line to plain intersection
    
    d = numpy.dot(point - origin, normal) / numpy.dot(direction, normal)

    final = origin + d*direction

    dZ = abs(origin[2] - final[2])

    worldSize = (size * dZ * numpy.tan(fov/2))

    final += (worldSize/2)*normal

    models[-1].pos = vec(final[0], final[1], final[2])
    models[-1].rotate(angle=angle, axis=vec(0,1,0))
    models[-1].size = vec(worldSize / modelSize, worldSize / modelSize, worldSize / modelSize)

    return 0

# Clear the current models before reconstruction

def clearModels():
    global models
    global s
    for model in s.objects:
        model.visible = False
        del model
    
    models.clear()

    bottom = box(pos=vec(0,-roomHeight/2, 0), size=vec(roomWidth,0.1,roomHeight))
    left = box(pos=vec(-roomWidth/2, 0, 0), size=vec(0.1,roomHeight,roomHeight))
    right = box(pos=vec(roomWidth/2, 0, 0), size=vec(0.1,roomHeight,roomHeight))
    top = box(pos=vec(0,roomHeight/2, 0), size=vec(roomWidth,0.1,roomHeight))
    back = box(pos=vec(0, 0, -roomHeight/2), size=vec(roomWidth, roomHeight, 0.1))
    return 0

# Determine the minimum and maximum of Vertices

def minmaxVertices(minV, maxV, v):
    for i in range(3):
        minV[i] = min(minV[i], v[i])
        maxV[i] = max(maxV[i], v[i])
    
    return minV, maxV

if __name__ == '__main__':

    # Set up the room scene

    bottom = box(pos=vec(0,-roomHeight/2, 0), size=vec(roomWidth,0.1,roomHeight))
    left = box(pos=vec(-roomWidth/2, 0, 0), size=vec(0.1,roomHeight,roomHeight))
    right = box(pos=vec(roomWidth/2, 0, 0), size=vec(0.1,roomHeight,roomHeight))
    top = box(pos=vec(0,roomHeight/2, 0), size=vec(roomWidth,0.1,roomHeight))
    back = box(pos=vec(0, 0, -roomHeight/2), size=vec(roomWidth, roomHeight, 0.1))
    Drawer()
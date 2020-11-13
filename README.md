---
#Read the original paper (here)[]
---

#INSTRUCTIONS FOR USE OF PROJECT MATERIAL FOR 3D SHAPE RETRIEVAL AND SKETCHED SCENE RECONSTRUCTION USING TRIPLET EMBEDDING

##VALUES

The values.json allows you to set:

- where the Dataset Folder is. At the moment you change it between "SHREC 2013" and "SHREC 2014".
- the paths to the Classifier and Comparator .pth files.
- the classes within the Dataset.
- the name to be appended to the testing results text file.


##IMPORTS

This Project was written using Python 3.7.2

Make sure you have the following Python packages imported, otherwise certain files will not work.

- torch
- torchvision
- torchsummary
- tkinter
- ttkthemes
- numpy
- PIL
- vpython
- models
- matplotlib

The following packages you should have installed with regular Python.

- json
- csv
- os
- sys
- time
- glob
- random
- datetime


##DATASET DOWNLOAD

https://drive.google.com/drive/folders/1HdqE51EP_X47Ri9AR8Xzj6oNrPD5DW7k

You can download each dataset here. Make sure to extract the correct folders. You only need the models, test sketches and training sketches. The following section should teach you how to correctly augment the dataset.

You can download the .pth files I generated for the Networks here:

https://drive.google.com/drive/folders/1DzSoL3XkGTHjW-uNePfZDkXoj7EJ-d03?usp=sharing


##DATASET USE AND AUGMENTATION

In order for the files to work, the database needs to be in the format:

datasetFolder
|
-> MODELS
   |
   -> images
      |
      -> classes[0]
	 |
	 -> mXXX_*.png
	 |
	 -> ...
      |
      -> ...
   |
   -> models
      |
      -> mXXX.off
      |
      -> ...
   |
   -> models-obj
      |
      -> mXXX.obj
      |
      -> ...
   |
   -> categories.csv
|
-> SKETCHES_TESTING
   |
   -> classes[0]
      |
      -> XXXX.png
      |
      -> ...
|
-> SKETCHES_TRAINING
   |
   -> classes[0]
      |
      -> XXXX.png
      |
      -> ...

Where categories.csv must be a .csv file containing the classifications for each model.


Dataset Tools is a folder containing the python files for altering a dataset into this format:

shrecChange.py uses Python's shrec library to move around the dataset. When downloaded, SHREC 2013 & 2014 has a single folder with two seperate folders "test" and "train".
By duplicating this folder and running this file, it will separate out these folders correctly. The file will also reduce the dataset to the set of classes specificed in the values.json.

The following files need to be loaded and ran in Blender's scripting window. Open the Default.blend file to have the correct scene loaded when using them. I used Blender v.2.79b but do not know if it will work for higher versions.

obj_conversation.py creates the models-obj folder by converting all the objects in the models folder

view_creation.py creates the image folder by rendering views of the OFF files.


##NETWORK TRAINING

classifier_train.py will train the Classifier using the values in the values.json file.
comparator_trian.py will train the Comparator using the values in the values.json file.
Change the epoch values by modifying these values but they are current set to the optimum values of this dataset


##NETWORK TESTING

In the Experiment folder you will find the results of the testing, which you can reproduce by running the final_testing.py


##INTERFACE

To use the interface, ensure that the dataset is in the correct format, there are the relevent path files and the room.png is in the same directory.

Then run it in the command line and the interface should load in. You can draw on the current layer by clicking the canvas.

Press the "+" button to add more layers and click on a layer to make it the current layer. 

Press the "Clear" button to clear the current layer.

Press the "Random Test" button to randomly a select a test sketch to put into the scene.

Press the "Reconstruct" button to reconstruct the sketch. Each layer will be treated as an individual sketch.

import shutil, os, glob, json

#path = "SHREC 2013\SHREC13_SBR_TRAINING_SKETCHES"

with open("values.json", "r") as v:
    values = json.load(v)

testpath = os.path.abspath("..")+"/"+values["datasetFolder"]+"/SKETCHES_TESTING"

trainpath = os.path.abspath("..")+"/"+values["datasetFolder"]+"/SKETCHES_TRAINING"

modelpath = os.path.abspath("..")+"/"+values["datasetFolder"]+"/MODELS/images"

def alter():
    for folderPath in glob.glob(testpath+"\*"):
        className = folderPath.split("\\")[-1]
        if className not in values["classes"]:
            if(os.path.isdir(folderPath)):
                shutil.rmtree(folderPath)
    for folderPath in glob.glob(trainpath+"\*"):
        className = folderPath.split("\\")[-1]
        if className not in values["classes"]:
            if(os.path.isdir(folderPath)):
                shutil.rmtree(folderPath)
    for folderPath in glob.glob(modelpath+"\*"):
        className = folderPath.split("\\")[-1]
        if className not in values["classes"]:
            if(os.path.isdir(folderPath)):
                shutil.rmtree(folderPath)

def distribute():

    for folderPath in glob.glob(testpath+"\*"):
        dest = folderPath+"\\test"
        if(os.path.isdir(dest)):
            for src in glob.glob(dest+"\*"):
                shutil.move(src, folderPath)
        if(os.path.isdir(dest)):
            os.rmdir(dest)
        dest = folderPath+"\\train"
        if(os.path.isdir(dest)):
            shutil.rmtree(dest)

    for folderPath in glob.glob(trainpath+"\*"):
        dest = folderPath+"\\train"
        if(os.path.isdir(dest)):
            for src in glob.glob(dest+"\*"):
                shutil.move(src, folderPath)
        if(os.path.isdir(dest)):
            os.rmdir(dest)
        dest = folderPath+"\\test"
        if(os.path.isdir(dest)):
            shutil.rmtree(dest)
    


if __name__ == "__main__":
    option = -1
    while option != 0:
        option = int(input("""
        Which Option Would You Like?
0 - Exit
1 - Distribute Testing and Training Sketches Correctly (SHREC 2013/2014 ONLY)
2 - Alter Current Dataset using Current Classes
"""))
        if option == 1:
            distribute()
        elif option == 2:
            alter()
            

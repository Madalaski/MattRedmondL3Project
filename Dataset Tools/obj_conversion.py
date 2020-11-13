import bpy
import glob
import time
import json
import numpy as np
from datetime import datetime
import os
import csv
startTime = datetime.now()

def main():
    with open(os.path.abspath("..")+"/values.json", "r") as v:
        values = json.load(v)
    PATH = os.path.abspath("..")+"/"+values["datasetFolder"]+"/MODELS/models/"

    with open(os.path.abspath("..")+"/"+values["datasetFolder"]+"/MODELS/categories.csv") as csvfile:
        f = csv.reader(csvfile, delimiter=',')
        count = 0

        for row in f:
            
            bpy.ops.import_mesh.off(filepath=PATH+row[0]+".off")
            bpy.data.objects[row[0]].select = True
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
            bpy.data.objects[row[0]].location = (0,0,0)

            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.editmode_toggle()

            path = os.path.abspath("..")+"/"+values["datasetFolder"]+"/MODELS/models-obj/" + row[0] + ".obj"
            bpy.ops.export_scene.obj(filepath=path)

            for item in bpy.data.meshes:
                bpy.data.scenes[0].objects.unlink(item)
                bpy.data.meshes.remove(item)
            
            count += 1
                
            print("{}% Done".format((count*100)/1259))

    
    

if __name__ == "__main__":
    print(datetime.now() - startTime)
    main()
    print(datetime.now() - startTime)
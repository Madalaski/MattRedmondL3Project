import bpy
import glob
import time
import numpy as np
from datetime import datetime
import os
import csv
import json
startTime = datetime.now()

def setup(scene, c):
    scene.camera.rotation_euler[0] = c[0] * (np.pi / 180.0)
    scene.camera.rotation_euler[1] = c[1] * (np.pi / 180.0)
    scene.camera.rotation_euler[2] = c[2] * (np.pi / 180.0)

    scene.camera.location.x = c[3]
    scene.camera.location.y = c[4]
    scene.camera.location.z = c[5]

    return

def main():
    with open(os.path.abspath("..")+"/values.json", "r") as v:
        values = json.load(v)

    PATH = os.path.abspath("..")+"/"+values["datasetFolder"]+"/MODEL/models/"
    path = os.path.abspath("..")+"/"+values["datasetFolder"]+"/MODEL/images/"

    x, y, z = 1, 2, 3
    bpy.ops.object.lamp_add(type="SUN")
    sun = bpy.data.objects['Sun']
    sun.location[0] = x
    sun.location[1] = y
    sun.location[2] = z

    scene = bpy.data.scenes["Scene"]
    newCamera = bpy.data.objects['Camera']

    newCamera.rotation_mode = 'XYZ'
    scene.camera = newCamera

    scene.render.resolution_percentage = 100
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512

    scene = bpy.context.scene

    with open(os.path.abspath("..")+"/"+values["datasetFolder"]+"/MODELS/categories.csv") as csvfile:
        f = csv.reader(csvfile, delimiter=',')

        for row in f:
            
            bpy.ops.import_mesh.off(filepath=PATH+row[0]+".off")
            bpy.data.objects[row[0]].select = True
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
            bpy.data.objects[row[0]].location = (0,0,0)
            bpy.data.objects[row[0]].data.materials.append(bpy.data.materials["Material"])
                
            for i in range(10):
                x = 0
                y = 0
                z = 0
                pi = np.pi
                x += 3.0 * np.sin(i*((2.0*pi)/10.0))
                z += 3.0 * np.cos(i*((2.0*pi)/10.0))

                cam = [0,i*36,0,x,y,z]
                setup(scene, cam)

                bpy.data.scenes['Scene'].render.filepath = path + row[1] + "\\" + row[0] + "_" + str(i) + ".png"
                bpy.ops.render.render(write_still = True)

            for i in range(5):
                x = 0
                y = 0
                z = 0
                pi = np.pi
                x += 3.0 * (np.sqrt(2.0) / 2.0) * np.sin(i*((2.0*pi)/5.0))
                z += 3.0 * (np.sqrt(2.0) / 2.0) * np.cos(i*((2.0*pi)/5.0))
                y -= 3.0 * (np.sqrt(2.0) / 2.0)

                cam = [45,i*72,0,x,y,z]
                setup(scene, cam)

                bpy.data.scenes['Scene'].render.filepath = path + row[1] + "\\" + row[0] + "_" + str(10+i) + ".png"
                bpy.ops.render.render(write_still = True)

            for i in range(5):
                x = 0
                y = 0
                z = 0
                pi = np.pi
                x += 3.0 * (np.sqrt(2.0) / 2.0) * np.sin(i*((2.0*pi)/5.0))
                z += 3.0 * (np.sqrt(2.0) / 2.0) * np.cos(i*((2.0*pi)/5.0))
                y += 3.0 * (np.sqrt(2.0) / 2.0)

                cam = [-45,i*72,0,x,y,z]
                setup(scene, cam)

                bpy.data.scenes['Scene'].render.filepath = path + row[1] + "\\" + row[0] + "_" + str(15+i) + ".png"
                bpy.ops.render.render(write_still = True)

            for item in bpy.data.meshes:
                bpy.data.meshes.remove(item)

    
    

if __name__ == "__main__":
    
    print(datetime.now() - startTime)

    
    main()
    
    print(datetime.now() - startTime)
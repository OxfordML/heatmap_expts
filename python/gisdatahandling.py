'''
Created on 7 Jan 2015

@author: edwin
'''

import numpy as np
import shapefile
import math
from pyproj import Proj
# import matplotlib.path as path # contains functions for point-in-polygon test

J = 2 # number of target classes
attmap = {1:0, 2:0, 3:1, 4:1, 5:1} #map the attributes in the shape file to target classes

minlat = 18.2#18.0
maxlat = 18.8#19.4
minlon = -72.6#-73.1
maxlon = -72.0#-71.7   
    
nx = 1000
ny = 1000

gridsizex = (maxlat - minlat) / nx
gridsizey = (maxlon - minlon) / ny

tgrid = np.zeros((nx,ny),dtype=np.float) # grid of target values starting from 0

#Treat 0 as default if no shape with another attribute is found in that square.

inputfname = "/home/edwin/Datasets/haiti_unosat/HTI_2010_shp/PDNA_HTI_2010_Atlas_of_Building_Damage_Assessment_UNOSAT_JRC_WB_v2"
outfname_grid_csv = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target2.csv"
outfname_grid_npy = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target2.npy"
outfname_list_csv = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target3.csv"
outfname_list_npy = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target3.npy"  

utm_to_lonlat = Proj("+init=EPSG:32618")    
    
def loadshapes():
    sf = shapefile.Reader(inputfname)
    shapes = sf.shapes() # some kind of python object
    
    print "number of shapes: " + str(len(shapes))
    
    for i, shape in enumerate(shapes):
#         shape.att = sf.record(i)[12][6]
        damagestr = sf.record(i)[12]
        att = [int(s) for s in damagestr.split() if s.isdigit()]
        if len(att)==1:
            shape.att = attmap[att[0]]
        elif "Possible Damage" in damagestr or "No Visible Damage" in damagestr:
            shape.att = 0
            print damagestr + ", " + str(i)
        elif "Substantial Damage" in damagestr or \
        "Heavy Damage" in damagestr or "Destruction" in damagestr: 
            shape.att = 1
            print damagestr + ", " + str(i)
            
#         if sf.record(i)[15] != 'Not yet field validated':
#             print "Valildation: " + sf.record(i)[15]
        shapes[i] = shape
    print "Shapes have been loaded. Now converting to gold-labelled grid."
    return shapes

def create_target_grid(shapes):
    for shape in shapes:
        process_shape(shape)
        
def point_to_grid(x,y):
    lon,lat = utm_to_lonlat(x, y, inverse=True)
    x = math.floor((lat-minlat) / gridsizex)
    y = math.floor((lon-minlon) / gridsizey) 
    return x,y 

def point_to_local_coords(x,y):
    lon,lat = utm_to_lonlat(x, y, inverse=True)
    x = math.floor((lat-minlat) / gridsizex)
    y = math.floor((lon-minlon) / gridsizey) 
    return x,y   
        
def process_shape(shape):
    t = shape.att    
    
    #Assume that there is a class precedence, so that if a grid square has multiple shapes,
    #the one with highest attribute class value will take precedence
    if t==0:
        #default class, lowest value, so can't override anything.
        return
    x,y = point_to_grid(shape.points[0][0], shape.points[0][1])
    if x<0 or x>nx or y<0 or y>ny:
        return
    if tgrid[x,y] < t:
        tgrid[x,y] = t
        print str(x)
        print str(y)

def create_target_list(shapes):
    '''
    Extract a list of coordinates and class values for the damaged buildings.
    '''
    targets = np.zeros((len(shapes),3))
    nshapes = 0
    for shape in shapes:
        x,y = point_to_grid(shape.points[0][0], shape.points[0][1])
        if x<0 or x>nx or y<0 or y>ny:
            continue
        targets[nshapes,0] = x
        targets[nshapes,1] = y
        targets[nshapes,2] = shape.att
        nshapes += 1
        print str(nshapes)
    return targets

if __name__ == '__main__':
    print("Read in a shapefile, and match the attributes to grid locations.")
    
    shapes = loadshapes()
    
    # Save the building locations only, rather than all points in a grid
    targets = create_target_list(shapes)
    print "Saving to numpy binary file."
    np.save(outfname_list_npy, targets)
    print "Saving to CSV text file."
    np.savetxt(outfname_list_csv, targets, delimiter=',', fmt='%f')    
    
    # Create a grid
    create_target_grid(shapes)
    print "Saving to numpy binary file."
    np.save(outfname_grid_npy, tgrid)
    print "Saving to CSV text file."
    np.savetxt(outfname_grid_csv, tgrid, delimiter=',', fmt='%i')
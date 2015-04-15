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
outfname_csv = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target2.csv"
outfname_npy = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target2.npy"
    
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

def createtargets(shapes):
    for shape in shapes:
        process_shape(shape)
        
def point_to_grid(x,y):
    lon,lat = utm_to_lonlat(x, y, inverse=True)
    x = math.floor((lat-minlat) / gridsizex)
    y = math.floor((lon-minlon) / gridsizey) 
    return x,y    

def findgridsquares(shape):
    #Shapes in this data set have actually got only one point
    x,y = point_to_grid(shape.points[0][0], shape.points[0][1])
    return [x],[y]
        
def process_shape(shape):
    t = shape.att    
    
    #Assume that there is a class precedence, so that if a grid square has multiple shapes,
    #the one with highest attribute class value will take precedence
    if t==0:
        #default class, lowest value, so can't override anything.
        return
    xlist, ylist = findgridsquares(shape)
    for i in range(len(xlist)):
        x = xlist[i]
        y = ylist[i]
        if x<0 or x>nx or y<0 or y>ny:
            continue
        if tgrid[x,y] < t:
            tgrid[x,y] = t
            print str(x)
            print str(y)

if __name__ == '__main__':
    print("Read in a shapefile, and match the attributes to grid locations.")
    
    shapes = loadshapes()
    createtargets(shapes)
    
    print "Saving to numpy binary file."
    np.save(outfname_npy, tgrid)
    print "Saving to CSV text file."
    np.savetxt(outfname_csv, tgrid, delimiter=',', fmt='%i')
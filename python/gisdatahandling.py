"""
Created on 7 Jan 2015

@author: edwin
"""

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
    
nx = 100
ny = 100

discrete=False

gridsizex = (maxlat - minlat) / nx
gridsizey = (maxlon - minlon) / ny

tgrid = np.zeros((nx,ny),dtype=np.float) # grid of target values starting from 0
tgrid_totals = np.zeros((nx, ny), dtype=np.int)

#Treat 0 as default if no shape with another attribute is found in that square.

inputfname = "/home/edwin/Datasets/haiti_unosat/HTI_2010_shp/PDNA_HTI_2010_Atlas_of_Building_Damage_Assessment_UNOSAT_JRC_WB_v2"
outfname_grid_csv = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target_grid.csv"
outfname_grid_npy = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target_grid.npy"
outfname_list_csv = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target_list.csv"
outfname_list_npy = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target_list.npy"  

utm_to_lonlat = Proj("+init=EPSG:32618")    
    
def loadshapes():
    sf = shapefile.Reader(inputfname)
    shapes_objs = sf.shapes() # some kind of python object
    
    print "number of shapes: " + str(len(shapes_objs))

    for i, shape in enumerate(shapes_objs):
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
        shapes_objs[i] = shape
    print "Shapes have been loaded. Now converting to gold-labelled grid."
    return shapes_objs

def create_target_grid(shapes_objs):
    for shape in shapes_objs:
        process_shape(shape)

    tgrid[tgrid_totals>0] = tgrid[tgrid_totals>0] / tgrid_totals[tgrid_totals>0]
        
def point_to_grid(x,y):
    lon,lat = utm_to_lonlat(x, y, inverse=True)
    x = (lat-minlat) / gridsizex
    y = (lon-minlon) / gridsizey
    if discrete:
        x = math.floor(x)
        y = math.floor(y)

    return x,y

def point_to_local_coords(x,y):
    lon,lat = utm_to_lonlat(x, y, inverse=True)
    x = (lat-minlat) / gridsizex
    y = (lon-minlon) / gridsizey
    if discrete:
        x = math.floor(x)
        y = math.floor(y)

    return x,y   
        
def process_shape(shape):
    t = shape.att    
    
    #Assume that there is a class precedence, so that if a grid square has multiple shapes,
    #the one with highest attribute class value will take precedence
    if t == 0:
        #default class, lowest value, so can't override anything.
        return
    x, y = point_to_grid(shape.points[0][0], shape.points[0][1])
    if x < 0 or x > nx or y < 0 or y > ny:
        return
    tgrid[x, y] += t
    tgrid_totals[x, y] += 1

def create_target_list(shape_objs):
    """
    Extract a list of coordinates and class values for the damaged buildings.
    """
    target_list = np.zeros((len(shape_objs),3))
    nshapes = 0
    for shape in shape_objs:
        x,y = point_to_grid(shape.points[0][0], shape.points[0][1])
        if x<0 or x>nx or y<0 or y>ny:
            continue
        target_list[nshapes,0] = x
        target_list[nshapes,1] = y
        target_list[nshapes,2] = shape.att
        nshapes += 1
        print str(nshapes)
    target_list = target_list[0:nshapes, :]
    return target_list

def get_unique_targets(all_targets):
    """
    Remove any duplicated target areas; choose highest value attribute.
    """
    print "number of targets: %i" % len(all_targets)
    unique_targets = all_targets.copy()
    for i, t in enumerate(all_targets):
        x = t[0]
        y = t[1]
        dupeidxs = (all_targets[:, 0] == x) & (all_targets[:, 1] == y)
        if np.sum(dupeidxs) > 1:
            unique_targets[dupeidxs, 2] = -1
            unique_targets[i, 2] = np.max(all_targets[dupeidxs, 2])
            print "duplicate of %.3f, %.3f, with damage classes %s" % (x, y, str(all_targets[dupeidxs, 2]))
    print "number of de-duplicated targets: %i" % len(unique_targets)
    unique_targets = unique_targets[unique_targets[:,2]>=0, :]
    return unique_targets

if __name__ == '__main__':
    print("Read in a shapefile, and match the attributes to grid locations.")
    
    shapes = loadshapes()
    
    # Save the building locations only, rather than all points in a grid
    targets = create_target_list(shapes)
    utargets = get_unique_targets(targets)
    print "Saving to numpy binary file."
    np.save(outfname_list_npy, utargets)
    print "Saving to CSV text file."
    np.savetxt(outfname_list_csv, utargets, delimiter=',', fmt='%f')    
    
    # Create a grid
    create_target_grid(shapes)
    print "Saving to numpy binary file."
    np.save(outfname_grid_npy, tgrid)
    print "Saving to CSV text file."
    np.savetxt(outfname_grid_csv, tgrid, delimiter=',', fmt='%i')
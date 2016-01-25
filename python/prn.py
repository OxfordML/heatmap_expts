'''
Created on 22 Jan 2016

@author: edwin

Read in the PRN dataset. 

Chop it into even grid sizes and run IBCC:
- show with all data.
- with data from each resolution category.

Visualise as a heatmap.

Compare with Heatmaps:
- using the same grid
- not using a grid at all

'''
import pandas as pd
import numpy as np
from ibcc import IBCC

if __name__ == '__main__':
    datafile = "./data/prn_pilot_classifications_onerowpermark_withsubjectinfo.csv"
    outdir = "./data/prn_out"
    
    dataframe = pd.read_csv(datafile, usecols=["user_id", "resolution_level", "lat_mark", "long_mark", "lat_lowerleft", 
                               "long_lowerleft", "lat_upperright", "long_upperright", "mark_type", "damage_assessment"])
    
    latmarks = dataframe["lat_mark"]
    longmarks = dataframe["long_mark"]
    
    nx = 100.0
    ny = 100.0
    
    N = latmarks.shape[0]
        
    marks = np.array(dataframe["mark_type"].tolist())
    
    nomarkidxs = np.argwhere(marks=='none').flatten()
    markidxs = np.argwhere(marks!='none').flatten()
    
    nmarks = np.sum(markidxs)
    gridx = np.zeros(nmarks)
    gridy = np.zeros(nmarks)
    
    maxlat = np.max(latmarks[markidxs])
    minlat = np.min(latmarks[markidxs])
    maxlon = np.max(longmarks[markidxs])
    minlon = np.min(longmarks[markidxs])
    
    def convert_lat_lon_to_grid(lat, lon):
        x = (lat - minlat) / (maxlat - minlat) * nx
        y = (lon - minlon) / (maxlon - minlon) * ny    
        return x, y
    
    gridx, gridy = convert_lat_lon_to_grid(latmarks[markidxs], longmarks[markidxs])
    
    marktypes, marks_num_original = np.unique(marks, return_inverse=True)
    marks_num = marks_num_original[markidxs]
    marktypes_nomark = 'none'
    marks = marks[markidxs]
    
    damagelevels = np.array(dataframe["damage_assessment"].tolist())
    idxs = np.argwhere(1 - np.isnan(damagelevels))
    damagelevels[np.argwhere(np.isnan(damagelevels))] = np.max(damagelevels[idxs]) + 1 # new values for the points where damage level not specified
    
    userids, users_original = np.unique(dataframe["user_id"], return_inverse=True)
    users = users_original[markidxs]
    # need to combine with resolution level, e.g. + reslevel * n workers -- for now we can avoid splitting.
    
    # bottom left grid squares for the none marks
    bl_gridx, bl_gridy = convert_lat_lon_to_grid( np.array(dataframe["lat_lowerleft"].tolist())[nomarkidxs], 
                                                  np.array(dataframe["long_lowerleft"].tolist())[nomarkidxs])
    # top right
    tr_gridx, tr_gridy = convert_lat_lon_to_grid(np.array(dataframe["lat_upperright"].tolist())[nomarkidxs], 
                                                 np.array(dataframe["long_upperright"].tolist())[nomarkidxs])
    
    n_nomarks = np.sum((tr_gridx - bl_gridx + 1) * (tr_gridy - bl_gridy + 1))
    gridx_nomarks = np.zeros(n_nomarks)
    
    for i in range(len(bl_gridx)):
        print "Processing no markings: %i out of %i" % (i, len(bl_gridx))
        # for each none mark, turn into a set of markings for all grid squares covered
        xvals = np.arange(bl_gridx[i], tr_gridx[i]+1)[:, np.newaxis]
        yvals = np.arange(bl_gridy[i], tr_gridy[i]+1)[:, np.newaxis]
        n_i = len(xvals) * len(yvals)
        
        gridx_i = np.tile(xvals, (len(yvals), 1)).reshape(-1)
        gridy_i = np.tile(yvals, (1, len(xvals))).reshape(-1)
        
        gridx = np.concatenate((gridx, gridx_i), axis=0)
        gridy = np.concatenate((gridy, gridy_i), axis=0)
        
        user_i = users_original[nomarkidxs][i]
        
        marks = np.concatenate((marks, np.zeros(n_i) + marktypes_nomark))
        users = np.concatenate((users, np.zeros(n_i) + user_i))
        damagelevels = np.concatenate((damagelevels, np.zeros(n_i)))
    
    # SEPARATE MARKING TYPES ------------------------------------------------------------------------------------------
    # Separate data into different datasets for each marking type, including each damage assessment level.
    
    marktypekeys = np.unique(marktypes) #['blocked_road', 'crowd', 'flooding', 'none', 'structural_damage', 'tarp']
    damagekeys = np.unique(damagelevels)

    testkeys = marktypekeys#[marktypekeys != 'structural_damage']
    #for dk in damagekeys:
    #    testkeys = np.concatenate((testkeys, 'structural_damage_' + dk))
        
    C = {}
    
    for tk in testkeys:
        idxs = (marks==tk) | (marks==marktypes_nomark)
        values = (marks[idxs] == tk)
        if tk=='structural_damage':
            values[values] = damagelevels[idxs][values]
        values = values[:, np.newaxis]
        C[tk] = np.concatenate((users[idxs][:, np.newaxis], gridx[idxs][:, np.newaxis], gridy[idxs][:, np.newaxis], values), axis=1)  
    
    # GOLD ------------------------------------------------------------------------------------------------------------
    # Run IBCC separately with each mark type and generate "gold standard" by using all available labels.        
    nclasses = 2
    nscores = 2 # except for damage assessment, where we need to allow any level + no level given
        
    ibcccombiner = IBCC(nclasses, nscores, alpha0, nu0, K, uselowerbound, dh)
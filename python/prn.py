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
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.sparse import coo_matrix

nx = 132.0#264.0
ny = 92.0#184.0
            
goldfile = './data/prn/prn_gold_%s_%s.npy'
Cfile = './data/prn/prn_crowdlabels_%s_%i.npy'

testparams_nu0 = [1, 10, 100, 1000]
bestnu0 = 100

if __name__ == '__main__':
    
    plot_results = True
    
    datafile = "./data/prn_pilot_classifications_onerowpermark_withsubjectinfo.csv"
    outdir = "./data/prn_out"
    
    dataframe = pd.read_csv(datafile, usecols=["user_id", "resolution_level", "lat_mark", "long_mark", "lat_lowerleft", 
                               "long_lowerleft", "lat_upperright", "long_upperright", "mark_type", "damage_assessment"])
    
    latmarks = dataframe["lat_mark"]
    longmarks = dataframe["long_mark"]
    
    N = latmarks.shape[0]
        
    marks = np.array(dataframe["mark_type"].tolist())
    
    wholesqidxs = np.argwhere((marks!='none') & (np.isnan(latmarks)) ).flatten() # the locations where the marking applies to the whole image
    nomarkidxs = np.argwhere(marks=='none').flatten()
    markidxs = np.argwhere((marks!='none') & (1 - np.isnan(latmarks)) ).flatten()
    
    nmarks = np.sum(markidxs)
    gridx = np.zeros(nmarks)
    gridy = np.zeros(nmarks)
    
    
    goldres = dataframe["resolution_level"] < 5
    maxlat = np.max(dataframe["lat_upperright"][goldres])
    minlat = np.min(dataframe["lat_lowerleft"][goldres])
    maxlon = np.max(dataframe["long_upperright"][goldres])
    minlon = np.min(dataframe["long_lowerleft"][goldres])
    
    # make sure the values all fit inside the grid, not on the edge
    maxlat += (maxlat - minlat) / nx * 0.0001
    maxlon += (maxlon - minlon) / ny * 0.0001
    
    def convert_lat_lon_to_grid(lat, lon):
        
        x = (lat - minlat) / (maxlat - minlat) * nx
        y = (lon - minlon) / (maxlon - minlon) * ny    
        return x, y
    
    gridx, gridy = convert_lat_lon_to_grid(latmarks[markidxs], longmarks[markidxs])
    
    marktypes, marks_num_original = np.unique(marks, return_inverse=True)
    marks_num = marks_num_original[markidxs]
    marktypes_nomark = np.zeros(len(nomarkidxs), dtype=str)
    marktypes_nomark[:] = 'none'
    marks = marks[markidxs]
    
    damagelevels = np.array(dataframe["damage_assessment"].tolist())
    idxs = np.argwhere(1 - np.isnan(damagelevels))
    damagelevels[np.argwhere(np.isnan(damagelevels))] = np.max(damagelevels[idxs]) + 1 # new values for the points where damage level not specified
    
    userids, users_original = np.unique(dataframe["user_id"], return_inverse=True)
    users = users_original[markidxs]
    # need to combine with resolution level, e.g. + reslevel * n workers -- for now we can avoid splitting.
    
    reslevels = dataframe["resolution_level"]
    reslevelgroups = {}
    reslevelgroups[0] = np.arange(1, 5)
    reslevelgroups[1] = np.arange(5, 10)
    reslevelgroups[2] = np.arange(10, 40)
    reslevelgroups[3] = np.arange(40, 70)
    
    
    def add_whole_sq_labels(nomarkidxs, marktypes_nomark, marks, users, damagelevels, gridx, gridy):
    
        # bottom left grid squares for the none marks
        bl_gridx, bl_gridy = convert_lat_lon_to_grid( np.array(dataframe["lat_lowerleft"].tolist())[nomarkidxs], 
                                                  np.array(dataframe["long_lowerleft"].tolist())[nomarkidxs])
        # top right
        tr_gridx, tr_gridy = convert_lat_lon_to_grid(np.array(dataframe["lat_upperright"].tolist())[nomarkidxs], 
                                                 np.array(dataframe["long_upperright"].tolist())[nomarkidxs])
    
        bl_gridx = np.round(bl_gridx).astype(int)
        bl_gridx[bl_gridx<0] = 0
        
        bl_gridy = np.round(bl_gridy).astype(int)
        bl_gridy[bl_gridy<0] = 0
        
        tr_gridx = np.round(tr_gridx).astype(int)
        tr_gridx[tr_gridx>=nx] = nx - 1
        
        tr_gridy = np.round(tr_gridy).astype(int)
        tr_gridy[tr_gridy>=ny] = ny - 1
    
        nlowres = 0
        
        smallest_sq = nx*ny
    
        for i in range(len(bl_gridx)):
            # for each none mark, turn into a set of markings for all grid squares covered
            xvals = np.arange(bl_gridx[i], tr_gridx[i])[:, np.newaxis]
            yvals = np.arange(bl_gridy[i], tr_gridy[i])[:, np.newaxis]
            n_i = len(xvals) * len(yvals)
            if n_i < smallest_sq:
                smallest_sq = n_i
                
            #print "Processing no markings: %i out of %i" % (i, len(bl_gridx))
            #print "Resolution level: %i, adding %i new labels" % (dataframe["resolution_level"][nomarkidxs[i]], n_i)
            
            if n_i > 20:
                nlowres += 1
            
            gridx_i = np.tile(xvals, (len(yvals), 1)).reshape(-1)
            gridy_i = np.tile(yvals, (1, len(xvals))).reshape(-1)
            
            gridx = np.concatenate((gridx, gridx_i))
            gridy = np.concatenate((gridy, gridy_i))
            
            user_i = users_original[nomarkidxs][i]
            
            newmarks = np.zeros(n_i, dtype=str)
            newmarks[:] = marktypes_nomark[i]
            marks = np.concatenate((marks, newmarks))
            users = np.concatenate((users, np.zeros(n_i) + user_i))
            damagelevels = np.concatenate((damagelevels, np.zeros(n_i)))
            
            
        print "Smallest image we found covered %i squares" % smallest_sq
        return marks, users, damagelevels, gridx, gridy
    
    marktypekeys = np.unique(marktypes) #['blocked_road', 'crowd', 'flooding', 'none', 'structural_damage', 'tarp']
    damagekeys = np.unique(damagelevels)
    damage_map = {}
    damage_map[0] = 0
    damage_map[0.5] = 1
    damage_map[1] = 2
    damage_map[2] = 3    
    
    testkeys = marktypekeys#[marktypekeys != 'structural_damage']
            
    gold = {} 
    gold_x = {}
    gold_y = {}
    gold_idxs = {} # indexes into the gold results set that indicate where we had > 3 labels per data point
    gold_linearidxs = {}
    results = {}
    result_idxs = {}
    aucs = {}
    
    sd_use_all = False#True
    
    reslevelidxs = {}
    # RUN EVERYTHING SEPARATELY FOR EACH RESOLUTION GROUP
    for r in range(1):#4):
        print "" # leave a blank line
        
        reslevelidxs[r] = np.in1d(reslevels.tolist(), reslevelgroups[r])
    
        marks_r = marks[reslevelidxs[r][markidxs]]
        nomarkidxs_r = nomarkidxs[reslevelidxs[r][nomarkidxs]]
        marktypes_nomark_r = marktypes_nomark[reslevelidxs[r][nomarkidxs]]
        users_r = users[reslevelidxs[r][markidxs]]
        damagelevels_r = damagelevels[reslevelidxs[r][markidxs]]
        gridx_r = gridx[reslevelidxs[r][markidxs]]
        gridy_r = gridy[reslevelidxs[r][markidxs]]
        wholesqidxs_r = wholesqidxs[reslevelidxs[r][wholesqidxs]]
    
        outsideidxs = np.array(((gridx_r < nx) & (gridx_r >= 0) & (gridy_r < ny) & (gridy_r >= 0)).tolist())
        marks_r = marks_r[outsideidxs]
        users_r = users_r[outsideidxs]
        damagelevels_r = damagelevels_r[outsideidxs]
        gridx_r = gridx_r[outsideidxs]
        gridy_r = gridy_r[outsideidxs]
    
        marks_r, users_r, damagelevels_r, gridx_r, gridy_r = add_whole_sq_labels(nomarkidxs_r, marktypes_nomark_r, marks_r, users_r, 
                                                         damagelevels_r, gridx_r, gridy_r)
        marks_r, users_r, damagelevels_r, gridx_r, gridy_r = add_whole_sq_labels(wholesqidxs_r, dataframe["mark_type"][wholesqidxs_r].tolist(), 
                                                         marks_r, users_r, damagelevels_r, gridx_r, gridy_r)
        
        # SEPARATE MARKING TYPES ------------------------------------------------------------------------------------------
        # Separate data into different datasets for each marking type, including each damage assessment level.
        
        C = {}
        
        gridx_r = np.floor(gridx_r).astype(int)
        gridy_r = np.floor(gridy_r).astype(int)
        
        for tk in testkeys:
            idxs = (marks_r==tk) | (marks_r==marktypes_nomark_r[0])
            values = (marks_r[idxs] == tk).astype(int)
            if tk=='structural_damage':
                values = np.array([damage_map[d] for d in damagelevels_r[idxs]])
                values = values[:, np.newaxis]
                for d in range(1, 4):
                    if not sd_use_all:
                        didxs = (values==d) | (values==0)
                        didxs = didxs.reshape(-1)
                        values_d = values[didxs, :] == d
                        values_d = values_d.astype(int)    
                        C[tk + '_' + str(d)] = np.concatenate((users_r[idxs][didxs][:, np.newaxis], 
                                                        gridx_r[idxs][didxs][:, np.newaxis], 
                                                        gridy_r[idxs][didxs][:, np.newaxis], values_d), axis=1).astype(int)
                    else:
                        C[tk + '_' + str(d)] = np.concatenate((users_r[idxs][:, np.newaxis], 
                                                        gridx_r[idxs][:, np.newaxis], 
                                                        gridy_r[idxs][:, np.newaxis], values), axis=1).astype(int)
                    np.save(Cfile % (tk + '_' + str(d), r), C[tk + '_' + str(d)])
            else:
#                 continue
                values = values[:, np.newaxis]
                C[tk] = np.concatenate((users_r[idxs][:, np.newaxis], gridx_r[idxs][:, np.newaxis], 
                                gridy_r[idxs][:, np.newaxis], values), axis=1).astype(int)
            
                np.save(Cfile % (tk, r), C[tk])
        # GOLD ------------------------------------------------------------------------------------------------------------
        # Run IBCC separately with each mark type and generate "gold standard" by using all available labels.      
        results[r] = {}
        result_idxs[r] = {}  
        for tk in C.keys():
            if tk=='none':
                continue
            if tk=='structural_damage':
                nclasses = 4
                nscores = 4
                
                alpha0 = np.ones((nclasses, nscores), dtype=float)
                alpha0[range(nclasses), range(nscores)] = 5
                alpha0[0, 0] = 19
                alpha0[1:, 0] = 15                
            elif tk=='structural_damage_1' and sd_use_all:
                nclasses = 2
                nscores = 4
                 
                alpha0 = np.ones((nclasses, nscores), dtype=float)
                alpha0[1, 1] = 2
                alpha0[0, 0] = 2
                
                #alpha0[1, 1] = 5
                #alpha0[0, 0] = 19
                #alpha0[1:, 0] = 15
                
                #alpha0[1:, 0] = 14
                #alpha0[1, 2:] = 1.5
                 
            elif tk=='structural_damage_2' and sd_use_all:
                nclasses = 2
                nscores = 4
                 
                alpha0 = np.ones((nclasses, nscores), dtype=float)
                alpha0[1, 2] = 2
                alpha0[0, 0] = 2
                
                #alpha0[1, 2] = 5
                #alpha0[0, 0] = 19
                #alpha0[1:, 0] = 15
                
                #alpha0[1:, 0] = 14
                #alpha0[1, 1] = 1.5
                #alpha0[1, 3] = 1.5
                                
            elif tk=='structural_damage_3' and sd_use_all:
                nclasses = 2
                nscores = 4
                                 
                alpha0 = np.ones((nclasses, nscores), dtype=float)
                alpha0[1, 3] = 2
                alpha0[0, 0] = 2#19
                #alpha0[1:, 0] = 15                
                #alpha0[1:, 0] = 14
                #alpha0[1, 1] = 1.5
                #alpha0[1, 2] = 1.5
                #alpha0[0, 1:3] = 2
                            
#             elif not 'structural_damage' in tk:
#                 continue
            else:
                nclasses = 2
                nscores = 2 # except for damage assessment, where we need to allow any level + no level given
                                
                alpha0 = np.ones((nclasses, nscores), dtype=float)
                alpha0[range(nclasses), range(nscores)] = 5
                alpha0[0, 0] = 19
                alpha0[1:, 0] = 15
            
            nu0 = np.ones(nclasses, dtype=float) # assume that damage classes are relatively rare
                      
            K = len(np.unique(C[tk][:, 0])) 
            
            #flatten the input data so it can be used with standard IBCC
            linearIdxs = np.ravel_multi_index((C[tk][:, 1], C[tk][:, 2]), dims=(nx, ny))
            C_flat = C[tk][:,[0,1,3]]            
            linear_unique, linear_source, linear_inv = np.unique(linearIdxs, return_index=True, return_inverse=True)
            C_flat[:,1] = linear_inv
            
            labelcounts = coo_matrix((np.ones(len(linear_inv)), (linear_inv, np.zeros(len(linear_inv)))))
            labelcounts = labelcounts.todense()
            std_counts = np.std(labelcounts)
            mincounts = np.min(labelcounts)
            goldlevelcounts = np.sum(labelcounts >= 3)
            print "Resolution level %i has %i classifications for %s" % (r, C_flat.shape[0], tk)
            print "Mean number of labels per square: %f. STD: %f. Min counts: %i. Data points with >= 3 labels: %i" \
                % (linearIdxs.shape[0] / float(linear_unique.shape[0]), std_counts, mincounts, goldlevelcounts )
                   
#             maxlb = -np.inf
#             for val in testparams_nu0:
#                 nu0[:] = val
#                 ibcccombiner = IBCC(nclasses, nscores, alpha0, nu0, K, uselowerbound=True)
#                 ibcccombiner.combine_classifications(C_flat, testidxs=np.ones(len(linear_unique)))
#                 lb = ibcccombiner.lowerbound()
#                 print "Nu0 = %f, lb=%.4f" % (val, lb) 
#                 if lb > maxlb:
#                     maxlb = lb
#                     bestnu0 = val
            #nu0[0] = 10
            nu0[:] = bestnu0#1000
            
            ibcccombiner = IBCC(nclasses, nscores, alpha0, nu0, K, uselowerbound=True)            
            results[r][tk] = ibcccombiner.combine_classifications(C_flat, testidxs=np.ones(len(linear_unique)))
            result_idxs[r][tk] = np.array((labelcounts>=3).T.tolist()[0], dtype=bool)
            
            if r == 0:
                print "Establishing gold from %i predictions using %i crowdsourced labels" % (len(results[r][tk]), len(C_flat))
                gold[tk] = results[r][tk]
                gold_x[tk] = C[tk][linear_source, 1][:, np.newaxis]
                gold_y[tk] = C[tk][linear_source, 2][:, np.newaxis]            
                gold_idxs[tk] = np.array((labelcounts>=3).T.tolist()[0], dtype=bool)
                gold_linearidxs[tk] = linear_unique
            else:                
                # classification accuracy
                goldlabels = np.argmax(gold[tk], axis=1)
                labels_r = np.argmax(results[r][tk], axis=1)
                
                # use only locations where both gold and the test resolution have > 3 labels
                idxs = np.intersect1d(gold_linearidxs[tk][gold_idxs[tk]], linear_unique[result_idxs[r][tk]])
                goldtestidxs = np.in1d(gold_linearidxs[tk], idxs)
                rtestidxs = np.in1d(linear_unique, idxs)
                
                print "No. test idxs = %i" % len(idxs)
                
                goldlabels = goldlabels[goldtestidxs]
                labels_r = labels_r[rtestidxs]
                
                goldprobs = gold[tk][goldtestidxs, 1]
                probs_r= results[r][tk][rtestidxs, 1]
                
                acc = np.sum(goldlabels == labels_r) / float(len(labels_r))
                
                print "Classification accuracy at resolution group %i = %f" % (r, acc)
                
                brier = np.sqrt(np.mean((probs_r - goldprobs)**2))
                
                print "Brier score (root mean squared error in probabilities) at res group %i = %f" % (r, brier)
            
                auc = roc_auc_score(goldlabels, probs_r)
                if not tk in aucs:
                    aucs[tk] = []
                aucs[tk].append(auc)
                print "AUC for resolution group %i = %f" % (r, auc)
            
#                 #run again using gold labelled data only to test accuracy of workers
#                 ibcccombiner = IBCC(nclasses, nscores, alpha0, nu0, K, uselowerbound=True)        
#                 accidxs = np.in1d(C_flat[:, 1], idxs)
#                 C_acc = C_flat[accidxs, :]
#                 trainidxs = goldlabels[goldtestidxs]                
#                 ibcccombiner.combine_classifications(C_acc, trainidxs=trainidxs)
                
            
            # Visualise the gold
            savedir = './prn_plots/bias' # bias -- use priors that assume that people are biased toward giving stronger labels

            if not os.path.exists(savedir):
                os.mkdir(savedir)

            if tk=='none':
                continue
            
            if plot_results:
                plt.figure()
                plt.title('Classifications for %s' % tk)
            
                labels = np.argmax(results[r][tk], axis=1)
                labels = labels[result_idxs[r][tk]]
                zero_idxs = labels == 0
                zero_x = C[tk][linear_source, 1][:, np.newaxis][result_idxs[r][tk]][zero_idxs]
                zero_y = C[tk][linear_source, 2][:, np.newaxis][result_idxs[r][tk]][zero_idxs]
                    
                plt.scatter(zero_x, zero_y, color='darkgreen')
                print "For %s plotting %i negative points" % (tk, len(zero_x))
                
                # get the coordinates of locations where problems were identified
                if tk=='structural_damage':
                    c = ['b', 'y', 'r']
                    for i in range(1,4):
                        nonzero_idxs = labels == i
                        nonzero_x = C[tk][linear_source, 1][:, np.newaxis][nonzero_idxs]
                        nonzero_y = C[tk][linear_source, 2][:, np.newaxis][nonzero_idxs]
                    
                        plt.scatter(nonzero_x, nonzero_y, color=c[i-1])
                        print "For %s plotting %i positive points" % (tk, len(nonzero_x))
                else:                        
                    nonzero_idxs = labels > 0
                    nonzero_x = C[tk][linear_source, 1][:, np.newaxis][nonzero_idxs]
                    nonzero_y = C[tk][linear_source, 2][:, np.newaxis][nonzero_idxs]
                    
                    plt.scatter(nonzero_x, nonzero_y, color='y')
                    print "For %s plotting %i positive points" % (tk, len(nonzero_x))
                    
                plt.savefig(savedir + '%s_gold_%i_res_%i.eps' % (tk, sd_use_all, r))
            
    for tk in aucs:
        print "AUCs for %s" % tk
        for r in range(len(aucs[tk])):
            print "Resolution group %i: %.3f" % (r+1, aucs[tk][r])
    
    for tk in gold:
        np.save(goldfile % ('p', tk), gold[tk])
        np.save(goldfile % ('x', tk), gold_x[tk])
        np.save(goldfile % ('y', tk), gold_y[tk])            
        np.save(goldfile % ('idxs', tk), gold_idxs[tk])
        np.save(goldfile % ('linearidxs', tk), gold_linearidxs[tk])
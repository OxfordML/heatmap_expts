'''
Created on 8 Jan 2015

@author: edwin
'''

import logging
import numpy as np
from heatmapbcc import HeatMapBCC
from ibccperformance import Evaluator
from ibcc import IBCC
from scipy.sparse import coo_matrix

class UshahidiDataHandler(object):
    
    K = 1
    nx = 1000
    ny = 1000
    
    minlat = 18.2#18.0
    maxlat = 18.8#19.4
    minlon = -72.6#-73.1
    maxlon = -72.0#-71.7   
    
    datadir = '../../HeatMapBCC/data'
    
    C = []
    
    #Dictionary defining how we map original categories 
    #to class IDs, using original categories as keys:
    categorymap = {1:1, 3:1,5:{'a':1},6:{'a':1,'b':1}} 
    
    def __init__(self, nx, ny, datadir):
        self.nx = nx
        self.ny = ny
        self.datadir = datadir
        
    def translate_points_to_local(self, latdata, londata):
        logging.debug('Translating original coords to local values.')
            
        latdata = np.float64(latdata)
        londata = np.float64(londata)
        
        normlatdata = (latdata-self.minlat)/(self.maxlat-self.minlat)
        normlondata = (londata-self.minlon)/(self.maxlon-self.minlon)    
            
        latdata = np.array(np.round(normlatdata*self.nx), dtype=np.int)
        londata = np.array(np.round(normlondata*self.ny), dtype=np.int)
            
        return latdata,londata           
    
    def load_ush_data(self):
        dataFile = self.datadir+'/exported_ushahidi.csv'
        self.K = 1
        #load the data
    #     reportIDdata = np.genfromtxt(dataFile, np.str, delimieter=',', skip_header=True, usecols=[])
    #     datetimedata = np.genfromtxt(dataFile, np.datetime64, delimiter=',', skip_header=True, usecols=[2,3])
        latdata = np.genfromtxt(dataFile, np.float64, delimiter=',', skip_header=True, usecols=[4])
        londata = np.genfromtxt(dataFile, np.float64, delimiter=',', skip_header=True, usecols=[5])
        reptypedata = np.genfromtxt(dataFile, np.str, delimiter=',', skip_header=True, usecols=[1])
        rep_list_all = np.genfromtxt(dataFile, np.str, delimiter=',', skip_header=True, usecols=[6])
        latdata,londata = self.translate_points_to_local(latdata,londata)
        rep_id_grid = {}

        rep_list = {}
        C = {}            
        instantiatedagents = []

        for i, reptypetext in enumerate(reptypedata):        
            typetoks = reptypetext.split('.')
            typetoks = typetoks[0].split(',')
            
            if "1a." in reptypetext:
                agentID = 0
                maintype = 1
            elif "1b." in reptypetext:
                agentID = 1
                maintype = 1
            elif "1c." in reptypetext:
                agentID = 2
                maintype = 1
            elif "1d." in reptypetext:
                agentID = 3
                maintype = 1
            elif "3a." in reptypetext:
                agentID = 4
                maintype = 1
            elif "3b." in reptypetext:
                agentID = 5
                maintype = 1
            elif "3c." in reptypetext:
                agentID = 6
                maintype = 1
            elif "3d." in reptypetext:
                agentID = 7
                maintype = 1
            elif "3e." in reptypetext:
                agentID = 8
                maintype = 1
            elif "5a." in reptypetext:
                agentID = 9
                maintype = 1
            elif "6a." in reptypetext:
                agentID = 10
                maintype = 1
            elif "6b." in reptypetext:
                agentID = 11
                maintype = 1
            elif "1." in reptypetext:
                agentID = 12
                maintype = 1
            elif "3." in reptypetext:
                agentID = 13
                maintype = 1  
            else: #we don't care about these categories in the demo anyway, but can add them easily here
                agentID = 14
                for typestring in typetoks:
                    if len(typestring)>1:
                        sectype = typestring[1] # second character should be a letter is available
                    else:
                        sectype = 0
                    try:
                        maintype = int(typestring[0]) #first character should be a number
                    except ValueError:
                        logging.warning('Not a report category: ' + typestring)
                        continue                   
                    if maintype in self.categorymap:
                        mappedID = self.categorymap[maintype]
                        if type(mappedID) is dict:
                            if sectype in mappedID:
                                maintype = mappedID[sectype]
                        else:
                            maintype = mappedID

            repx = latdata[i]
            repy = londata[i]
            if repx>=self.nx or repx<0 or repy>=self.ny or repy<0:
                continue
            
            try:
                Crow = np.array([agentID, repx, repy, 1]) # all report values are 1 since we only have confirmations of an incident, not confirmations of nothing happening
            except ValueError:
                logging.error('ValueError creating a row of the crowdsourced data matrix.!')        

            if C=={} or maintype not in C:
                C[maintype] = Crow.reshape((1,4))
                rep_id_grid[maintype] = np.empty((self.nx, self.ny), dtype=np.object)
                rep_list[maintype] = [rep_list_all[i]]
            else:
                C[maintype] = np.concatenate((C[maintype], Crow.reshape(1,4)), axis=0)
                rep_list[maintype].append(rep_list_all[i])
                
            if rep_id_grid[maintype][repx, repy] == None:
                rep_id_grid[maintype][repx, repy] = []
            rep_id_grid[maintype][repx, repy].append(len(rep_list[maintype])-1)               
            
            if agentID not in instantiatedagents:
                instantiatedagents.append(agentID)
                
        instantiatedagents = np.array(instantiatedagents)
        for j in C.keys():
            for r in range(C[j].shape[0]):
                C[j][r,0] = np.argwhere(instantiatedagents==C[j][r,0])[0,0]
        self.K = len(instantiatedagents)
                     
        self.C = C
        print "Number of type one reports: " + str(self.C[1].shape[0])

if __name__ == '__main__':
    print "Run tests to determine the accuracy of the bccheatmaps method."
    
    #LOAD THE DATA to run unsupervised learning tests ------------------------------------------------------------------
    #Load up some ground truth
#     goldfile = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target_182_188_726_720_100_100.csv"
#     tgrid = np.genfromtxt(goldfile).astype(int)
    goldfile = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target2.npy"
    tgrid = np.load(goldfile).astype(int)    
    
    nx = 1000
    ny = 1000
    datahandler = UshahidiDataHandler(nx,ny, '/home/edwin/Datasets/ushahidi/')
    datahandler.load_ush_data()
    K = datahandler.K
    print "No. report types = %i" % K
    C = datahandler.C[1]
    
    # default hyperparameters
    alpha0 = np.array([[2.0, 1.0], [1.0, 2.0]])
    nu0 = np.array([5,1])    
    
    # containers for results
    results = {}
    
    # RUN HEATMAP BCC --------------------------------------------------------------------------------------------------
    combiner = HeatMapBCC(nx, ny, 2, 2, alpha0, nu0, K, calc_full_grid=True)
    combiner.minNoIts = 5
    combiner.maxNoIts = 200
    combiner.convThreshold = 0.1

    # Need to replace with optimised version!
    bcc_pred = combiner.combine_classifications(C)
    bcc_pred = bcc_pred[1,:,:].reshape((nx,ny)) # only interested in positive "damage class"
    
    results['heatmapbcc'] = bcc_pred
    
    # RUN SEPARATE IBCC AND GP STAGES ----------------------------------------------------------------------------------
    
    # run standard IBCC
    combiner = IBCC(2, 2, alpha0, nu0, K)
    combiner.minNoIts = 5
    combiner.maxNoIts = 200
    combiner.convThreshold = 0.1

    bcc_pred = combiner.combine_classifications(C)

    # use IBCC output to train GP
    combiner = HeatMapBCC(nx, ny, 2, 2, alpha0, nu0, K, calc_full_grid=True)
    combiner.minNoIts = 1
    combiner.maxNoIts = 1
    combiner.E_t = bcc_pred
    bcc_pred = combiner.combine_classifications(C)
    
    bcc_pred = bcc_pred[1,:,:].reshape((nx,ny)) # only interested in positive "damage class"
    
    results['IBCC_then_GP'] = bcc_pred
    
    # TRAIN GP WITHOUT BCC ---------------------------------------------------------------------------------------------
    
    # get values for training points by taking frequencies
    counts_pos = coo_matrix((C[:,3]==1, (C[:,1], C[:,2])), (nx,ny))
    counts_neg = coo_matrix((C[:,3]==0, (C[:,1], C[:,2])), (nx,ny))
    density_estimates = counts_pos / (counts_pos + counts_neg)
    
    E_t_pos = density_estimates[(counts_pos+counts_neg)>0]
    E_t_pos = E_t_pos.reshape((E_t_pos.size,1))
    E_t_neg = 1-E_t_pos
        
    #run GP
    combiner = HeatMapBCC(nx, ny, 2, 2, alpha0, nu0, K, calc_full_grid=True)
    combiner.minNoIts = 1
    combiner.maxNoIts = 1
    combiner.E_t = np.concatenate((E_t_pos, E_t_neg), axis=0)
    bcc_pred = combiner.combine_classifications(C)
    
    bcc_pred = bcc_pred[1,:,:].reshape((nx,ny)) # only interested in positive "damage class"
    
    results['Train_GP_on_Freq'] = bcc_pred
        
    # EVALUATE ALL RESULTS ---------------------------------------------------------------------------------------------
    evaluator = Evaluator("", "BCCHeatmaps", "Ushahidi_Haiti_Building_Damage")
    
    for method in results:
        print "Results for %s" % method
        pred = results[method]
        testresults = pred.flatten()
        labels = tgrid.flatten()
        acc = np.sum(np.round(testresults)==labels) / float(len(labels))
        print acc  
    
        defaultval = nu0[1] / float(np.sum(nu0))
        labels_nondef = labels[testresults!=defaultval] # labels where we have made a non-default prediction
        testresults_nondef = testresults[testresults!=defaultval] 
        mce = evaluator.eval_crossentropy(testresults_nondef, labels_nondef) * len(labels_nondef)
        mce_def = evaluator.eval_crossentropy(np.array([1-defaultval]), np.array([0]))
        nneg_def = np.sum(labels==0) - np.sum(labels_nondef==0)
        mce += mce_def*nneg_def
        mce_def = evaluator.eval_crossentropy(np.array([defaultval]), np.array([1]))
        npos_def = np.sum(labels==1) - np.sum(labels_nondef==1)    
        mce += mce_def*npos_def
        # Make this the cross entropy per data point
        mce = mce/len(labels)
        print str(mce)
        
        auc,ap = evaluator.eval_auc(testresults,labels)
        print str(auc)
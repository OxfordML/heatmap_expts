'''
Created on 9 May 2015

@author: edwin
'''
import numpy as np
import pandas as pd
import logging
from prediction_tests import Tester
from scipy.stats import gamma

class CicadaDataHandler(object):
    
    def __init__(self, nx= 2000, ny= 2640, datadir= './data', datafile = '/cicada-edwin.csv'):
        # default grid size gives squares of approximately 25x25m
        self.nx = nx
        self.ny = ny
        self.datadir = datadir
        self.datafile = self.datadir + datafile
        self.data = []
        
    def translate_points_to_local(self, latdata, londata):
        logging.debug('Translating original coords to local values.')
            
        latdata = np.float64(latdata)
        londata = np.float64(londata)
        
        normlatdata = (latdata-self.minlat)/(self.maxlat-self.minlat)
        normlondata = (londata-self.minlon)/(self.maxlon-self.minlon)    
         
        #latdata = np.array(normlatdata * self.nx)
        #londata = np.array(normlondata * self.ny)   
        latdata = np.array(np.floor(normlatdata*self.nx), dtype=np.int)
        londata = np.array(np.floor(normlondata*self.ny), dtype=np.int)
            
        return latdata,londata         
    
    def load_data(self):
        #load the data
        if not np.any(self.data):
            self.data = pd.read_csv(self.datafile, delimiter=',', usecols=[3,4,5,6,8,9])
        data = self.data
        
        latdata = data['latitude'].convert_objects(convert_numeric=True)
        londata = data['longitude'].convert_objects(convert_numeric=True)
        
        self.minlat = 50.72
        self.maxlat = 50.99
        self.minlon = -1.88
        self.maxlon = -1.31
        
        valididxs = ~np.isnan(latdata) & (londata > self.minlon) & (londata  < self.maxlon) & (latdata > self.minlat)\
                        & (latdata < self.maxlat)
        
        latdata,londata = self.translate_points_to_local(latdata[valididxs], londata[valididxs])
                
        classids = data['output_id'].convert_objects(convert_numeric=True)[valididxs]
        deviceids = data['device_uuid'][valididxs]
        values = data['value'].convert_objects(convert_numeric=True)[valididxs]
        targetdata = data['true_id'].convert_objects(convert_numeric=True)[valididxs]
        
        C = []
        C_with_gold = []
        agenttypes = []
        targets = []       
        instantiatedagents = []
        
        #original_target_values = np.sort(np.unique(targetdata))
        
        latlon_counter = 0
        for i, deviceid in deviceids.iteritems():
            
            reportvalue = values[i]
            classid = classids[i]
            
            # agent ID needs to be a fusion between the class it is detecting and the device
            classid = classids[i]
            agentIDstr = "%s_%s" % (classid, deviceid)
            
            agenttype = np.int(classid) # use the agenttype to determine which prior is suitable. Will require changin the optimizer.
            if agenttype > 90:
                agenttype = 99
            elif agenttype > 4:
                # ignore other classes
                continue
                        
            if agentIDstr not in instantiatedagents:
                instantiatedagents.append(agentIDstr)
                agenttypes.append(agenttype)

            agentID = np.argwhere(np.in1d(instantiatedagents, agentIDstr))[0, 0]

#             if not deviceid in instantiatedagents:
#                 instantiatedagents.append(deviceid)
            
#             agentID = np.argwhere(np.in1d(instantiatedagents, deviceid))[0, 0]
            
            repx = latdata[latlon_counter]
            repy = londata[latlon_counter]
            latlon_counter += 1
            
            if np.isnan(repx) or np.isnan(repy):
                continue
            
            C.append([agentID, repx, repy, reportvalue])
            
            if not np.isnan(targetdata[i]):
                #targetvalue = np.argwhere(original_target_values==targetdata[i])[0,0]
                targetvalue = int(targetdata[i])
                
                if targetvalue > 90: # group everything that is not in the first four categories together as "none"
                    targetvalue = 99
                elif targetvalue > 4:
                    targetvalue = 5
                
                C_with_gold.append([agentID, repx, repy, reportvalue, targetvalue])                
                targets.append([repx, repy, targetvalue])
                
        self.K = len(instantiatedagents)     
        C = np.array(C)
        print "Number of reports: %i" % (C.shape[0])   
        self.C = C
        self.C_with_gold = np.array(C_with_gold)
        self.targets = np.array(targets)
        self.agenttypes = np.array(agenttypes)
        
if __name__=="__main__":
    print "Loading Cicada Data..."

    loader = CicadaDataHandler()    
    if 'data' in globals():
        loader.data = data
    loader.load_data()
    data = loader.data
    
    methods = ['KDE', 'IBCC', 'HeatmapBCC', 'GP', 'IBCC+GP']#['HeatmapBCC']
    Nreports = loader.C.shape[0]
    
    nscores = 2 
    
    accsums = {}
    totals = {}
    
    class4score = 0
    otherclassscore = 0
    
    # calculate the accuracy of the agents
    for agentID in range(loader.K):
        at = loader.agenttypes[agentID]
        idxs = loader.C_with_gold[:, 0] == agentID
        if np.sum(idxs) == 0:
            continue
        posidxs = (at == loader.C_with_gold[idxs, 4]) 
#         total_correct = np.sum((loader.C_with_gold[idxs, 3]> 0) * posidxs + (loader.C_with_gold[idxs, 3] == 0) * (1 - posidxs))
#         accuracy = total_correct / float(np.sum(idxs))
#         print accuracy

        total_correct = np.sum((loader.C_with_gold[idxs, 3]) * posidxs + (1 - loader.C_with_gold[idxs, 3]) * (1 - posidxs))
        accuracy = total_correct / float(np.sum(idxs))
        #print "accuracy: %f. type: %i" % (accuracy, at)
        
        if at == 5:
            class4score += np.sum(loader.C_with_gold[idxs, 3])
            otherclassscore += np.sum(1 - loader.C_with_gold[idxs, 3])
        else:
            class4score += np.sum(1 - loader.C_with_gold[idxs, 3])
            otherclassscore += np.sum(loader.C_with_gold[idxs, 3])
        
        if at not in accsums:
            accsums[at] = 0
            totals[at] = 0
        accsums[at] += accuracy 
        totals[at] += 1.0
        
    print "class 4 percentage: %f" % (class4score / (class4score + otherclassscore))
        
    for at in range(5):
        print "type = %i, mean acc = %f" % (at, accsums[at]/totals[at])
    
#     Run separately for each class
    for l, c in enumerate([5]):#np.unique(loader.targets[:, 2])):
        for d in range(1):
            print "processing class %i" % c
            
            alpha0 = np.ones((2, nscores, loader.K))
           
            for agentID in range(loader.K):
                if loader.agenttypes[agentID] == c: # make only the classifications for the relevant type have strong diagonals
                    alpha0[0, 0, agentID] = 2
                    alpha0[1, 1, agentID] = 2
                else:
                    alpha0[0, 1, agentID] = 2 # likely to give a higher value if it is not this class
                    alpha0[1, 0, agentID] = 2
                    
            alpha0 *= 1000
                
            nu0 = np.array([10000, 10000]).astype(float)
            z0 = nu0[1] / np.sum(nu0)
            shape_s0 = 0.5
            rate_s0 = 10.0 * 0.5    
            
            ls = [4, 8, 16, 32, 64]
            lpls = gamma.logpdf(ls, 2, scale=22)
        
            outputdir = "./data/cicada/%i_%i" % (c, d)
            tester = Tester(outputdir, methods, Nreports, z0, alpha0, nu0, shape_s0, rate_s0, 
                        ls, optimise=False, verbose=False, lpls=lpls)
            
            Nreps_initial = loader.C.shape[0]#1000
            Nrep_inc = 1000
            
            x_gold = loader.targets[:, 0]
            y_gold = loader.targets[:, 1]
            t_gold = loader.targets[:, 2] == c
            
            C = loader.C.copy()
            shuffle_idxs = np.random.permutation(C.shape[0])
            C = C[shuffle_idxs, :]
            
#             remove the type 4 reports
#            C = C[C[:, 3] != 4, :]
            
            tester.run_tests(C, loader.nx, loader.ny, x_gold, y_gold, t_gold, [], Nreps_initial, Nrep_inc)
            tester.save_separate_results()        
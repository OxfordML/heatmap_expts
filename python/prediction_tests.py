'''
Created on 8 Jan 2015

@author: edwin
'''

import logging
import numpy as np
from heatmapbcc import HeatMapBCC
from ibccperformance import Evaluator
from ibcc import IBCC
from gpgrid import GPGrid
from scipy.sparse import coo_matrix
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

class UshahidiDataHandler(object):
    
    K = 1
    nx = 1000
    ny = 1000
    
    minlat = 18.2#18.0
    maxlat = 18.8#19.4
    minlon = -72.6#-73.1
    maxlon = -72.0#-71.7   
    
    datadir = './data'
    
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
            maintype = 1
            if "1a." in reptypetext: # Trapped people
                agentID = 0
            elif "1b." in reptypetext: 
                agentID = 1
            elif "1c." in reptypetext:
                agentID = 2
            elif "1" in reptypetext or "1d." in reptypetext:
                agentID = 3
            elif "2a." in reptypetext:
                agentID = 4
            elif "2b." in reptypetext:
                agentID = 5
            elif "2c." in reptypetext:
                agentID = 6
            elif "2d." in reptypetext:
                agentID = 7
            elif "2e." in reptypetext:
                agentID = 8
            elif "2f." in reptypetext:
                agentID = 9
            elif "2g." in reptypetext:
                agentID = 10              
            elif "3a." in reptypetext:
                agentID = 11
            elif "3." in reptypetext or "3b." in reptypetext or "3c." in reptypetext or "3d." in reptypetext or \
                "3e." in reptypetext  or "6b." in reptypetext:
                agentID = 12
            elif "4a." in reptypetext:
                agentID = 13
            elif "4." in reptypetext or "4b." in reptypetext:
                agentID = 14
            elif "5a." in reptypetext:
                agentID = 15
            elif "5." in reptypetext or "5c." in reptypetext or "5d." in reptypetext:
                agentID = 16
            elif "6." in reptypetext or "6a." in reptypetext:
                agentID = 17
            elif "7a." in reptypetext:
                agentID = 18     
            elif "7b." in reptypetext:
                agentID = 19
            elif "7c." in reptypetext:
                agentID = 20
            elif "7." in reptypetext or "7d." in reptypetext or "7e." in reptypetext or "7f." in reptypetext:
                agentID = 21
            elif "8a." in reptypetext:
                agentID = 22
            elif "8b." in reptypetext or "6c." in reptypetext or "6d." in reptypetext:
                agentID = 23
            elif "8c." in reptypetext:
                agentID = 24
            elif "8e." in reptypetext or "8." in reptypetext or  "8d." in reptypetext or "8f." in reptypetext:
                agentID = 25                
            else: #we don't care about these categories in the demo anyway, but can add them easily here
                agentID = 26
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
    
    logging.basicConfig(level=logging.DEBUG)

    
    #LOAD THE DATA to run unsupervised learning tests ------------------------------------------------------------------
    #Load up some ground truth
#     goldfile = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target_182_188_726_720_100_100.csv"
#     tgrid = np.genfromtxt(goldfile).astype(int)
    goldfile = "./data/haiti_unosat_target3.npy"
    targets = np.load(goldfile).astype(int)
    
    targetsx = targets[:,0]
    targetsy = targets[:,1]
    labels = targets[:,2]  
    
    nx = 1000
    ny = 1000
    datahandler = UshahidiDataHandler(nx,ny, './data/')
    datahandler.load_ush_data()
    # Need to see how the methods vary with number of messages, i.e. when the messages come in according to the real
    # time line. Can IBCC do better early on?
    C = datahandler.C[1]
    C_all = C # save for later
    
    K = datahandler.K
    # default hyperparameters
    alpha0 = np.array([[1.1, 1.0], [1.0, 1.1]])[:,:,np.newaxis]
    alpha0 = np.tile(alpha0, (1,1,K))
    # set stronger priors for more meaningful categories
    alpha0[:,:,range(0,4)] = np.array([[2.0,1.0],[1.0,2.0]])[:,:,np.newaxis]
    alpha0[:,:,range(15,18)] = np.array([[2.0,1.0],[1.0,2.0]])[:,:,np.newaxis]
    alpha0_all = alpha0
    # set an uninformative prior over the spatial GP
    nu0 = np.array([1.1, 1.0])    
    
#     #subset for testing
#     C = C[1:100,:]
#     alpha0 = alpha0[:, :, np.sort(np.unique(C[:,0]))]
#     K = len(np.unique(C[:,0]))
    
    plt.figure()
    plt.scatter(targetsx, targetsy)
    plt.show()
    
    print "No. report types = %i" % K
    
    results_all = {}
    auc_all = {}
    mce_all = {}
    acc_all = {}
    navailable = C.shape[0]
    nlabels = 100
    while nlabels <= navailable:
        C = C_all[0:nlabels, :]
        agents = np.unique(C[:,0])
        K = len(agents)
        alpha0 = alpha0_all[:,:,agents]
        
        # containers for results
        results = {}
                
        # indicator array to show whether reports are positive or negative
        posreports = (C[:,3]==1).astype(float)
        negreports = (C[:,3]==0).astype(float)
        
        # Report coords for this round
        reportsx = C[:, 1].astype(int)
        reportsy = C[:, 2].astype(int) 
        
        # grid indicating where we made observations
        obs_grid = coo_matrix((np.ones(reportsx.shape[0]), (reportsx, reportsy)), (nx,ny))
        
        # Matrix of counts of reports at each location
        counts_pos = coo_matrix((posreports, (reportsx, reportsy)), (nx,ny))
        counts_neg = coo_matrix((negreports, (reportsx, reportsy)), (nx,ny))               
          
        # Coordinates where we made observations, i.e. without duplicates due to multiple reports at some points 
        obs_coords = np.argwhere(obs_grid.toarray()>0)
        obsx = obs_coords[:, 0]
        obsy = obs_coords[:, 1]          
                
        # KERNEL DENSITY ESTIMATION ---------------------------------------------------------------------------------------
        
        # reuse values for training points given by frequencies
        #kernels and sum. Iterate and try to minimise L2 risk to select kernel bandwidth (mean integrated squared error). 
        # This is given by: expected difference between kernel function evaluated at a point and the true value. Can look at
        # this for the training points only, but this would cause overfitting. Alternatively, can use cross-validation.
        # Matlab kdensity. In python, see:
        # http://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
        #x_idxs, y_idxs = np.mgrid[0:nx, 0:ny]
        grid_idxs = np.vstack((targetsx,targetsy))#(x_idxs.ravel(), y_idxs.ravel()))
        posinputdata = np.vstack((reportsx[posreports>0], reportsy[posreports>0]))
        neginputdata = np.vstack((reportsx[negreports>0], reportsy[negreports>0]))
        logging.info("Running KDE... pos data size: %i, neg data size: %i " % (posinputdata.shape[1], neginputdata.shape[1]) )
        kdepos = gaussian_kde(posinputdata)
        if neginputdata.shape[1] != 0:
            kdeneg = gaussian_kde(neginputdata)
        #else:
            # no_obs_coords = np.argwhere(obs_grid.toarray()==0)
            # neginputdata = np.vstack((no_obs_coords[:,0], no_obs_coords[:,1]))
        grid_lower = np.vstack((targetsx - 0.5, targetsy - 0.5))
        grid_upper = np.vstack((targetsx + 0.5, targetsy + 0.5))
        p_loc_giv_damage = np.zeros(len(targetsx))
        p_loc_giv_nodamage = np.zeros(len(targetsx))
        for i in range(len(targetsx)):
            if i%1000 == 0:
                logging.debug("Processing %i of %i" % (i,len(targetsx)))
            p_loc_giv_damage[i] = kdepos.integrate_box(grid_lower[:, i], grid_upper[:, i])
            if neginputdata.shape[1] != 0:
                p_loc_giv_nodamage[i] = kdeneg.integrate_box(grid_lower[:, i], grid_upper[:, i])
            else:
                p_loc_giv_nodamage[i] = 1.0 / (nx*ny)
        p_damage = nu0[1] / np.sum(nu0)
        p_damage_loc = p_loc_giv_damage * p_damage
        p_nodamage_loc = p_loc_giv_nodamage * (1.0-p_damage)
        p_damage_giv_loc  = p_damage_loc / (p_damage_loc + p_nodamage_loc)
        results['KDE'] = p_damage_giv_loc
        logging.info("KDE complete.")
        
        # TRAIN GP WITHOUT BCC ---------------------------------------------------------------------------------------------
        
        # get values for training points by taking frequencies
        density_estimates = counts_pos / (obs_grid)
        density_estimates = np.array(density_estimates[obsx, obsy]).flatten()
        #run GP
        gpgrid = GPGrid(nx, ny)
        gpgrid.fit([obsx, obsy], density_estimates)
        gp_preds, _ = gpgrid.predict([targetsx, targetsy])
        results['Train_GP_on_Freq'] = gp_preds        
        
        # RUN SEPARATE IBCC AND GP STAGES ----------------------------------------------------------------------------------
        
        # run standard IBCC
        combiner = IBCC(2, 2, alpha0, nu0, K)
        combiner.min_iterations = 5
        combiner.max_iterations = 200
        combiner.conv_threshold = 0.1
        
        #flatten the input data so it can be used with standard IBCC
        linearIdxs = np.ravel_multi_index((reportsx, reportsy), dims=(nx,ny))
        C_flat = C[:,[0,1,3]]
        C_flat[:,1] = linearIdxs
        bcc_pred = combiner.combine_classifications(C_flat)
        bcc_pred = bcc_pred[np.ravel_multi_index((obsx, obsy), dims=(nx,ny)), 1]    
    
        # use IBCC output to train GP
        gpgrid = GPGrid(nx, ny)
        gpgrid.fit([obsx, obsy], bcc_pred)
        gp_preds, _ = gpgrid.predict([targetsx, targetsy])
        
        results['IBCC_then_GP'] = gp_preds        
            
        # RUN HEATMAP BCC --------------------------------------------------------------------------------------------------
        combiner = HeatMapBCC(nx, ny, 2, 2, alpha0, nu0, K, force_update_all_points=True, outputx=targetsx, outputy=targetsy)
        combiner.min_iterations = 5
        combiner.max_iterations = 200
        combiner.conv_threshold = 0.1
    
        # Need to replace with optimised version!
        bcc_pred = combiner.combine_classifications(C)
        bcc_pred = bcc_pred[1,:] # only interested in positive "damage class"
        
        results['heatmapbcc'] = bcc_pred
                
        # EVALUATE ALL RESULTS ---------------------------------------------------------------------------------------------
        evaluator = Evaluator("", "BCCHeatmaps", "Ushahidi_Haiti_Building_Damage")
        
        for method in results:
            print "Results for %s" % method
            pred = results[method]
            testresults = pred.flatten()
            acc = np.sum(np.round(testresults)==labels) / float(len(labels))
            print "accuracy: %.4f" % acc  
            mce = evaluator.eval_crossentropy(testresults, labels) 
            print "cross entropy: %.4f" % mce 
            auc,ap = evaluator.eval_auc(testresults,labels)
            print "AUC: %.4f" % auc
            if method not in auc_all:
                auc_all[method] = [auc]
                acc_all[method] = [acc]
                mce_all[method] = [mce]
            else:
                auc_all[method].append(auc)
                acc_all[method].append(acc)
                mce_all[method].append(mce)
            
        # set up next iteration
        results_all[nlabels] = results
        if nlabels==C_all.shape[0]:
            break
        elif C_all.shape[0]-nlabels<100:
            nlabels = C_all.shape[0] 
        else:
            nlabels += 400
            
    np.save("./data/output/results.npy", results_all)
    np.save("./data/output/acc.npy", acc_all)
    np.save("./data/output/auc.npy", auc_all)
    np.save("./data/output/mce.npy", mce_all)
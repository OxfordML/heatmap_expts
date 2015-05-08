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
from ushahididata import UshahidiDataHandler

if __name__ == '__main__':
    print "Run tests to determine the accuracy of the bccheatmaps method."
    
    logging.basicConfig(level=logging.DEBUG)

    
    #LOAD THE DATA to run unsupervised learning tests ------------------------------------------------------------------
    #Load up some ground truth
#     goldfile = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target_182_188_726_720_100_100.csv"
#     tgrid = np.genfromtxt(goldfile).astype(int)
    goldfile = "./data/haiti_unosat_target3.npy"
    targets = np.load(goldfile).astype(int)
    
    # SUBSET FOR TESTING
#     targets = targets[0:1000,:]
    
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
    nu0 = np.array([1.0, 1.0])    
    
#     #subset for testing
#     C = C[1:100,:]
#     alpha0 = alpha0[:, :, np.sort(np.unique(C[:,0]))]
#     K = len(np.unique(C[:,0]))
    
    print "No. report types = %i" % K
    
    results_all = {}
    auc_all = {}
    mce_all = {}
    acc_all = {}
    # number of available data points
    navailable = C.shape[0]
    # number of labels in first iteration dataset
    nlabels = 250
    # increment the number of labels at each iteration
    stepsize = 250
    while nlabels <= navailable:
        C = C_all[0:nlabels, :]
        agents = np.unique(C[:,0])
        K = len(agents)
        alpha0 = alpha0_all[:,:,agents]
        
        # containers for results
        if not 'results' in globals():
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
        gpgrid = GPGrid(nx, ny, s=4, ls=4, nu0=[200,200])
        gpgrid.optimize([obsx, obsy], density_estimates)
        gp_preds = gpgrid.predict([targetsx, targetsy])
        results['Train_GP_on_Freq'] = gp_preds        
         
        # RUN SEPARATE IBCC AND GP STAGES ----------------------------------------------------------------------------------
           
        # run standard IBCC
        combiner = IBCC(2, 2, alpha0, nu0, K)
        combiner.verbose = False
        combiner.min_iterations = 5
        combiner.max_iterations = 200
        combiner.conv_threshold = 0.1
           
        #flatten the input data so it can be used with standard IBCC
        linearIdxs = np.ravel_multi_index((reportsx, reportsy), dims=(nx,ny))
        C_flat = C[:,[0,1,3]]
        C_flat[:,1] = linearIdxs
        bcc_pred = combiner.combine_classifications(C_flat, optimise_hyperparams=True)
        bcc_pred = bcc_pred[np.ravel_multi_index((obsx, obsy), dims=(nx,ny)), 1]    
       
        # use IBCC output to train GP
        gpgrid = GPGrid(nx, ny, s=4, ls=4, nu0=[200,200])
        gpgrid.optimize([obsx, obsy], bcc_pred)
        gp_preds = gpgrid.predict([targetsx, targetsy])
           
        results['IBCC_then_GP'] = gp_preds        
             
        # RUN HEATMAP BCC --------------------------------------------------------------------------------------------------
        combiner = HeatMapBCC(nx, ny, 2, 2, alpha0, nu0, K, force_update_all_points=True, outputx=targetsx, outputy=targetsy)
        combiner.min_iterations = 5
        combiner.max_iterations = 200
        combiner.conv_threshold = 0.1
      
        # Need to replace with optimised version!
        bcc_pred = combiner.combine_classifications(C, optimise_hyperparams=True)
        bcc_pred = bcc_pred[1,:] # only interested in positive "damage class"
          
        results['heatmapbcc'] = bcc_pred
                
        # EVALUATE ALL RESULTS ---------------------------------------------------------------------------------------------
        evaluator = Evaluator("", "BCCHeatmaps", "Ushahidi_Haiti_Building_Damage")
        
        for method in results:
            print "Results for %s" % method
            pred = results[method]
            testresults = pred.flatten()
            mce = evaluator.eval_crossentropy(testresults, labels)
            print "cross entropy: %.4f" % mce 
            acc = np.sum(np.round(testresults)==labels) / float(len(labels))            
            print "accuracy: %.4f" % acc  
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
            nlabels += stepsize
            
    np.save("./data/output/results.npy", results_all)
    np.save("./data/output/acc.npy", acc_all)
    np.save("./data/output/auc.npy", auc_all)
    np.save("./data/output/mce.npy", mce_all)
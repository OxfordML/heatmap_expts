'''
Created on 8 Jan 2015

@author: edwin


Notes/TO DO:

See if the data points have been extracted correctly. We currently have 294170 test points, which is too many -- let's 
run this one 5 different sub-samples if this is the correct number. Make sure we are using the original house locations,
not the discrete grid values.

'''

import logging
import numpy as np
from heatmapbcc import HeatMapBCC
from ibccperformance import Evaluator
from ibcc import IBCC
from gpgrid import GPGrid
from scipy.sparse import coo_matrix
from scipy.stats import gaussian_kde, mannwhitneyu
from scipy.optimize import fmin_cobyla
from ushahididata import UshahidiDataHandler

if __name__ == '__main__':
    print "Run tests to determine the accuracy of the bccheatmaps method."

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    methods = [
               #'KDE',
               #'GP',
               #'IBCC+GP',
               'HeatmapBCC'
               ]
    
    #LOAD THE DATA to run unsupervised learning tests ------------------------------------------------------------------
    #Load up some ground truth
#     goldfile = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target_182_188_726_720_100_100.csv"
#     tgrid = np.genfromtxt(goldfile).astype(int)
    goldfile = "./data/haiti_unosat_target3.npy"
    targets = np.load(goldfile).astype(int)
    
    # SUBSET FOR TESTING - whole dataset is too long (~300,000 data points)
    testidxs = np.random.randint(targets.shape[0], size=10000)
    targets = targets[testidxs,:]
    
    targetsx = targets[:,0]
    targetsy = targets[:,1]
    labels = targets[:,2]  
    
    nx = 1000
    ny = 1000
    datahandler = UshahidiDataHandler(nx,ny, './data/')
    datahandler.discrete = False # do not snap-to-grid
    datahandler.load_data()
    # Need to see how the methods vary with number of messages, i.e. when the messages come in according to the real
    # time line. Can IBCC do better early on?
    C = datahandler.C[1]
    C_all = C # save for later    
    
    K = datahandler.K
    # default hyper-parameters
    # default hyper-parameters
    alpha0 = np.array([[2.0, 1.0], [1.0, 2.0]])[:,:,np.newaxis]
    alpha0 = np.tile(alpha0, (1,1,3))
    # set stronger priors for more meaningful categories
    alpha0[:,:,1] = np.array([[5.0,1.0],[1.0,5.0]]) # confident agents
    alpha0[:,:,2] = np.array([[1.0,1.0],[1.0,1.0]]) # agents with no prior knowledge of correlations
    clusteridxs_all = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2, 0, 0, 2, 2]) 
    alpha0_all = alpha0
    # set an uninformative prior over the spatial GP
    nu0 = np.array([5.0, 1.0])    
    z0 = nu0[1] / np.sum(nu0)
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
    nlabels = 2000
    # increment the number of labels at each iteration
    stepsize = 100
    
    #GPGRID OBJECT
    gpgrid = GPGrid(nx, ny, z0=z0, shape_ls=10, rate_ls=10.0/100)
    gpgrid.verbose = False
    
    #HEATMAPBCC OBJECT
    heatmapcombiner = HeatMapBCC(nx, ny, 2, 2, alpha0, K, z0=z0, shape_ls=10, rate_ls=10.0/100,
                              force_update_all_points=True, outputx=targetsx, outputy=targetsy)
    heatmapcombiner.min_iterations = 5
    heatmapcombiner.max_iterations = 200
    heatmapcombiner.conv_threshold = 0.1
    heatmapcombiner.verbose = False

    # Initialize the optimal grid size for the separate IBCC method.
    opt_nx = float(nx)
    opt_ny = float(ny)

    while nlabels <= navailable:
        C = C_all[0:nlabels, :]
        agents = np.unique(C[:,0])
        K = int(np.max(agents)+1)
        clusteridxs = clusteridxs_all[0:K]
        alpha0 = alpha0_all
        
        # containers for results
        if not 'results' in globals():
            results = {}
                
        # indicator array to show whether reports are positive or negative
        posreports = (C[:, 3] == 1).astype(float)
        negreports = (C[:, 3] == 0).astype(float)
        
        # Report coords for this round
        reportsx = C[:, 1]
        reportsy = C[:, 2]                 
                 
        # KERNEL DENSITY ESTIMATION ---------------------------------------------------------------------------------------
        if 'KDE' in methods:
            # Method used here performs automatic bandwidth determination - see help docs
            posinputdata = np.vstack((reportsx[posreports>0], reportsy[posreports>0]))
            neginputdata = np.vstack((reportsx[negreports>0], reportsy[negreports>0]))
            logging.info("Running KDE... pos data size: %i, neg data size: %i " % (posinputdata.shape[1], neginputdata.shape[1]) )
            kdepos = gaussian_kde(posinputdata)
            if neginputdata.shape[1] != 0:
                kdeneg = gaussian_kde(neginputdata)
            #else: # Treat the points with no observation as negatives
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
            p_damage_loc = p_loc_giv_damage * z0
            p_nodamage_loc = p_loc_giv_nodamage * (1.0-z0)
            p_damage_giv_loc  = p_damage_loc / (p_damage_loc + p_nodamage_loc)
            results['KDE'] = p_damage_giv_loc
            logging.info("KDE complete.")
           
        # TRAIN GP WITHOUT BCC ---------------------------------------------------------------------------------------------
        if 'GP' in methods:
            logging.info("Using a density GP without BCC...")
            # Get values for training points by taking frequencies -- only one report at each location, so give 1 for
            # positive reports, 0 otherwise
            gpgrid.optimize([reportsx, reportsy],
                            np.concatenate((posreports[:,np.newaxis]*0.8, (posreports+negreports)[:,np.newaxis]), axis=1))
            gp_preds = gpgrid.predict([targetsx, targetsy])
            results['Train_GP_on_Freq'] = gp_preds
           
        # RUN SEPARATE IBCC AND GP STAGES ----------------------------------------------------------------------------------
        if 'IBCC+GP' in methods:
            # Should this be dropped from the experiments, as it won't work without gridding the points? --> then run into
            # question of grid size etc. Choose grid so that squares have an average of 3 reports?
            logging.info("Running separate IBCC and GP...")

            # run standard IBCC
            ibcc_combiner = IBCC(2, 2, alpha0, nu0, K)
            ibcc_combiner.clusteridxs_alpha0 = clusteridxs
            ibcc_combiner.verbose = False
            ibcc_combiner.min_iterations = 5
            ibcc_combiner.max_iterations = 200
            ibcc_combiner.conv_threshold = 0.1

            #initialise a GP
            initialguess = [opt_nx, opt_ny]
            # noinspection PyTypeChecker
            constraints = [lambda hp: 1 if np.all(np.asarray(hp[0:2]) >= 1) else -1, lambda hp: 1 if not np.isnan(hp[0]) and hp[0] - int(hp[0]) == 0 else -1,
                           lambda hp: 1 if not np.isnan(hp[1]) and hp[1] - int(hp[1]) == 0 else -1]
            gpgrid2 = GPGrid(opt_nx, opt_ny, z0=z0, shape_ls=10, rate_ls=10.0/100)
            # noinspection PyTypeChecker
            def train_gp_on_ibcc_output(hyperparams):
                logging.debug("fmin gridx and gridy values: %f, %f" % (hyperparams[0], hyperparams[1]))
                opt_nx = np.round(hyperparams[0])
                opt_ny = np.round(hyperparams[1])
                if opt_nx < 0 or opt_ny < 0 or np.isnan(opt_nx) or np.isnan(opt_ny):
                    return np.inf
                gpgrid2.nx = opt_nx
                gpgrid2.ny = opt_ny

                # grid indicating where we made observations
                reportsx_grid = (reportsx * opt_nx/nx).astype(int)
                reportsy_grid = (reportsy * opt_ny/ny).astype(int)

                obs_grid = coo_matrix((np.ones(reportsx.shape[0]), (reportsx_grid, reportsy_grid)), (opt_nx, opt_ny))
                # Coordinates where we made observations, i.e. without duplicates due to multiple reports at some points
                obs_coords = np.argwhere(obs_grid.toarray()>0)
                obsx = obs_coords[:, 0]
                obsy = obs_coords[:, 1]

                #flatten the input data so it can be used with standard IBCC
                linearIdxs = np.ravel_multi_index((reportsx_grid, reportsy_grid), dims=(opt_nx, opt_ny))
                C_flat = C[:,[0,1,3]]
                C_flat[:,1] = linearIdxs
                bcc_pred = ibcc_combiner.combine_classifications(C_flat, optimise_hyperparams=False)
                bcc_pred = bcc_pred[np.ravel_multi_index((obsx, obsy), dims=(opt_nx, opt_ny)), 1]

                # use IBCC output to train GP
                _, ls = gpgrid2.optimize([obsx, obsy], bcc_pred)
                logging.debug("fmin param value for lengthscale: %f" % ls[0])
                nlml = gpgrid2.neg_marginal_likelihood(np.log(ls))
                return nlml

            opt_hyperparams = fmin_cobyla(train_gp_on_ibcc_output, initialguess, constraints, rhobeg=500, rhoend=100)
            # use optimized grid size to make predictions
            gp_preds = gpgrid2.predict([targetsx, targetsy])

            results['IBCC_then_GP'] = gp_preds

        # RUN HEAT MAP BCC --------------------------------------------------------------------------------------------------
        if 'HeatmapBCC' in methods:
            logging.info("Running HeatmapBCC...")
            # to do:
            # make sure optimise works
            # make sure the optimal hyper-parameters are passed to the next iteration
            heatmapcombiner.clusteridxs_alpha0 = clusteridxs
            bcc_pred = heatmapcombiner.combine_classifications(C, optimise_hyperparams=True)
            bcc_pred = bcc_pred[1, :] # only interested in positive "damage class"
            results['heatmapbcc'] = bcc_pred
                
        # EVALUATE ALL RESULTS ---------------------------------------------------------------------------------------------
        evaluator = Evaluator("", "BCCHeatmaps", "Ushahidi_Haiti_Building_Damage")
        
        for method in results:
            print 'Results for %s with %i labels' % (method, nlabels)
            pred = results[method]
            testresults = pred.flatten()
            mce = evaluator.eval_crossentropy(testresults, labels)
            print "Cross entropy (individual data points): %.4f" % mce
            rmse = np.sqrt( np.sum((testresults-labels)**2) / float(len(labels)) )
            print "RMSE (individual data points): %.4f" % rmse
            #acc = np.sum(np.round(testresults)==labels) / float(len(labels))
            #print "accuracy: %.4f" % acc
            auc,ap = evaluator.eval_auc(testresults,labels) 
            print "AUC (individual data points): %.4f" % auc

            # assume gold density and est density have 1 row for each class
            est_density = testresults
            if est_density.ndim == 1 or est_density.shape[0] == 1:
                est_density = est_density.reshape(est_density.size, 1)
                est_density = np.concatenate((est_density, 1 - est_density), axis=1)

            mced = - np.sum(gold_density * np.log(est_density)) / float(gold_density.shape[1])
            print "Cross entropy (density estimation): %.4f" % mced

            rmsed = np.sqrt( np.sum((est_density-gold_density)**2) / float(gold_density.shape[1]) )
            print "RMSE (density estimation): %.4f" % rmsed

            u = np.zeros(est_density.shape[0])
            for j in range(est_density.shape[0]):
                u[j], _ = mannwhitneyu(est_density, gold_density)
            print "U-statistic (density estimation): %.4f" % u

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
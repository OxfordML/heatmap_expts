'''
Created on 8 Jan 2015

@author: edwin


Notes/TO DO:

See if the data points have been extracted correctly. We currently have 294170 test points, which is too many -- let's 
run this one 5 different sub-samples if this is the correct number. Make sure we are using the original house locations,
not the discrete grid values.

'''
import os
import numpy as np
import logging
from heatmapbcc import HeatMapBCC
from ibccperformance import Evaluator
from ibcc import IBCC
from gpgrid import GPGrid
from scipy.sparse import coo_matrix
from scipy.stats import gaussian_kde, kendalltau

class Tester(object):
    
    results_all = {}
    densityresults_all = {}
    auc_all = {}
    mce_all = {}
    rmse_all = {}
    
    auc_all_var = {}
    mce_all_var = {}
    rmse_all_var = {}
    
    tau_all = {}
    mced_all = {}
    rmsed_all = {}
    
    gpgrid = None
    heatmapcombiner = None
    gpgrid2 = None
    ibcc_combiner = None
    
    outputdir = None
    
    def __init__(self, outputdir, methods, Nlabels_all, z0, alpha0_all, nu0, clusteridxs_all=None):
        self.outputdir = outputdir
        self.methods = methods
        self.Nlabels_all = Nlabels_all
        self.z0 = z0
        self.alpha0_all = alpha0_all
        self.nu0 = nu0
        if not np.any(clusteridxs_all):
            clusteridxs_all = np.zeros(self.alpha0_all.shape[2], dtype=int)
        self.clusteridxs_all = clusteridxs_all
    
    def run_tests(self, C_all, nx, ny, targetsx, targetsy, gold_labels, gold_density, Nlabels, Nrep_inc):

        C = C_all
    
        # Initialize the optimal grid size for the separate IBCC method.
        opt_nx = np.ceil(float(nx))
        opt_ny = np.ceil(float(ny))
    
        while Nlabels <= self.Nlabels_all:
            
            logging.info("Running methods with %d labels" % Nlabels)
            
            C = C_all[0:Nlabels, :]
            agents = np.unique(C[:,0])
            K = int(np.max(agents)+1)
            clusteridxs = self.clusteridxs_all[0:K]
            alpha0 = self.alpha0_all
    
            results = {}
            densityresults = {}
    
            # indicator array to show whether reports are positive or negative
            posreports = (C[:, 3] == 1).astype(float)
            negreports = (C[:, 3] == 0).astype(float)
    
            # Report coords for this round
            reportsx = C[:, 1]
            reportsy = C[:, 2]
    
            # KERNEL DENSITY ESTIMATION ---------------------------------------------------------------------------------------
            if 'KDE' in self.methods:
                results['KDE'] = {}
                densityresults['KDE'] = {}
                # Method used here performs automatic bandwidth determination - see help docs
                posinputdata  = np.vstack((reportsx[posreports>0], reportsy[posreports>0]))
                neginputdata = np.vstack((reportsx[negreports>0], reportsy[negreports>0]))
                logging.info("Running KDE... pos data size: %i, neg data size: %i " % (posinputdata.shape[1], neginputdata.shape[1]) )
                if posinputdata.shape[1] != 0:
                    kdepos = gaussian_kde(posinputdata, 'scott')
                if neginputdata.shape[1] != 0:
                    kdeneg = gaussian_kde(neginputdata, 'scott')
                #else: # Treat the points with no observation as negatives
                    # no_obs_coords = np.argwhere(obs_grid.toarray()==0)
                    # neginputdata = np.vstack((no_obs_coords[:,0], no_obs_coords[:,1]))
    
                def kde_prediction(x, y):
                    if len(np.unique(y)) > 1:
                        grid_lower = np.vstack((x - 0.5, y - 0.5))
                        grid_upper = np.vstack((x + 0.5, y + 0.5))
                    else:
                        grid_lower = np.vstack((x-0.5, np.zeros(len(y))))
                        grid_upper = np.vstack((x+0.5, np.ones(len(y))))
                        
                    p_loc_giv_damage = np.zeros(len(x))
                    p_loc_giv_nodamage = np.zeros(len(x))
    
                    for i in range(len(x)):
                        if i%1000 == 0:
                            logging.debug("Processing %i of %i" % (i,len(x)))
                            
                        if posinputdata.shape[1] != 0:
                            p_loc_giv_damage[i] = kdepos.integrate_box(grid_lower[:, i], grid_upper[:, i])
                        else:
                            p_loc_giv_damage[i] = 1.0 / (nx * ny)
                            
                        if neginputdata.shape[1] != 0:
                            p_loc_giv_nodamage[i] = kdeneg.integrate_box(grid_lower[:, i], grid_upper[:, i])
                        else:
                            p_loc_giv_nodamage[i] = 1.0 / (nx*ny)
    
                    p_damage_loc = p_loc_giv_damage * self.z0
                    p_nodamage_loc = p_loc_giv_nodamage * (1.0 - self.z0)
                    p_damage_giv_loc  = p_damage_loc / (p_damage_loc + p_nodamage_loc)
                    return p_damage_giv_loc
    
                # Repeat for each of the test datasets
                results['KDE'] = kde_prediction(targetsx, targetsy)
                densityresults['KDE'] = results['KDE']
                #results['KDE']['grid'] = kde_prediction(gridoutputx, gridoutputy)
    
                logging.info("KDE complete.")
    
            # TRAIN GP WITHOUT BCC ---------------------------------------------------------------------------------------------
            if 'GP' in self.methods:      
                
                logging.info("Using a density GP without BCC...")
                # Get values for training points by taking frequencies -- only one report at each location, so give 1 for
                # positive reports, 0 otherwise
                ls_initial = [2, 10, 50, 100]
                nlml = np.inf
                gpgrid_opt = None
                chosen_i = 0
                for i in range(len(ls_initial)):
                    rate_ls = 2.0 / ls_initial[i]
                    self.gpgrid = GPGrid(nx, ny, z0=self.z0, shape_ls=2.0, rate_ls=rate_ls)
                    self.gpgrid.verbose = False    
                    self.gpgrid.p_rep = 0.9
                    self.gpgrid.optimize([reportsx, reportsy], 
                        np.concatenate((posreports[:,np.newaxis], (posreports+negreports)[:,np.newaxis]), axis=1), maxfun=100)
                    if self.gpgrid.nlml < nlml:
                        nlml = self.gpgrid.nlml
                        gpgrid_opt = self.gpgrid
                        chosen_i = i
                logging.info("Chosen ls_initial=%.2f" % ls_initial[chosen_i])
                ls_initial = ls_initial[chosen_i]
                self.gpgrid = gpgrid_opt
    
                gp_preds, _ = self.gpgrid.predict([targetsx, targetsy])
                results['Train_GP_on_Freq'] = gp_preds
                densityresults['Train_GP_on_Freq'] = gp_preds
                #results['Train_GP_on_Freq']['grid'] = self.gpgrid.predict([gridoutputx, gridoutputy]).reshape(nx, ny)
            else:
                ls_initial = nx
                
            # default hyperparameter initialisation points for all the GPs used below
            shape_ls = 2.0
            rate_ls = shape_ls / ls_initial
    
            # RUN SEPARATE IBCC AND GP STAGES ----------------------------------------------------------------------------------
            self.gpgrid2 = GPGrid(opt_nx, opt_ny, z0=self.z0, shape_ls=shape_ls, rate_ls=rate_ls)
            self.gpgrid2.verbose = False # use this verbose flag for this whole method
            self.ibcc_combiner = IBCC(2, 2, alpha0, self.nu0, K)
            if 'IBCC+GP' in self.methods:
                # Should this be dropped from the experiments, as it won't work without gridding the points? --> then run into
                # question of grid size etc. Choose grid so that squares have an average of 3 reports?
                logging.info("Running separate IBCC and GP...")
    
                # run standard IBCC
                self.ibcc_combiner.clusteridxs_alpha0 = clusteridxs
                self.ibcc_combiner.verbose = self.gpgrid2.verbose
                self.ibcc_combiner.min_iterations = 5
                self.ibcc_combiner.max_iterations = 200
                self.ibcc_combiner.conv_threshold = 0.1
    
                def train_gp_on_ibcc_output(opt_nx, opt_ny):
                    #opt_nx = np.ceil(np.exp(hyperparams[0]))
                    #opt_ny = np.ceil(np.exp(hyperparams[1]))
                    logging.debug("fmin gridx and gridy values: %f, %f" % (opt_nx, opt_ny))
                    if opt_nx <= 0 or opt_ny <= 0 or np.isnan(opt_nx) or np.isnan(opt_ny):
                        return np.inf
                    self.gpgrid2.nx = opt_nx
                    self.gpgrid2.ny = opt_ny
    
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
                    bcc_pred = self.ibcc_combiner.combine_classifications(C_flat, optimise_hyperparams=False)
                    bcc_pred = bcc_pred[np.ravel_multi_index((obsx, obsy), dims=(opt_nx, opt_ny)), 1]
    
                    # use IBCC output to train GP
                    pT, _ = self.gpgrid2.optimize([obsx, obsy], bcc_pred, maxfun=100) # fit([obsx, obsy], bcc_pred)
                    ls = self.gpgrid2.ls
                    if self.gpgrid2.verbose:
                        logging.debug("fmin param value for lengthscale: %f, %f" % (ls[0], ls[1]))
                    
                    self.ibcc_combiner.lnjoint(alldata=True)
                    lnpCT = np.sum(pT * self.ibcc_combiner.lnpCT)            
                    lnpPi = self.ibcc_combiner.post_lnpi()
                    lnpKappa = self.ibcc_combiner.post_lnkappa()
                    EEnergy = lnpCT + lnpPi + lnpKappa
                    
                    # Entropy of the variational distribution
                    lnqT = np.sum(pT[pT != 0] * np.log(pT[pT != 0])) #self.ibcc_combiner.q_ln_t()
                    lnqPi = self.ibcc_combiner.q_lnPi()
                    lnqKappa = self.ibcc_combiner.q_lnkappa()
                    H = lnqT + lnqPi + lnqKappa
                    if self.gpgrid2.verbose:
                        logging.debug('EEnergy %.3f, H %.3f, lnpCT %.3f, lnqT %.3f, lnpKappa %.3f, lnqKappa %.3f, lnpPi %.3f, lnqPi %.3f' % \
                          (EEnergy, H, lnpCT, lnqT, lnpKappa, lnqKappa, lnpPi, lnqPi))                              
                    # Lower Bound
                    nlml = - EEnergy + H                
                    
                    if self.gpgrid2.verbose:
                        logging.debug("NLML: " + str(nlml))
                    return nlml
                
                nunique_x = np.unique(reportsx).shape[0]
                nunique_y = np.unique(reportsy).shape[0]
                # try different levels of separation with on average 3 data points per grid square, 5 per grid square and 10 per grid square.
                topx = 0
                topy = 0
                lowest_nlml = np.inf
                for grouping in np.arange(1, 11, dtype=float) * 2:
                    gridx = int(np.ceil(nunique_x / grouping) )
                    gridy = int(np.ceil(nunique_y / grouping) )
                    nlml = train_gp_on_ibcc_output(gridx, gridy)#initialguess, maxfun=20, full_output=False, xtol=10, ftol=1)
                    logging.debug("NLML = %.2f, lowest so far = %.2f" % (nlml, lowest_nlml))
                    if nlml < lowest_nlml:
                        topx = gridx
                        topy = gridy
                        lowest_nlml = nlml
                        conv_counter = 0
                logging.info("Best grid size found so far: %i %i" % (topx, topy))
                train_gp_on_ibcc_output(topx, topy)
                #fmin_cobyla(train_gp_on_ibcc_output, initialguess, constraints, rhobeg=500, rhoend=100)
                # use optimized grid size to make predictions
                targetsx_grid = (targetsx * topx/nx).astype(int)
                targetsy_grid = (targetsy * topy/ny).astype(int)
                gp_preds, _ = self.gpgrid2.predict([targetsx_grid, targetsy_grid])
                results['IBCC_then_GP'] = gp_preds
                densityresults['IBCC_then_GP'] = gp_preds
                #gp_preds = self.gpgrid2.predict([gridoutputx, gridoutputy])
                #results['IBCC_then_GP']['grid'] = gp_preds.reshape(nx, ny)
    
            # RUN HEAT MAP BCC ---------------------------------------------------------------------------------------------        
            if 'HeatmapBCC' in self.methods:
                #HEATMAPBCC OBJECT
                self.heatmapcombiner = HeatMapBCC(nx, ny, 2, 2, alpha0, K, z0=self.z0, shape_ls=shape_ls, rate_ls=rate_ls, force_update_all_points=True)
                self.heatmapcombiner.min_iterations = 4
                self.heatmapcombiner.max_iterations = 200
                self.heatmapcombiner.verbose = False
                self.heatmapcombiner.uselowerbound = True
                if self.heatmapcombiner.uselowerbound:
                    self.heatmapcombiner.conv_threshold = 1
                else:
                    self.heatmapcombiner.conv_threshold = 0.01            
                
                logging.info("Running HeatmapBCC...")
                # to do:
                # make sure optimise works
                # make sure the optimal hyper-parameters are passed to the next iteration
                self.heatmapcombiner.clusteridxs_alpha0 = clusteridxs
                self.heatmapcombiner.combine_classifications(C, optimise_hyperparams=True)
    
                bcc_pred, bcc_density, _ = self.heatmapcombiner.predict(targetsx, targetsy)
                results['heatmapbcc'] = bcc_pred[1, :] # only interested in positive "damage class"
                densityresults['heatmapbcc'] = bcc_density[1, :]
    
                #_, bcc_pred = self.heatmapcombiner.predict_grid() # take the second argument to get the density rather than state at observed points
                #results['heatmapbcc']['grid'] = bcc_pred[1, :]
                
            # EVALUATE ALL RESULTS -----------------------------------------------------------------------------------------
            evaluator = Evaluator("", "BCCHeatmaps", "Ushahidi_Haiti_Building_Damage")
    
            gold_density = gold_density.flatten()
    
            for method in results:
                print ''
                print 'Results for %s with %i labels' % (method, Nlabels)
                pred = results[method]
                est_density = densityresults[method]
    
                mce = []
                rmse = []
                auc = []
                best_thresholds = []
                mced = []
                rmsed = []
                tau = []
                testresults = pred.flatten()
                testresults[testresults==0] = 0.000001 # use a very small value to avoid log errors with cross entropy
                testresults[testresults==1] = 0.999999

                mce_ts = - np.sum(gold_labels * np.log(testresults)) - np.sum((1-gold_labels) * np.log(1 - testresults))
                mce_ts = mce_ts / float(len(gold_labels))
                mce.append( mce_ts )
                
                rmse.append( np.sqrt( np.sum((testresults - gold_labels)**2) / float(len(gold_labels)) ) )

                auc_by_class, _, best_thresholds_ts = evaluator.eval_auc(testresults, gold_labels)
                auc.append( np.sum(np.bincount(gold_labels) * auc_by_class) / len(gold_labels) )
                best_thresholds.append( np.sum(np.bincount(gold_labels) * best_thresholds_ts) / len(gold_labels))
    
                mce_mean = np.sum(mce) / float(len(gold_labels))
                rmse_mean = np.sum(rmse) / float(len(gold_labels))
                auc_mean = np.sum(auc) / float(len(gold_labels))
    
                mce_var = np.sum((mce - mce_mean)**2) / float(len(gold_labels))
                rmse_var = np.sum((rmse - rmse_mean)**2) / float(len(gold_labels))
                auc_var = np.sum((auc - auc_mean)**2) / float(len(gold_labels))
    
                print "Cross entropy (individual data points): %.4f with SD %.4f" % (mce_mean, mce_var**0.5)
                print "RMSE (individual data points): %.4f with SD %.4f" % (rmse_mean, rmse_var**0.5)
                print "AUC (individual data points): %.4f with SD %.4f; best threshold %.2f" % (auc_mean, auc_var**0.5, np.sum(best_thresholds) / float(len(gold_labels)) )
    
                # assume gold density and est density have 1 row for each class
                est_density = est_density.flatten()
                est_density[est_density==0] = 0.0000001
                est_density[est_density==1] = 0.9999999
    
                mced = - np.sum(gold_density * np.log(est_density)) - np.sum((1-gold_density) * np.log(1-est_density))
                mced = mced / float(len(gold_density))
                print "Cross entropy (density estimation): %.4f" % mced
    
                rmsed = np.sqrt( np.sum((est_density - gold_density)**2) / float(len(gold_density)) )
                print "RMSE (density estimation): %.4f" % rmsed
    
                tau, _ = kendalltau(est_density, gold_density)
                if np.isnan(tau):
                    print "Kendall's Tau --> NaNs are mapped to zero for plotting"
                    tau = 0
                print "Kendall's Tau (density estimation): %.4f " % tau
    
                if method not in self.auc_all:
                    self.auc_all[method] = [auc_mean]
                    self.rmse_all[method] = [rmse_mean]
                    self.mce_all[method] = [mce_mean]
    
                    self.auc_all_var[method] = [auc_var]
                    self.rmse_all_var[method] = [rmse_var]
                    self.mce_all_var[method] = [mce_var]
    
                    self.tau_all[method] = [tau]
                    self.rmsed_all[method] = [rmsed]
                    self.mced_all[method] = [mced]
                else:
                    self.auc_all[method].append(auc_mean)
                    self.rmse_all[method].append(rmse_mean)
                    self.mce_all[method].append(mce_mean)
    
                    self.auc_all_var[method].append(auc_var)
                    self.rmse_all_var[method].append(rmse_var)
                    self.mce_all_var[method].append(mce_var)
    
                    self.tau_all[method].append(tau)
                    self.rmsed_all[method].append(rmsed)
                    self.mced_all[method].append(mced)
    
            # set up next iteration
            self.results_all[Nlabels] = results
            self.densityresults_all[Nlabels] = densityresults
            if Nlabels==C_all.shape[0]:
                break
            elif C_all.shape[0]-Nlabels<100:
                Nlabels = C_all.shape[0]
            else:
                Nlabels += Nrep_inc
    
    
    def save_separate_results(self):
        if not os.path.exists(self.outputdir):
            os.mkdir(self.outputdir)
        np.save(self.outputdir + "results.npy", self.results_all)
        np.save(self.outputdir + "density_results.npy", self.results_all)
        np.save(self.outputdir + "rmse.npy", self.rmse_all)
        np.save(self.outputdir + "auc.npy", self.auc_all)
        np.save(self.outputdir + "mce.npy", self.mce_all)
        np.save(self.outputdir + "rmse_var.npy", self.rmse_all_var)
        np.save(self.outputdir + "auc_var.npy", self.auc_all_var)
        np.save(self.outputdir + "mce_var.npy", self.mce_all_var)
        np.save(self.outputdir + "rmsed.npy", self.rmsed_all)
        np.save(self.outputdir + "tau.npy", self.tau_all)
        np.save(self.outputdir + "mced.npy", self.mced_all)
        
    def save_self(self):
        np.save(self.outputdir + "tester.npy", self)
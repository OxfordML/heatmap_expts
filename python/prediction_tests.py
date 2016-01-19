'''
Created on 8 Jan 2015

@author: edwin


Notes/TO DO:

See if the data points have been extracted correctly. We currently have 294170 test points, which is too many -- let's 
run this one 5 different sub-samples if this is the correct number. Make sure we are using the original house locations,
not the discrete grid values.

The simulations will use a fixed length scale to compare models. This will isolate the task of searching for 
length-scales (i.e. hyperparameter optimisation) from the model comparison and mean that the results are not 
dependent on our choice of method for optimising hyperparameters. 

We may also need to constrain length-scale with the real data too because we can use background knowledge. Also,
we do not want one method to get lucky with the optimisation process, while other methods suffer -- so perhaps best to 
test all with same length scale. We can also look at whether the choice of length scale with a subset of data is a 
problem by comparing the model with length scale optimised on a subset with length scale optimised on the whole dataset.

In practice, length scale can be determined in an informative way from background knowledge.
A second experiment might be to take a few simulations with unreliable data, or use real data, and see how easy it is
to learn the length scale with ML or MAP. I.e. do we need a better method or informative priors to learn length scale
using our model and unreliable data, or to replace VB? Do other models avoid this problem? Or is it that the length 
scale only becomes a problem when we have only got noisy reports and very little signal? An alternative to ML might be 
to use a set of equally-spaced values, then fine-tune the best; we would need an experiment to look at the sensitivity 
of results to the length scale to determine the spacing of possible values, and whether any tuning is really necessary.

'''
import os
import numpy as np
import logging
from heatmapbcc import HeatMapBCC
from ibccperformance import Evaluator
from ibcc import IBCC
from gpgrid import GPGrid
from scipy.sparse import coo_matrix
from scipy.stats import gaussian_kde, kendalltau, multivariate_normal as mvn, beta

class Tester(object):
        
    def __init__(self, outputdir, methods, Nlabels_all, z0, alpha0_all, nu0, shape_s0, rate_s0, ls_initial=0, 
                 optimise=True, clusteridxs_all=None, verbose=False):
        
        # Controls whether we optimise hyper-parameters, including length scale
        self.optimise = optimise
        
        # Set this to zero to use the optimal value from the standard GP
        self.ls_initial = ls_initial
        
        self.shape_s0 = shape_s0
        self.rate_s0 = rate_s0
        
        self.outputdir = outputdir
        self.methods = methods
        self.Nlabels_all = Nlabels_all
        self.z0 = z0
        self.alpha0_all = alpha0_all
        self.nu0 = nu0
        if not np.any(clusteridxs_all):
            clusteridxs_all = np.zeros(self.alpha0_all.shape[2], dtype=int)
        self.clusteridxs_all = clusteridxs_all

        self.results_all = {}
        self.densityresults_all = {}
        self.densityVar_all = {}
        self.auc_all = {}
        self.mce_all = {}
        self.rmse_all = {}

        self.tau_all = {}
        self.mced_all = {}
        self.rmsed_all = {}
        
        self.gpgrid = None
        self.heatmapcombiner = None
        self.gpgrid2 = None
        self.ibcc_combiner = None
        
        self.verbose = verbose
    
    def run_tests(self, C_all, nx, ny, targetsx, targetsy, building_density, gold_density, Nlabels, Nrep_inc):

        C = C_all
    
        while Nlabels <= self.Nlabels_all:
            
            logging.info("Running methods with %d labels" % Nlabels)
            
            C = C_all[0:Nlabels, :]
            agents = np.unique(C[:,0])
            K = int(np.max(agents)+1)
            clusteridxs = self.clusteridxs_all[0:K]
            alpha0 = self.alpha0_all
    
            results = {}
            densityresults = {}
            density_var = {}
    
            # indicator array to show whether reports are positive or negative
            posreports = (C[:, 3] == 1).astype(float)
            negreports = (C[:, 3] == 0).astype(float)
    
            # Report coords for this round
            reportsx = C[:, 1]
            reportsy = C[:, 2]
    
            # KERNEL DENSITY ESTIMATION ---------------------------------------------------------------------------------------
            if 'KDE' in self.methods:
                # Method used here performs automatic bandwidth determination - see help docs
                posinputdata  = np.vstack((reportsx[posreports>0], reportsy[posreports>0]))
                neginputdata = np.vstack((reportsx[negreports>0], reportsy[negreports>0]))
                logging.info("Running KDE... pos data size: %i, neg data size: %i " % (posinputdata.shape[1], neginputdata.shape[1]) )
                if posinputdata.shape[1] != 0:
                    if self.optimise:
                        kdepos = gaussian_kde(posinputdata, 'scott')
                    else:
                        kdepos = gaussian_kde(posinputdata, self.ls_initial)
                if neginputdata.shape[1] != 0:
                    if self.optimise:
                        kdeneg = gaussian_kde(neginputdata, 'scott')
                    else:
                        kdeneg = gaussian_kde(neginputdata, self.ls_initial)
                    
                #else: # Treat the points with no observation as negatives
                    # no_obs_coords = np.argwhere(obs_grid.toarray()==0)
                    # neginputdata = np.vstack((no_obs_coords[:,0], no_obs_coords[:,1]))
    
                def logit(a):
                    a[a==1] = 1 - 1e-6
                    a[a==0] = 1e-6
                    return np.log(a / (1-a))
    
                def kde_prediction(targets):                       
                    if posinputdata.shape[1] != 0:
                        logp_loc_giv_damage = kdepos.logpdf(targets)
                    else:
                        norma = np.array([nx, ny], dtype=float)[np.newaxis, :]
                        logp_loc_giv_damage = mvn.logpdf(logit(targets.T / norma), mean=norma.flatten() / 2.0, cov=10000)
                            
                    if neginputdata.shape[1] != 0:
                        logp_loc_giv_nodamage = kdeneg.logpdf(targets)#integrate_box(grid_lower[:, i], grid_upper[:, i])
                    else:
                        norma = np.array([nx, ny], dtype=float)[np.newaxis, :]
                        logp_loc_giv_nodamage = mvn.logpdf(logit(targets.T / norma), mean=norma.flatten() / 2.0, cov=10000)
    
                    p_damage = self.z0#(1.0 + posinputdata.shape[1]) / (2.0 + posinputdata.shape[1] + neginputdata.shape[1])
                    p_damage_loc = np.exp(logp_loc_giv_damage + np.log(p_damage))
                    p_nodamage_loc = np.exp(logp_loc_giv_nodamage + np.log(1 - p_damage))
                    p_damage_giv_loc  = p_damage_loc / (p_damage_loc + p_nodamage_loc)
                    return p_damage_giv_loc
        
                targets_single_arr = np.concatenate(( targetsx.reshape(1, len(targetsx)), targetsy.reshape(1, len(targetsy)) ))
                results['KDE'] = kde_prediction(targets_single_arr)
                densityresults['KDE'] = results['KDE']
                density_var['KDE'] = np.zeros(len(results['KDE']))
                
                logging.info("KDE complete.")
    
            # TRAIN GP WITHOUT BCC. Train using sample of ground truth density (not reports) to test if GP works--------            
            if 'GP' in self.methods:      
                
                logging.info("Using a density GP without BCC...")
                # Get values for training points by taking frequencies -- only one report at each location, so give 1 for
                # positive reports, 0 otherwise
                if not np.any(self.ls_initial) and self.optimise:
                    ls_initial = [2, 10, 50, 100]
                else:
                    ls_initial = [self.ls_initial]
                nlml = np.inf
                gpgrid_opt = None
                for ls in ls_initial:
                    rate_ls = 2.0 / ls
                    self.gpgrid = GPGrid(nx, ny, z0=self.z0, shape_s0=self.shape_s0, rate_s0=self.rate_s0, shape_ls=2.0, 
                                         rate_ls=rate_ls)
                    self.gpgrid.verbose = self.verbose
#                     self.gpgrid.p_rep = 0.9
                    
                    if self.optimise:                    
                        self.gpgrid.optimize([reportsx, reportsy], np.concatenate((posreports[:,np.newaxis], 
                                                       (posreports+negreports)[:,np.newaxis]), axis=1), maxfun=100)
                        if self.gpgrid.nlml < nlml:
                            nlml = self.gpgrid.nlml
                            gpgrid_opt = self.gpgrid                        
                    else:
                        self.gpgrid.fit([reportsx, reportsy], np.concatenate((posreports[:,np.newaxis], 
                                                       (posreports+negreports)[:,np.newaxis]), axis=1) )

                if self.ls_initial==0:
                    self.ls_initial = gpgrid_opt.ls[0]
                if self.optimise:
                    self.gpgrid = gpgrid_opt
                logging.debug("GP found output scale %.5f" % self.gpgrid.s)
                gp_preds, gp_var = self.gpgrid.predict([targetsx, targetsy], variance_method='sample')
                results['GP'] = gp_preds
                densityresults['GP'] = gp_preds
                density_var['GP'] = gp_var
            elif not np.any(self.ls_initial):
                self.ls_initial = nx
                
            logging.info("Chosen self.ls_initial=%.2f" % self.ls_initial)
                
            # default hyper-parameter initialisation points for all the GPs used below
            shape_ls = 2.0
            rate_ls = shape_ls / self.ls_initial
    
            # RUN SEPARATE IBCC AND GP STAGES ----------------------------------------------------------------------------------
            if 'IBCC+GP' in self.methods:
                # Should this be dropped from the experiments, as it won't work without gridding the points? --> then run into
                # question of grid size etc. Choose grid so that squares have an average of 3 reports?
                logging.info("Running separate IBCC and GP...")
        
                def train_gp_on_ibcc_output(opt_nx, opt_ny):
                    # set the initial length scale according to the grid size
                    ls_initial = (self.ls_initial / float(nx)) * opt_nx
                    rate_ls = shape_ls / ls_initial
                    
                    # run standard IBCC
                    self.gpgrid2 = GPGrid(opt_nx, opt_ny, z0=self.z0, shape_s0=self.shape_s0, rate_s0=self.rate_s0,
                                           shape_ls=shape_ls, rate_ls=rate_ls)
                    self.gpgrid2.verbose = self.verbose
                    self.ibcc_combiner = IBCC(2, 2, alpha0, self.nu0, K)                
                    self.ibcc_combiner.clusteridxs_alpha0 = clusteridxs
                    self.ibcc_combiner.verbose = self.verbose
                    self.ibcc_combiner.min_iterations = 5
                    self.ibcc_combiner.max_iterations = 200
                    self.ibcc_combiner.conv_threshold = 0.1    
                    
                    #opt_nx = np.ceil(np.exp(hyperparams[0]))
                    #opt_ny = np.ceil(np.exp(hyperparams[1]))
                    logging.debug("fmin gridx and gridy values: %f, %f" % (opt_nx, opt_ny))
                    if opt_nx <= 0 or opt_ny <= 0 or np.isnan(opt_nx) or np.isnan(opt_ny):
                        return np.inf
                    self.gpgrid2.nx = opt_nx
                    self.gpgrid2.ny = opt_ny
    
                    # grid indicating where we made observations
                    reportsx_grid = (reportsx * opt_nx/float(nx)).astype(int)
                    reportsy_grid = (reportsy * opt_ny/float(ny)).astype(int)
                    
                    reportsy_grid = reportsy_grid[reportsx_grid < opt_nx]
                    reportsx_grid = reportsx_grid[reportsx_grid < opt_nx]
                    
                    reportsx_grid = reportsx_grid[reportsy_grid < opt_ny]
                    reportsy_grid = reportsy_grid[reportsy_grid < opt_ny]
    
                    obs_grid = coo_matrix((np.ones(reportsx_grid.shape[0]), (reportsx_grid, reportsy_grid)), (opt_nx, opt_ny))
                    # Coordinates where we made observations, i.e. without duplicates due to multiple reports at some points
                    obs_coords = np.argwhere(obs_grid.toarray()>0)
                    obsx = obs_coords[:, 0]
                    obsy = obs_coords[:, 1]
    
                    #flatten the input data so it can be used with standard IBCC
                    linearIdxs = np.ravel_multi_index((reportsx_grid, reportsy_grid), dims=(opt_nx, opt_ny))
                    C_valid = C[C[:, 1] < nx, :]
                    C_valid = C_valid[C_valid[:, 2] < ny, :]
                    C_flat = C_valid[:,[0,1,3]]
                    C_flat[:,1] = linearIdxs
                    bcc_pred = self.ibcc_combiner.combine_classifications(C_flat, optimise_hyperparams=False)
                    bcc_pred = bcc_pred[np.ravel_multi_index((obsx, obsy), dims=(opt_nx, opt_ny)), 1]
    
                    # use IBCC output to train GP
                    if self.optimise:
                        self.gpgrid2.optimize([obsx, obsy], bcc_pred, maxfun=100)
                    else:
                        self.gpgrid2.fit([obsx, obsy], bcc_pred)
                    pT, _ = self.gpgrid2.predict([reportsx_grid,  reportsy_grid], variance_method='sample') # use the report locations
                    pT = np.concatenate((pT.T, 1-pT.T), axis=0)
                    ls = self.gpgrid2.ls
                    if self.gpgrid2.verbose:
                        logging.debug("fmin param value for lengthscale: %f, %f" % (ls[0], ls[1]))
                    
                    self.ibcc_combiner.lnjoint(alldata=True)
                    lnpCT = np.sum(pT * (self.ibcc_combiner.lnPi[:, C_flat[:, 2].astype(int), C_flat[:, 0].astype(int)] 
                                         + self.ibcc_combiner.lnkappa))
                    lnpPi = self.ibcc_combiner.post_lnpi()
                    lnpKappa = self.ibcc_combiner.post_lnkappa()
                    EEnergy = lnpCT + lnpPi + lnpKappa
                     
                    # Entropy of the variational distribution
                    lnqT = np.sum(pT[pT != 0] * np.log(pT[pT != 0])) #self.ibcc_combiner.q_ln_t()
                    lnqPi = self.ibcc_combiner.q_lnPi()
                    lnqKappa = self.ibcc_combiner.q_lnkappa()
                    H = lnqT + lnqPi + lnqKappa
            
                    nlml = - (EEnergy - H)
                    
                    if self.gpgrid2.verbose:
                        logging.debug("NLML: " + str(nlml))
                    return nlml
                
                # try different levels of separation with on average 3 data points per grid square, 5 per grid square and 10 per grid square.
                Nper_grid_sq = [3, 5, 10]
                for grouping in Nper_grid_sq:
                    gridsize = int(np.ceil(len(reportsx) / grouping) )
                    nlml = train_gp_on_ibcc_output(gridsize, gridsize)
                    logging.debug("NLML = %.2f" % nlml)
                    targetsx_grid = (targetsx * gridsize / float(nx)).astype(int)
                    targetsy_grid = (targetsy * gridsize / float(ny)).astype(int)
                    gp_preds, gp_var = self.gpgrid2.predict([targetsx_grid, targetsy_grid], variance_method='sample')
                    results['IBCC+GP_%i' % grouping] = gp_preds
                    densityresults['IBCC+GP_%i' % grouping] = gp_preds
                    density_var['IBCC+GP_%i' % grouping] = gp_var
    
            # RUN HEAT MAP BCC ---------------------------------------------------------------------------------------------        
            if 'HeatmapBCC' in self.methods:
                #HEATMAPBCC OBJECT
                self.heatmapcombiner = HeatMapBCC(nx, ny, 2, 2, alpha0, K, z0=self.z0, shape_s0=self.shape_s0, 
                              rate_s0=self.rate_s0, shape_ls=shape_ls, rate_ls=rate_ls, force_update_all_points=True)
                self.heatmapcombiner.min_iterations = 4
                self.heatmapcombiner.max_iterations = 200
                self.heatmapcombiner.verbose = self.verbose
                self.heatmapcombiner.uselowerbound = True
                
                logging.info("Running HeatmapBCC...")
                # to do:
                # make sure optimise works
                # make sure the optimal hyper-parameters are passed to the next iteration
                #self.heatmapcombiner.clusteridxs_alpha0 = clusteridxs
                self.heatmapcombiner.combine_classifications(C, optimise_hyperparams=self.optimise)
                logging.debug("output scale: %.5f" % self.heatmapcombiner.heatGP[1].s)
    
                bcc_pred, rho_mean, rho_var = self.heatmapcombiner.predict(targetsx, targetsy, variance_method='sample')
                results['HeatmapBCC'] = bcc_pred[1, :] # only interested in positive "damage class"
                densityresults['HeatmapBCC'] = rho_mean[1, :]
                density_var['HeatmapBCC'] = rho_var[1, :]
    
            # EVALUATE ALL RESULTS -----------------------------------------------------------------------------------------
            if np.any(building_density) and np.any(gold_density):
            
                evaluator = Evaluator("", "BCCHeatmaps", "Ushahidi_Haiti_Building_Damage")
    
                gold_density = gold_density.flatten()
    
                for method in results:
                    print ''
                    print 'Results for %s with %i labels' % (method, Nlabels)
                    pred = results[method]
                    est_density = densityresults[method]
                    est_density_var = density_var[method].flatten()
        
                    best_thresholds = []
                    mced = []
                    rmsed = []
                    tau = []
                    testresults = pred.flatten()
                    testresults[testresults==0] = 0.000001 # use a very small value to avoid log errors with cross entropy
                    testresults[testresults==1] = 0.999999
    
                    testresults = testresults[building_density>=0]
                    building_density = building_density[building_density>=0]
    
                    # This will be the same as MCED unless we "snap-to-grid" so that reports and test locations overlap
                    mce = - np.sum(building_density * np.log(testresults)) - np.sum((1-building_density) * np.log(1 - testresults))
                    mce = mce / float(len(building_density))
                    
                    rmse = np.sqrt( np.mean((testresults - building_density)**2) )
    
                    auc_by_class, _, _ = evaluator.eval_auc(testresults, building_density)
                    if testresults.ndim == 2:
                        auc = np.sum(np.bincount(building_density) * auc_by_class) / len(building_density)
                    else:
                        auc = auc_by_class
            
                    print "Cross entropy (individual data points): %.4f" % (mce)
                    print "RMSE (individual data points): %.4f" % (rmse)
                    print "AUC (individual data points): %.4f; best threshold %.2f" % (auc, np.sum(best_thresholds) / float(len(building_density)) )
        
                    # assume gold density and est density have 1 row for each class
                    est_density = est_density.flatten()
                    est_density[est_density==0] = 0.0000001
                    est_density[est_density==1] = 0.9999999
        
                    mced = nlpd_beta(gold_density, est_density, est_density_var) 
                    print "Cross entropy (density estimation): %.4f" % mced
        
                    rmsed = np.sqrt( np.mean((est_density - gold_density)**2) )
                    print "RMSE (density estimation): %.4f" % rmsed
        
                    tau, _ = kendalltau(est_density, gold_density)
                    if np.isnan(tau):
                        print "Kendall's Tau --> NaNs are mapped to zero for plotting"
                        tau = 0
                    print "Kendall's Tau (density estimation): %.4f " % tau
        
                    if method not in self.auc_all:
                        self.auc_all[method] = [auc]
                        self.rmse_all[method] = [rmse]
                        self.mce_all[method] = [mce]
        
                        self.tau_all[method] = [tau]
                        self.rmsed_all[method] = [rmsed]
                        self.mced_all[method] = [mced]
                    else:
                        self.auc_all[method].append(auc)
                        self.rmse_all[method].append(rmse)
                        self.mce_all[method].append(mce)
        
                        self.tau_all[method].append(tau)
                        self.rmsed_all[method].append(rmsed)
                        self.mced_all[method].append(mced)
    
            # set up next iteration
            self.results_all[Nlabels] = results
            self.densityresults_all[Nlabels] = densityresults
            self.densityVar_all[Nlabels] = density_var
            if Nlabels==C_all.shape[0]:
                break
            else:
                Nlabels += Nrep_inc
                if C_all.shape[0] < Nlabels:
                    Nlabels = C_all.shape[0]
    
    
    def save_separate_results(self):
        if not os.path.exists(self.outputdir):
            os.mkdir(self.outputdir)
        np.save(self.outputdir + "results.npy", self.results_all)
        np.save(self.outputdir + "density_results.npy", self.densityresults_all)
        np.save(self.outputdir + "density_var.npy", self.densityVar_all)
        np.save(self.outputdir + "rmse.npy", self.rmse_all)
        np.save(self.outputdir + "auc.npy", self.auc_all)
        np.save(self.outputdir + "mce.npy", self.mce_all)
        np.save(self.outputdir + "rmsed.npy", self.rmsed_all)
        np.save(self.outputdir + "tau.npy", self.tau_all)
        np.save(self.outputdir + "mced.npy", self.mced_all)
        
    def save_self(self):
        np.save(self.outputdir + "tester.npy", self)
        
def nlpd_beta(gold, est_mean, est_var):
    '''
    This should be the same as cross entropy. Gives the negative log probability density of a ground-truth density value
    according to a beta distribution with given mean and variance.
    '''
    a_plus_b = (1.0 / est_var) * est_mean * (1-est_mean) - 1
    a = est_mean * a_plus_b
    b = (1-est_mean) * a_plus_b
    
    # gold density will break if it's actually set to zero or one.
    minval = 1e-6
    gold[gold > 1.0 - minval] = 1.0 - minval
    gold[gold < minval] = minval
        
    a[a<minval] = minval
    b[b<minval] = minval
    
    nlpd = np.sum(- beta.logpdf(gold, a, b)) 
    return nlpd / len(gold) # we return the mean per data point
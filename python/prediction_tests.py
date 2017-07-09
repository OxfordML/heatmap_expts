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
from gp_classifier_vb import GPClassifierVB
from scipy.sparse import coo_matrix
from scipy.stats import gaussian_kde, kendalltau, beta
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn import svm

class Tester(object):
        
    def __init__(self, outputdir, methods, Nlabels_all, z0, alpha0_all, nu0, shape_s0, rate_s0, ls_initial=0, 
                 optimise=True, clusteridxs_all=None, verbose=False, lpls=None):
        
        # Controls whether we optimise hyper-parameters, including length scale
        self.optimise = optimise
        
        # Set this to zero to use the optimal value from the standard GP
        self.ls_initial = ls_initial
        
        if isinstance(ls_initial, np.ndarray) or isinstance(ls_initial, list):
            # we have to marginalise over the list of length scales with uniform prior
            self.margls = True
            if np.any(lpls):
                self.lpls = lpls
            else:
                self.lpls = np.zeros(len(ls_initial)) + np.log(1.0 / len(ls_initial))
        else:
            self.margls = False
        
        self.shape_s0 = shape_s0
        self.rate_s0 = rate_s0
        
        self.outputdir = outputdir
        self.methods = methods
        self.Nlabels_all = Nlabels_all
        self.z0 = z0
        self.alpha0_all = alpha0_all
        
        self.L = self.alpha0_all.shape[1]
        
        self.nu0 = nu0
        self.clusteridxs_all = clusteridxs_all

        self.results_all = {}
        self.densityresults_all = {}
        self.densityVar_all = {}
        self.auc_all = {}
        self.mce_all = {}
        self.kl_all = {}
        self.rmse_all = {}
        self.errvar_all = {}

        self.tau_all = {}
        self.mced_all = {}
        self.rmsed_all = {}
        
        self.acc_all = {}
        self.pos_all = {}
        self.neg_all = {}
        self.precision_all = {}
        self.recall_all = {}
        
        self.gpgrid = None
        self.heatmapcombiner = None
        self.gpgrid2 = None
        self.ibcc_combiner = None
        
        self.verbose = verbose
        
        self.ignore_report_point_density = False
    
    def evaluate_discrete(self, results, densityresults, Nlabels, gold_labels, gold_density):
        evaluator = Evaluator("", "BCCHeatmaps", "Ushahidi_Haiti_Building_Damage")
        
        for method in results:
            print ''
            print 'Results for %s with %i labels' % (method, Nlabels)
            
            # scores given to each data point indicating strength of class membership -- may not always be probabilities
            pred_scores = results[method] 
            
            # some metrics require probabilities...
            if 'SVM' in method:
                pred_probs = densityresults[method]
            else:
                pred_probs = pred_scores
            
            best_thresholds = []
            testresults = pred_probs.flatten()
            testresults[testresults==0] = 0.000001 # use a very small value to avoid log errors with cross entropy
            testresults[testresults==1] = 0.999999

            pos_percent = (np.sum(testresults >= 0.5) / float(len(testresults)))
            neg_percent = (np.sum(testresults < 0.5) / float(len(testresults)))

            testresults_scores = pred_scores.flatten()[gold_labels>=0]
            testresults = testresults[gold_labels>=0]
            gold_labels_i = gold_labels[gold_labels>=0]
            gold_density_i = gold_density[gold_density>=0]

            #Use only confident labels -- useful for e.g. PRN
            testresults_scores = testresults_scores[(gold_labels_i>=0.9) | (gold_labels_i <= 0.1)]
            testresults = testresults[(gold_labels_i>=0.9) | (gold_labels_i <= 0.1)]
            gold_labels_i = gold_labels_i[(gold_labels_i>=0.9) | (gold_labels_i <= 0.1)]

            # This will be the same as MCED unless we "snap-to-grid" so that reports and test locations overlap
            mce = - np.sum(gold_labels_i * np.log(testresults)) - np.sum((1-gold_labels_i) * np.log(1 - testresults))
            mce = mce / float(len(gold_labels_i))
            
            rmse = np.sqrt( np.mean((testresults - gold_labels_i)**2) )
            error_var = np.var(testresults - gold_labels_i)

            discrete_gold = np.round(gold_labels_i)

            auc_by_class, _, _ = evaluator.eval_auc(testresults_scores, discrete_gold)
            if testresults.ndim == 2:
                auc = np.sum(np.bincount(discrete_gold) * auc_by_class) / len(discrete_gold)
            else:
                auc = auc_by_class
                
            kl = mce - ( np.sum(gold_density_i * np.log(gold_density_i)) / float(len(gold_labels_i)) )
            mce = np.log2(np.e) * mce # in bits, but KL in nats
            
            acc = np.sum(np.round(testresults)==discrete_gold) / float(len(gold_labels_i))
            tp = np.sum((testresults >= 0.5) & (gold_labels_i >= 0.5))
            fp = np.sum((testresults >= 0.5) & (gold_labels_i < 0.5))
            if tp > 0:
                precision = tp / float(tp + fp)
            else:
                precision = 0
                
            recall = tp / float(np.sum(gold_labels_i >= 0.5 ))
            print "Classification accuracy: %.4f" % acc
            print "Percentage marked as +ve: %.4f" % pos_percent
            print "Percentage marked as -ve: %.4f" % neg_percent
            print "Precision: %.4f" % precision
            print "Recall: %.4f" % recall
    
            print "Cross entropy (individual data points): %.4f" % (mce)
            print "RMSE (individual data points): %.4f" % (rmse)
            print "Error variance: %.4f" % error_var
            print "AUC (individual data points): %.4f; best threshold %.2f" % (auc, np.sum(best_thresholds) / float(len(gold_labels)) )                    

            if method not in self.auc_all:
                self.auc_all[method] = [auc]
                self.rmse_all[method] = [rmse]
                self.errvar_all[method] = [error_var]
                self.mce_all[method] = [mce]
                self.kl_all[method] = [kl]
                
                self.acc_all[method] = [acc]
                self.pos_all[method] = [pos_percent]
                self.neg_all[method] = [neg_percent]
                self.precision_all[method] = [precision]
                self.recall_all[method] = [recall]
            else:
                self.auc_all[method].append(auc)
                self.rmse_all[method].append(rmse)
                self.errvar_all[method].append(error_var)
                self.mce_all[method].append(mce)
                self.kl_all[method].append(kl)

                self.acc_all[method].append(acc)
                self.pos_all[method].append(pos_percent)
                self.neg_all[method].append(neg_percent)
                self.precision_all[method].append(precision)
                self.recall_all[method].append(recall)          
                     
    def evaluate_density(self, densityresults, density_var, Nlabels, gold_density, C_all, targetsx, targetsy, nx, ny):
   
        gold_density_i = gold_density.flatten()

        for method in densityresults:
            print ''
            print 'Results for %s with %i labels' % (method, Nlabels)
            
            mced = []
            rmsed = []
            tau = []                    
            
            est_density = densityresults[method]
            est_density_var = density_var[method].flatten()
    
            # assume gold density and est density have 1 row for each class
            est_density = est_density.flatten()
            est_density[est_density==0] = 0.0000001
            est_density[est_density==1] = 0.9999999
            
            # ignore points with reports
            if self.ignore_report_point_density:
                #idxs with reports
                repids = np.unique(np.ravel_multi_index((C_all[:, 1], C_all[:, 2]), dims=(nx, ny)))
                targetids = np.ravel_multi_index((targetsx, targetsy), dims=(nx, ny))
                nonreppoints = np.logical_not(np.in1d(targetids, repids))
                est_density = est_density[nonreppoints]
                est_density_var = est_density_var[nonreppoints]
                gold_density_i = gold_density.flatten()[nonreppoints]  
                
            #remove any blanked out locations where density is invalid
            est_density = est_density[gold_density_i >= 0]     
            est_density_var = est_density_var[gold_density_i >= 0]
            gold_density_i = gold_density_i[gold_density_i >= 0]                                      

            mced = nlpd_beta(gold_density_i, est_density, est_density_var) 
            mced = np.log2(np.e) * mced # in bits
            print "Cross entropy (density estimation): %.4f" % mced

            rmsed = np.sqrt( np.mean((est_density - gold_density_i)**2) )
            print "RMSE (density estimation): %.4f" % rmsed
            
            tau, _ = kendalltau(est_density, gold_density_i)
            if np.isnan(tau):
                print "Kendall's Tau --> NaNs are mapped to zero for plotting"
                tau = 0
            print "Kendall's Tau (density estimation): %.4f " % tau

            if method not in self.tau_all:
                self.tau_all[method] = [tau]
                self.rmsed_all[method] = [rmsed]
                self.mced_all[method] = [mced]

            else:
                self.tau_all[method].append(tau)
                self.rmsed_all[method].append(rmsed)
                self.mced_all[method].append(mced)        
    
    def run_tests(self, C_all, nx, ny, targetsx, targetsy, gold_labels, gold_density, Nlabels, Nrep_inc):

        nx = int(nx)
        ny = int(ny)

        C = C_all
    
        while Nlabels <= self.Nlabels_all:
            
            logging.info("Running methods with %d labels" % Nlabels)
            
            C = C_all[0:Nlabels, :]
            agents = np.unique(C[:,0])
            K = int(np.max(agents)+1)
            if not np.any(self.clusteridxs_all):
                self.clusteridxs_all = np.zeros(K, dtype=int)
            clusteridxs = self.clusteridxs_all[0:K]
            alpha0 = self.alpha0_all
    
            results = {}
            densityresults = {}
            density_var = {}
    
            # indicator array to show whether reports are positive or negative
            posreports = (C[:, 3] >= 1).astype(float) # it previously had '==' instead of '>='
            negreports = (C[:, 3] == 0).astype(float)
    
            # Report coords for this round
            reportsx = C[:, 1]
            reportsy = C[:, 2]
            
# SIMPLE BASELINES ------------------------------------------------------------------------------------------------

            if 'MV' in self.methods:
                posinputdata  = np.vstack((reportsx[posreports>0], reportsy[posreports>0]))
                neginputdata = np.vstack((reportsx[negreports>0], reportsy[negreports>0]))                
                
                poscounts = coo_matrix((np.ones(posinputdata.shape[1], dtype=float), 
                                        (posinputdata[0,:], posinputdata[1,:])), 
                                       shape=(nx, ny)).tocsr()
                negcounts = coo_matrix((np.ones(neginputdata.shape[1], dtype=float), 
                                        (neginputdata[0,:], neginputdata[1,:])), 
                                       shape=(nx, ny)).tocsr()
                totals = poscounts + negcounts
                
                fraction_pos = poscounts[reportsx, reportsy] / totals[reportsx, reportsy]
                
                tr_idxs = [np.argwhere((reportsx==targetsx[i]) & (reportsy==targetsy[i]))[0][0] if 
                           np.any((reportsx==targetsx[i]) & (reportsy==targetsy[i])) else -1 for i in range(len(targetsx))]
                targets_single_arr = np.concatenate(( targetsx.reshape(1, len(targetsx)),
                                                      targetsy.reshape(1, len(targetsy)) ))

                densityresults['MV'] = np.array([fraction_pos[0, i] if i > -1 else 0.5 for i in tr_idxs])
                results['MV'] = np.round(densityresults['MV'])
                density_var['MV'] = np.zeros(len(results['MV']))
                
                logging.info("MV complete.")                

            if 'oneclassSVM' in self.methods:
                posinputdata  = np.vstack((reportsx[posreports>0], reportsy[posreports>0])).T
                svc = svm.OneClassSVM()
                
                svc.fit(posinputdata)
                
                targets_single_arr = np.hstack((targetsx[:, np.newaxis], targetsy[:, np.newaxis]))

                results['1-class SVM'] = svc.decision_function(targets_single_arr) # confidence scores not probabilities
                densityresults['1-class SVM'] = svc.predict(targets_single_arr) # need values between 0 and 1 for this. no probabilities available, so only have discrete
                density_var['1-class SVM'] = np.zeros(len(results['1-class SVM']))
                
                logging.info("SVM complete.") 

            if 'SVM' in self.methods:
                svc = svm.SVC(probability=True)
                svc.fit(C[:, 1:3], posreports)
                
                targets_single_arr = np.hstack((targetsx[:, np.newaxis], targetsy[:, np.newaxis]))

                results['SVM'] = svc.decision_function(targets_single_arr)
                densityresults['SVM'] = svc.predict_proba(targets_single_arr)[:, 1] # confidence scores not probabilities
                density_var['SVM'] = np.zeros(len(results['SVM']))
                
                logging.info("SVM complete.") 
                                
            if 'NN' in self.methods:
                nn_classifier = KNeighborsClassifier()
                nn_classifier.fit(C[:, 1:3], posreports)
                
                targets_single_arr = np.hstack((targetsx[:, np.newaxis], targetsy[:, np.newaxis]))

                results['NN'] = nn_classifier.predict(targets_single_arr)
                densityresults['NN'] = nn_classifier.predict_proba(targets_single_arr)
                if densityresults['NN'].shape[1] == 2:
                    densityresults['NN'] = densityresults['NN'][:, 1]
                density_var['NN'] = np.zeros(len(results['NN']))
                
                logging.info("NN complete.")                
    
# KERNEL DENSITY ESTIMATION ---------------------------------------------------------------------------------------
            if 'KDE' in self.methods:
                # Method used here performs automatic bandwidth determination - see help docs
                posinputdata  = np.vstack((reportsx[posreports>0], reportsy[posreports>0]))
                neginputdata = np.vstack((reportsx[negreports>0], reportsy[negreports>0]))
                logging.info("Running KDE... pos data size: %i, neg data size: %i " % (posinputdata.shape[1], neginputdata.shape[1]) )
                if posinputdata.shape[1] != 0:
                    try:
                        kdepos = gaussian_kde(posinputdata, 'scott')
                    except:
                        logging.error("Couldn't run KDE")
                        kdepos = []
                
                if neginputdata.shape[1] != 0:
                    try:
                        kdeneg = gaussian_kde(neginputdata, 'scott')
                    except:
                        logging.error("Couldn't run KDE")
                        kdeneg = []
                    
                def logit(a):
                    a[a==1] = 1 - 1e-6
                    a[a==0] = 1e-6
                    return np.log(a / (1-a))
    
                def kde_prediction(targets):
                    
                    # start with a flat function
                    logp_loc_giv_damage = np.log(1.0 / (nx * ny * 2.0))
                    logp_loc_giv_nodamage = np.log(1.0 / (nx * ny * 2.0))
                    # put kernel density estimator over it, weighted by number of data points
                    w_damage = posinputdata.shape[1] / float(posinputdata.shape[1] + 5)
                    w_nodamage = neginputdata.shape[1] / float(neginputdata.shape[1] + 5)
                    if posinputdata.shape[1] != 0 and kdepos:
                        logp_loc_giv_damage = logp_loc_giv_damage*(1-w_damage) + np.log(kdepos.evaluate(targets))*w_damage
                            
                    if neginputdata.shape[1] != 0 and kdeneg:
                        logp_loc_giv_nodamage = logp_loc_giv_nodamage*(1-w_nodamage) + np.log(kdeneg.evaluate(targets))*w_nodamage
    
                    p_damage = self.z0
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
                
                if not self.optimise and self.margls:
                    ls_initial = self.ls_initial
                    lpls_data = np.zeros(len(ls_initial))
                    gp_preds = np.zeros((len(ls_initial), len(targetsx))) 
                    gp_var = np.zeros((len(ls_initial), len(targetsx)))
                else:
                    ls_initial = np.array([self.ls_initial]).flatten()
                    
                nlml = np.inf
                gpgrid_opt = None
                for l, ls in enumerate(ls_initial):
                    rate_ls = 2.0 / ls
                    self.gpgrid = GPClassifierVB(2, z0=self.z0, shape_s0=self.shape_s0, rate_s0=self.rate_s0, 
                                                 shape_ls=2.0, rate_ls=rate_ls, ls_initial=[ls])
                    self.gpgrid.verbose = self.verbose
#                     self.gpgrid.p_rep = 0.9
                    
                    if self.optimise:                    
                        _, current_nlml = self.gpgrid.fit([reportsx, reportsy], np.concatenate((posreports[:,np.newaxis], 
                                           (posreports+negreports)[:,np.newaxis]), axis=1), 
                                        optimize=True, maxfun=100)
                        if current_nlml < nlml:
                            nlml = current_nlml
                            gpgrid_opt = self.gpgrid       
                    else:
                        self.gpgrid.fit([reportsx, reportsy], np.concatenate((posreports[:,np.newaxis], 
                                                       (posreports+negreports)[:,np.newaxis]), axis=1) )
                        if self.margls:
                            lpls_data[l] = self.lpls[l] + self.gpgrid.lowerbound() 
                            preds, var = self.gpgrid.predict([targetsx, targetsy], variance_method='sample')
                            gp_preds[l, :] = preds.reshape(-1)
                            gp_var[l, :] = var.reshape(-1) 

                if self.ls_initial==0:
                    self.ls_initial = gpgrid_opt.ls[0]
                if self.optimise:
                    self.gpgrid = gpgrid_opt
                logging.debug("GP found output scale %.5f" % self.gpgrid.s)
                
                if not self.margls:
                    gp_preds, gp_var = self.gpgrid.predict([targetsx, targetsy], variance_method='sample')
                else:
                    lpls_data -= np.max(lpls_data)                    
                    pls_data = np.exp(lpls_data)
                    pls_giv_data = pls_data / np.sum(pls_data)
                    gp_preds = np.sum(gp_preds * pls_giv_data[:, np.newaxis], axis=0)
                    gp_var = np.sum(np.sqrt(gp_var) * pls_giv_data[:, np.newaxis], axis=0) ** 2
                    
                    print "GP LENGTH SCALE PROBABILITIES:"
                    print pls_giv_data
                    
                results['GP'] = gp_preds
                densityresults['GP'] = gp_preds
                density_var['GP'] = gp_var
            elif not np.any(self.ls_initial):
                self.ls_initial = nx
                               
# RUN SEPARATE IBCC AND GP STAGES ----------------------------------------------------------------------------------
            if 'IBCC+GP' in self.methods:
                # Should this be dropped from the experiments, as it won't work without gridding the points? --> then run into
                # question of grid size etc. Choose grid so that squares have an average of 3 reports?
                logging.info("Running separate IBCC and GP...")
        
                def train_gp_on_ibcc_output(opt_nx, opt_ny, ls_initial):
                    # set the initial length scale according to the grid size
                    ls_initial = (ls_initial / float(nx)) * opt_nx
                    
                    # default hyper-parameter initialisation points for all the GPs used below
                    shape_ls = 2.0
                    rate_ls = shape_ls / ls_initial
                    
                    # run standard IBCC
                    self.gpgrid2 = GPClassifierVB(2, z0=self.z0, shape_s0=self.shape_s0, rate_s0=self.rate_s0,
                                           shape_ls=shape_ls, rate_ls=rate_ls, ls_initial=[ls_initial])
                    self.gpgrid2.verbose = self.verbose
                    self.ibcc_combiner = IBCC(2, alpha0.shape[1], alpha0, self.nu0, K)                
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
                        self.gpgrid2.fit([obsx, obsy], bcc_pred, maxfun=100, optimize=True)
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
                #Nper_grid_sq = [3, 5, 10]
                #for grouping in Nper_grid_sq:
                #gridsize = int(np.ceil(len(reportsx) / grouping) )
                
                if self.margls:
                    gp_preds = np.zeros((len(self.ls_initial), len(targetsx))) 
                    gp_var = np.zeros((len(self.ls_initial), len(targetsx)))
                    lpls_data = np.zeros(len(self.ls_initial))
                    for l, ls_initial in enumerate(self.ls_initial):
                        nlml = train_gp_on_ibcc_output(nx, ny, ls_initial)#gridsize, gridsize)
                        logging.debug("NLML = %.2f" % nlml)
                        targetsx_grid = targetsx#(targetsx * gridsize / float(nx)).astype(int)
                        targetsy_grid = targetsy#(targetsy * gridsize / float(ny)).astype(int)
                        gppredsl, gpvarl = self.gpgrid2.predict([targetsx_grid, targetsy_grid], variance_method='sample')
                        gp_preds[l, :] = gppredsl.flatten()
                        gp_var[l, :] = gpvarl.flatten()                    
                        lpls_data[l] = self.lpls[l] + self.gpgrid2.lowerbound() 
                      
                    lpls_data -= np.max(lpls_data)  
                    pls_data = np.exp(lpls_data)
                    pls_giv_data = pls_data / np.sum(pls_data)
                    gp_preds = np.sum(gp_preds * pls_giv_data[:, np.newaxis], axis=0)
                    gp_var = np.sum(np.sqrt(gp_var) * pls_giv_data[:, np.newaxis], axis=0) ** 2
                    
                else:
                    nlml = train_gp_on_ibcc_output(nx, ny, self.ls_initial)#gridsize, gridsize)
                    logging.debug("NLML = %.2f" % nlml)
                    targetsx_grid = targetsx#(targetsx * gridsize / float(nx)).astype(int)
                    targetsy_grid = targetsy#(targetsy * gridsize / float(ny)).astype(int)
                    gp_preds, gp_var = self.gpgrid2.predict([targetsx_grid, targetsy_grid], variance_method='sample')
                    
                #results['IBCC+GP_%i' % grouping] = gp_preds
                #densityresults['IBCC+GP_%i' % grouping] = gp_preds
                #density_var['IBCC+GP_%i' % grouping] = gp_var
                results['IBCC+GP'] = gp_preds
                densityresults['IBCC+GP'] = gp_preds
                density_var['IBCC+GP'] = gp_var
                    
# RUN IBCC -- will only detect the report locations properly, otherwise defaults to kappa ------------------
            if 'IBCC' in self.methods:
                self.ibcc2 = IBCC(2, alpha0.shape[1], alpha0, self.nu0, K, uselowerbound=True)                
                self.ibcc2.clusteridxs_alpha0 = clusteridxs
                self.ibcc2.verbose = self.verbose
                self.ibcc2.min_iterations = 5
                
                report_coords = (C[:,1].astype(int), C[:, 2].astype(int))
                linearIdxs = np.ravel_multi_index(report_coords, dims=(nx, ny))
                C_flat = C[:,[0,1,3]]
                C_flat[:,1] = linearIdxs
                testidxs = np.ravel_multi_index((targetsx.astype(int), targetsy.astype(int)), dims=(nx, ny))
                bintestidxs = np.zeros(np.max([np.max(testidxs), np.max(linearIdxs)]) + 1, dtype=bool)
                bintestidxs[testidxs] = True
                bintestidxs[linearIdxs] = True # make sure we use all observed data points during inference
                bcc_pred = self.ibcc2.combine_classifications(C_flat, optimise_hyperparams=False, testidxs=bintestidxs)
                                
                results['IBCC'] = bcc_pred[testidxs, 1]
                densityresults['IBCC'] = results['IBCC']
                density_var['IBCC'] = np.zeros(len(results['IBCC']))
    
# RUN HEAT MAP BCC ---------------------------------------------------------------------------------------------        
            if 'HeatmapBCC' in self.methods:
                
                if self.margls:
                    bcc_pred = np.zeros((len(self.ls_initial), len(targetsx))) 
                    rho_mean = np.zeros((len(self.ls_initial), len(targetsx)))
                    rho_var = np.zeros((len(self.ls_initial), len(targetsx)))

                    lpls_data = np.zeros(len(self.ls_initial))

                    for l, ls_initial in enumerate(self.ls_initial):                    
                    
                        # default hyper-parameter initialisation points for all the GPs used below
                        shape_ls = 2.0
                        rate_ls = shape_ls / ls_initial                    
                    
                        #HEATMAPBCC OBJECT
                        self.heatmapcombiner = HeatMapBCC(nx, ny, 2, alpha0.shape[1], alpha0, K, z0=self.z0, shape_s0=self.shape_s0, 
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
            
                        pred, densitymean, densityvar = self.heatmapcombiner.predict(targetsx, targetsy, variance_method='sample')
                        bcc_pred[l, :] = pred[1, :]
                        rho_mean[l, :] = densitymean[1, :]
                        rho_var[l, :] = densityvar[1, :]
                        
                        lpls_data[l] = self.lpls[l] + self.heatmapcombiner.lowerbound() 
            
                    lpls_data -= np.max(lpls_data)
                    pls_data = np.exp(lpls_data)
                    pls_giv_data = pls_data / np.sum(pls_data)
                    
                    print "LENGTH SCALE PROBABILITIES:"
                    print pls_giv_data
                    
                    bcc_pred = np.sum(bcc_pred * pls_giv_data[:, np.newaxis], axis=0)
                    rho_mean = np.sum(rho_mean * pls_giv_data[:, np.newaxis], axis=0)
                    rho_var = np.sum(np.sqrt(rho_var) * pls_giv_data[:, np.newaxis], axis=0) ** 2
                else:       
                    shape_ls = 2.0
                    rate_ls = shape_ls / self.ls_initial
                             
                    #HEATMAPBCC OBJECT
                    self.heatmapcombiner = HeatMapBCC(nx, ny, 2, alpha0.shape[1], alpha0, K, z0=self.z0, shape_s0=self.shape_s0, 
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
                    bcc_pred = bcc_pred[1, :] # only interested in positive "damage class"
                    rho_mean = rho_mean[1, :]
                    rho_var = rho_var[1, :]
                results['HeatmapBCC'] = bcc_pred
                densityresults['HeatmapBCC'] = rho_mean
                density_var['HeatmapBCC'] = rho_var
    
# EVALUATE ALL RESULTS -----------------------------------------------------------------------------------------
            if np.any(gold_labels):
                self.evaluate_discrete(results, densityresults, Nlabels, gold_labels, gold_density)

            if np.any(gold_density):
                self.evaluate_density(densityresults, density_var, Nlabels, gold_density, C_all, targetsx, targetsy, nx, ny)

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
                    
    def reevaluate(self, C_all, nx, ny, targetsx, targetsy, gold_labels, gold_density, Nlabels, Nrep_inc):
        
        self.results_all = np.load(self.outputdir + "results.npy").item()
        self.densityresults_all = np.load(self.outputdir + "density_results.npy").item()
        self.densityVar_all = np.load(self.outputdir + "density_var.npy").item()
        
        while Nlabels <= self.Nlabels_all:

            print self.results_all.keys()

            results = self.results_all[Nlabels]
            densityresults = self.densityresults_all[Nlabels]
            density_var = self.densityVar_all[Nlabels]
    
            # EVALUATE ALL RESULTS -----------------------------------------------------------------------------------------
            if np.any(gold_labels):
                self.evaluate_discrete(results, densityresults, Nlabels, gold_labels, gold_density)

            if np.any(gold_density):
                self.evaluate_density(densityresults, density_var, Nlabels, gold_density, C_all, targetsx, targetsy, nx, ny)

    
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
        np.save(self.outputdir + "errvar.npy", self.errvar_all)
        np.save(self.outputdir + "auc.npy", self.auc_all)
        np.save(self.outputdir + "kl.npy", self.kl_all)
        np.save(self.outputdir + "mce.npy", self.mce_all)
        np.save(self.outputdir + "rmsed.npy", self.rmsed_all)
        np.save(self.outputdir + "tau.npy", self.tau_all)
        np.save(self.outputdir + "mced.npy", self.mced_all)
        np.save(self.outputdir + "acc.npy", self.acc_all)
        np.save(self.outputdir + "pos.npy", self.pos_all)
        np.save(self.outputdir + "neg.npy", self.neg_all)
        np.save(self.outputdir + "precision.npy", self.precision_all)
        np.save(self.outputdir + "recall.npy", self.recall_all)
        
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
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
    
    def run_tests(self, C_all, nx, ny, targetsx, targetsy, gold_labels, gold_density, Nlabels, Nrep_inc):
        
        self.results_all = np.load(self.outputdir + "results.npy").item()
        self.densityresults_all = np.load(self.outputdir + "density_results.npy").item()
        self.densityVar_all = np.load(self.outputdir + "density_var.npy").item()
        
        C = C_all
        
        while Nlabels <= self.Nlabels_all:

            C = C_all[0:Nlabels, :]

            print self.results_all.keys()

            results = self.results_all[Nlabels]
            densityresults = self.densityresults_all[Nlabels]
            density_var = self.densityVar_all[Nlabels]
    
            # EVALUATE ALL RESULTS -----------------------------------------------------------------------------------------
            if np.any(gold_labels):
                evaluator = Evaluator("", "BCCHeatmaps", "Ushahidi_Haiti_Building_Damage")
                
                for method in results:
                    print ''
                    print 'Results for %s with %i labels' % (method, Nlabels)
                    
                    pred = results[method]
                    
                    best_thresholds = []
                    testresults = pred.flatten()
                    testresults[testresults==0] = 0.000001 # use a very small value to avoid log errors with cross entropy
                    testresults[testresults==1] = 0.999999
    
                    pos_percent = (np.sum(testresults >= 0.5) / float(len(testresults)))
                    neg_percent = (np.sum(testresults < 0.5) / float(len(testresults)))
    
                    testresults = testresults[gold_labels>=0]
                    gold_labels_i = gold_labels[gold_labels>=0]

                    #Use only confident labels -- useful for e.g. PRN
                    testresults = testresults[(gold_labels_i>=0.9) | (gold_labels_i <= 0.1)]
                    gold_labels_i = gold_labels_i[(gold_labels_i>=0.9) | (gold_labels_i <= 0.1)]
                    print "Evaluating with %i gold labels" % len(gold_labels_i)

                    # This will be the same as MCED unless we "snap-to-grid" so that reports and test locations overlap
                    mce = - np.sum(gold_labels_i * np.log(testresults)) - np.sum((1-gold_labels_i) * np.log(1 - testresults))
                    mce = mce / float(len(gold_labels_i))
                    
                    rmse = np.sqrt( np.mean((testresults - gold_labels_i)**2) )
                    error_var = np.var(testresults - gold_labels_i)
    
                    discrete_gold = np.round(gold_labels_i)
    
                    auc_by_class, _, _ = evaluator.eval_auc(testresults, discrete_gold)
                    if testresults.ndim == 2:
                        auc = np.sum(np.bincount(discrete_gold) * auc_by_class) / len(discrete_gold)
                    else:
                        auc = auc_by_class
                    
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
        
                        self.acc_all[method].append(acc)
                        self.pos_all[method].append(pos_percent)
                        self.neg_all[method].append(neg_percent)
                        self.precision_all[method].append(precision)
                        self.recall_all[method].append(recall)                        

            if np.any(gold_density):
    
                gold_density_i = gold_density.flatten()
    
                for method in results:
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
                        print "Evaluating with %i gold density points" % len(gold_density_i)
        
                    mced = nlpd_beta(gold_density_i, est_density, est_density_var) 
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
        np.save(self.outputdir + "rmse.npy", self.rmse_all)
        np.save(self.outputdir + "errvar.npy", self.errvar_all)
        np.save(self.outputdir + "auc.npy", self.auc_all)
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
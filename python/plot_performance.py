'''
Plot the results of the simulations with increasing numbers of data points. Show accuracy etc. 

Created on 23 Nov 2015

@author: edwin
'''

import numpy as np
import matplotlib.pyplot as plt
import logging, os

import run_synthetic_bias_nogrid as expmt_module
#import run_synthetic_bias as expmt_module
#import run_synthetic_noise as expmt_module
#import run_synthetic_noise_nogrid as expmt_module
#import ushahidi_loader_damage as expmt_module
#import ushahidi_loader_emergencies as expmt_module
#import prn_simulation as expmt_module

if hasattr(expmt_module, 'cluster_spreads'):
    cluster_spreads = expmt_module.cluster_spreads
elif hasattr(expmt_module, 'featurenames'):
    cluster_spreads = [expmt_module.featurenames[0]]
    print cluster_spreads
else:
    cluster_spreads = [0]

nruns = expmt_module.nruns
if hasattr(expmt_module, 'weak_proportions'):
    weak_proportions = expmt_module.weak_proportions
else:
    weak_proportions = [-1]

def get_data_dir(d, p, p_idx, cluster_spread):
    if p==-1:
        dataset_label = "d%i" % d # no proportion indices
    else:
        dataset_label = "p%f_d%i" % (p, d)
    
    expt_label = expmt_module.expt_label_template
    if '%' in expt_label:
        expt_label = expt_label % cluster_spread
         
    logging.info("Loading results for proportion %f, Dataset %d, cluster spread %s" % (p, d, str(cluster_spread)))
    datadir, _ = expmt_module.dataset_location(expt_label, dataset_label)
    return datadir

def get_output_dir(category, p, cluster_spread=0):    
    if p==-1:
        dataset_label = "" # no proportion indices
    else:
        dataset_label = "p%.3f" % p
    
    expt_label = expmt_module.expt_label_template
    if '%' in expt_label:
        expt_label = expt_label % cluster_spread
        
    outputdir =  './output/' 
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    if 'synth/' in expt_label and not os.path.isdir(outputdir + '/synth/'):
        os.mkdir(outputdir + '/synth/')        
    if 'prn4/' in expt_label and not os.path.isdir(outputdir + '/prn4/'):
        os.mkdir(outputdir + '/prn4/')        
    outputdir += expt_label + '/'
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir) 
    outputdir += category + '/'
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    outputdir += dataset_label
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    print "Using output dir %s" % outputdir
    return outputdir

def load_average_results(nruns, weak_proportions, filename, avg_type='median'):
    avg_results = {}
    lq_results = {}
    uq_results = {}
    
    nprops = len(weak_proportions)
    
    if avg_type == 'median':
        avgmethod = np.median
        def lq(r, axis): 
            return np.percentile(r, 25, axis=axis)
        lqmethod = lq
        def uq(r, axis): 
            return np.percentile(r, 75, axis=axis)
        uqmethod = uq
    elif avg_type == 'mean':
        avgmethod = np.mean
        def lq(r):
            return np.mean(r) - np.std(r) 
        lqmethod = lq
        def uq(r): 
            return np.mean(r) + np.std(r)
        uqmethod = uq
        
    for p_idx, p in enumerate(weak_proportions):
        for cluster_spread in cluster_spreads:
            results_pcs = {}
            for d in range(nruns):
                datadir = get_data_dir(d, p, p_idx, cluster_spread)
    
                if not os.path.exists(datadir + filename):
                    methods = None
                    continue
    
                current_results = np.load(datadir + filename).item()
                methods = current_results.keys()
                    
                if cluster_spread not in avg_results:
                    avg_results[cluster_spread] = {}
                    lq_results[cluster_spread] = {}
                    uq_results[cluster_spread] = {}                
                
                for m in methods:
                    if not m in results_pcs:
                        results_pcs[m] = np.zeros((nruns, len(Nreps_iter))) 
                    results_pcs[m][d, :] = current_results[m]
                    if p_idx == 0:
                        avg_results[cluster_spread][m] = np.zeros((nprops, len(current_results[m])))
                        lq_results[cluster_spread][m] = np.zeros((nprops, len(current_results[m])))
                        uq_results[cluster_spread][m] = np.zeros((nprops, len(current_results[m]))) 
        
            for m in results_pcs:
                avg_results[cluster_spread][m][p_idx, :] = avgmethod(results_pcs[m], axis=0)
                lq_results[cluster_spread][m][p_idx, :] = lqmethod(results_pcs[m], axis=0)
                uq_results[cluster_spread][m][p_idx, :] = uqmethod(results_pcs[m], axis=0)
                
    return avg_results, lq_results, uq_results, methods

def load_diff_results(nruns, weak_proportions, filename):
    avg_results = {}
    lq_results = {}
    uq_results = {}
    nprops = len(weak_proportions)
    for p_idx, p in enumerate(weak_proportions):
        for cluster_spread in cluster_spreads:
            results_pcs = {}
            for d in range(nruns):
                datadir = get_data_dir(d, p, p_idx, cluster_spread)
    
                current_results = np.load(datadir + filename).item()
                methods = current_results.keys()
                
                testmethod = 'HeatmapBCC' #compare this against the
                    
                if cluster_spread not in avg_results:
                    avg_results[cluster_spread] = {}
                    uq_results[cluster_spread] = {}
                    lq_results[cluster_spread] = {}                
                
                testresults = np.array(current_results[testmethod])
                
                for m in methods:
                    if not m in results_pcs:
                        results_pcs[m] = np.zeros((nruns, len(Nreps_iter))) 
                        
                    if not m==testmethod:
                        current_results[m] = testresults - np.array(current_results[m])
                        results_pcs[m][d, :] = current_results[m]
        
                    if p_idx == 0:
                        avg_results[cluster_spread][m] = np.zeros((nprops, len(current_results[m])))
                        lq_results[cluster_spread][m] = np.zeros((nprops, len(current_results[m])))
                        uq_results[cluster_spread][m] = np.zeros((nprops, len(current_results[m]))) 
        
            for m in results_pcs:
                avg_results[cluster_spread][m][p_idx, :] = np.median(results_pcs[m], axis=0)
                lq_results[cluster_spread][m][p_idx, :] = np.percentile(results_pcs[m], 25, axis=0)
                uq_results[cluster_spread][m][p_idx, :] = np.percentile(results_pcs[m], 75, axis=0)
    return avg_results, lq_results, uq_results, methods

def plot_performance(Nreps_iter, weak_proportions, plot_separately, cluster_spreads, filename, title, xlabel, ylabel, 
                     outfilename, plot_diffs = False, nats_to_bits=False, ylim=None):
    
    global plotnum # record the global plot number so we can add data to existing plots
    
    if plot_separately == 'nreps':
        separate_plot_variable = Nreps_iter
        separate_idx = 1
        xvals = weak_proportions
    elif plot_separately == 'proportions':
        separate_plot_variable = weak_proportions
        separate_idx = 0  
        xvals = Nreps_iter
    
    if plot_diffs:
        y_avg, y_l, y_u, methods = load_diff_results(nruns, weak_proportions, filename)
    else:
        y_avg, y_l, y_u, methods = load_average_results(nruns, weak_proportions, filename)

    if methods is None:
        logging.info('Could not find data file %s' % filename)
        return

    for lidx, l in enumerate(separate_plot_variable):
        for cs in cluster_spreads:
            plt.figure(plotnum)
            plt.title(title)
            for m in methods:
                if m in methods_to_skip:
                    continue
                
                if separate_idx == 1:
                    yvals = y_avg[cs][m][:, lidx]
                elif separate_idx == 0:
                    yvals = y_avg[cs][m][lidx, :]
                    
                if nats_to_bits:
                    yvals *= np.log2(np.e)
                     
                # don't plot the invalid outputs -- some metrics are not applicable to all methods and produce all NaNs
                if np.sum(np.isnan(yvals)) != len(yvals):
                    plt.plot(xvals, yvals, label=m, color=colors[m], marker=marks[m])
            plt.xlim(np.min(xvals), np.max(xvals))     
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            #plt.ylim(0, 1.0)
            plt.legend(loc='best')
            plt.grid(True)
    
            for m in methods:
                if m in methods_to_skip:
                    continue
                                
                if separate_idx == 1:
                    lvals = y_l[cs][m][:, lidx]
                    uvals = y_u[cs][m][:, lidx]
                elif separate_idx == 0:
                    lvals = y_l[cs][m][lidx, :]
                    uvals = y_u[cs][m][lidx, :]
                    
                if nats_to_bits:
                    uvals *= np.log2(np.e)
                    lvals *= np.log2(np.e)                    
                    
                plt.fill_between(xvals, lvals, uvals, alpha=0.1, edgecolor=colors[m], facecolor=colors[m])
            
            outputdir = get_output_dir(plot_separately, l, cs)
            plt.savefig(outputdir + "/%s.pdf" % (outfilename))
            plotnum += 1
            
            if ylim is not None:
                plt.ylim(ylim)

if __name__ == '__main__':
    plotnum = 0 # count plots. Can run script multiple times to plot different data on same plots.
    
    if hasattr(expmt_module, 'Nreports'):
        Nreports = expmt_module.Nreports
    else:
        _, Nreports, _, _, _, _ = expmt_module.load_data()
    
    # import settings from where the experiments were run
    if hasattr(expmt_module, "Nreps_initial"):
        Nreps_initial = expmt_module.Nreps_initial
    else:
        Nreps_initial = expmt_module.Nreps_initial_fraction * Nreports
        
    if Nreps_initial < 1:
        Nreps_initial = Nreps_initial * Nreports # Nreps_initial is the initial fraction
        
    if hasattr(expmt_module, 'Nrep_inc'):
        Nrep_inc = expmt_module.Nrep_inc
    else:
        Nrep_inc = (Nreports - Nreps_initial) / (expmt_module.nsteps - 1) 
          
    if hasattr(expmt_module, 'nsteps'):
        Nsteps = expmt_module.nsteps
    else:
        Nsteps = (Nreports - Nreps_initial) / float(Nrep_inc) + 1
    
    # get a set of x-coordinates for the number of reports at each iteration
    Nreps_iter = np.arange(Nsteps) * Nrep_inc + Nreps_initial
    
    colors = {'HeatmapBCC':'g',
              'IBCC':'r',
              'IBCC+GP':'purple',
              'GP':'saddlebrown',
              'KDE':'b', 
              'MV': 'cyan', 
              'NN': 'magenta',
              'SVM': 'y',
              'oneclassSVM': 'orange',
              '1-class SVM': 'orange'}
    
    marks = {'HeatmapBCC':'x',
              'IBCC':'^',
              'IBCC+GP':'o',
              'GP':'v',
              'KDE':'+',
              'MV': '>', 
              'NN': '<',
              'SVM': 'D',
              'oneclassSVM': '*',
              '1-class SVM': '*'}    
    
    methods_to_skip = []#['HeatmapBCC', 'GP', 'IBCC+GP']
        
    # load results for the density estimation        
    # Root mean squared error    
    plot_performance(Nreps_iter, weak_proportions, 'proportions', cluster_spreads, "rmsed.npy", 
                     'Root Mean Square Error of State Probabilities',
                     'Number of crowdsourced labels',
                     'RMSE',
                     "/rmsed.pdf")    
      
    # Mean Cross Entropy
    if hasattr(expmt_module, 'topdir') and expmt_module.topdir == 'prn4/':
        nats_to_bits = True # wasn't applied when these tests were run; in later tests, results are in bits already
    else:
        nats_to_bits = False    
    plot_performance(Nreps_iter, weak_proportions, 'proportions', cluster_spreads, "mced.npy",
                     "Negative Log Probability Density of State Probabilities",
                     'Number of crowdsourced labels', 
                     'NLPD or Cross Entropy (bits)',
                     "/mce_density", nats_to_bits=nats_to_bits)            
 
#     # Mean Cross Entropy
#     plot_performance(Nreps_iter, weak_proportions, 'proportions', cluster_spreads, "mced.npy",
#                      "Improvement of HeatmapBCC in \n Negative Log Probability Density of State Probabilities",
#                      'Number of crowdsourced labels', 
#                      'NLPD or Cross Entropy (bits)',
#                      "/mce_density_diff", True, nats_to_bits=nats_to_bits)

#     # KL divergence
#     plot_performance(Nreps_iter, weak_proportions, 'proportions', cluster_spreads, "kl.npy",
#                      "KL Divergence of State Probabilities",
#                      'Number of crowdsourced labels', 
#                      'KL-divergence',
#                      "/kl", False)
    
#     plot_performance(Nreps_iter, weak_proportions, 'proportions', cluster_spreads, "kl.npy",
#                      "Improvement of HeatmapBCC in \n KL Divergence of State Probabilities",
#                      'Number of crowdsourced labels', 
#                      'KL-divergence',
#                      "/kl_diff", False)

    # load results for predicting individual data points
    # Brier score
    plot_performance(Nreps_iter, weak_proportions, 'proportions', cluster_spreads, "rmse.npy",
                     "Brier Score",
                     'Number of crowdsourced labels', 
                     'Brier Score',
                     "/brier", False)
    
    # AUC
    plot_performance(Nreps_iter, weak_proportions, 'proportions', cluster_spreads, "auc.npy",
                     "AUC ROC",
                     'Number of crowdsourced labels', 
                     'AUC',
                     "/auc", False, ylim=[0.55, 1.0])
    
#     # Differences in AUC
#     plot_performance(Nreps_iter, weak_proportions, 'proportions', cluster_spreads, "auc.npy",
#                      "AUC ROC Improvement of HeatmapBCC",
#                      'Number of crowdsourced labels', 
#                      'AUC',
#                      "/auc_diffs", True)
    
    # Accuracy
    plot_performance(Nreps_iter, weak_proportions, 'proportions', cluster_spreads, "acc.npy",
                     "Classification Accuracy",
                     'Number of crowdsourced labels', 
                     'Fraction of Points Correctly Classified',
                     "/acc", False)
    
    methods_to_skip.append('NN')
    methods_to_skip.append('MV') # don't plot these    
     
    # Mean cross entropy
    plot_performance(Nreps_iter, weak_proportions, 'proportions', cluster_spreads, "mce.npy",
                     "Cross Entropy Classification Error",
                     'Number of crowdsourced labels', 
                     'Cross Entropy (bits)',
                     "/mce_discrete", False, nats_to_bits=nats_to_bits)
             
#     # Mean cross entropy difference
#     plot_performance(Nreps_iter, weak_proportions, 'proportions', cluster_spreads, "mce.npy",
#                      "Improvement of HeatmapBCC in \n Cross Entropy Error",
#                      'Number of crowdsourced labels', 
#                      'Cross Entropy (bits)',
#                      "/mce_discrete_diffs", True, nats_to_bits=nats_to_bit)

    methods_to_skip = []
                    
    # PLOT PROPORTIONS -------------------------------------------------------------------------------------------------
    # load results for the density estimation        
#     # Root mean squared error
#     plot_performance(Nreps_iter, weak_proportions, 'nreps', cluster_spreads, "rmsed.npy", 
#                      'Root Mean Square Error of State Probabilities', 
#                      'Proportion of Reliable Reporters', 
#                      'RMSE', 
#                      '/rmsed_p')

#     # AUC
#     plot_performance(Nreps_iter, weak_proportions, 'nreps', cluster_spreads, "auc.npy", 
#                      "ROC AUC", 
#                      'Proportion of Reliable Reporters', 
#                      'AUC', 
#                      '/auc_p')
# 
#     plot_performance(Nreps_iter, weak_proportions, 'nreps', cluster_spreads, "auc.npy", 
#                      "Improvement of HeatmapBCC in ROC AUC", 
#                      'Proportion of Reliable Reporters', 
#                      'AUC', 
#                      '/auc_p_diff', True)
#         
#     # KL divergence from gold density
#     plot_performance(Nreps_iter, weak_proportions, 'nreps', cluster_spreads, "kl.npy", 
#                      "KL-divergence", 
#                      'Proportion of Reliable Reporters', 
#                      'KL-divergence', 
#                      '/kl_p')
# 
#     plot_performance(Nreps_iter, weak_proportions, 'nreps', cluster_spreads, "kl.npy", 
#                      "Improvement of HeatmapBCC in KL-divergence", 
#                      'Proportion of Reliable Reporters', 
#                      'KL-divergence', 
#                      '/kl_p_diff')     

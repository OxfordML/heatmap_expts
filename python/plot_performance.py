'''
Plot the results of the simulations with increasing numbers of data points. Show accuracy etc. 

Currently using the results generated by gen_synthetic -- can alter to use the Ushahidi and Cicada data later.

Created on 23 Nov 2015

@author: edwin
'''

import numpy as np
import matplotlib.pyplot as plt
import gen_synthetic
import logging
from gen_synthetic import Nreports

def load_mean_results(methods, nruns, weak_proportions, filename):
    mean_results = {}
    for p_idx, p in enumerate(weak_proportions):
        for d in range(nruns):
            dataset_label = "p%i_d%i" % (p_idx, d)
            logging.info("Loading results for proportion %i, Dataset %d" % (p_idx, d))
            outputdir, _ = gen_synthetic.dataset_location(dataset_label)

            current_results = np.load(outputdir + filename).item()
            for m in methods:
                if not m in current_results:
                    m_rhs = m.lower() # see if this works using the lower case key instead
                    current_results[m] = np.array(current_results[m_rhs])
                    del current_results[m_rhs]
                else:
                    current_results[m] = np.array(current_results[m])

            if p not in mean_results:
                mean_results[p] = current_results                
            else:
                for m in methods:
                    if not m in current_results:
                        m_rhs = m.lower() # see if this works using the lower case key instead
                    else:
                        m_rhs = m
                    mean_results[p][m] += current_results[m_rhs]
        
        for m in mean_results[p]:
            mean_results[p][m] = mean_results[p][m] / float(nruns)
    return mean_results

if __name__ == '__main__':
    # import settings from where the experiments were run    
    nruns = gen_synthetic.nruns
    weak_proportions = gen_synthetic.weak_proportions
    Nreps_initial = gen_synthetic.Nreps_initial
    Nreports = gen_synthetic.Nreports 
    Nrep_inc = gen_synthetic.Nrep_inc
    Nsteps = gen_synthetic.nsteps
    methods = gen_synthetic.methods
    
    # get a set of x-coordinates for the number of reports at each iteration
    Nreps_iter = np.arange(Nsteps) * Nrep_inc + Nreps_initial
    
    # load results for the density estimation        
    # Root mean squared error
    rmsed = load_mean_results(methods, nruns, weak_proportions, "rmsed.npy")
    
    for p in weak_proportions:
        plt.figure()
        plt.title('Root Mean Square Error of Density Estimates')
        for m in methods:
            plt.plot(Nreps_iter, rmsed[p][m], label='%.2f reliable, %s' % (p, m))
        plt.xlabel('Number of crowdsourced labels')
        plt.ylabel('RMSE')
        plt.ylim(0, 1.0)
        plt.legend(loc='best')
    
    # Kendall's Tau
    tau = load_mean_results(methods, nruns, weak_proportions, "tau.npy")

    for p in weak_proportions:
        plt.figure()
        plt.title("Kendall's Tau for Density Estimates")
        for m in methods:
            plt.plot(Nreps_iter, tau[p][m], label='%.2f reliable, %s' % (p, m))
        plt.xlabel('Number of crowdsourced labels')
        plt.ylabel('tau')
        plt.ylim(-1.0, 1.0)
        plt.legend(loc='best')
    
    # Mean Cross Entropy
    mced = load_mean_results(methods, nruns, weak_proportions, "mced.npy")
    
    for p in weak_proportions:
        plt.figure()
        plt.title("Mean Cross Entropy of Density Estimates")
        for m in methods:
            plt.plot(Nreps_iter, mced[p][m], label='%.2f reliable, %s' % (p, m))
        plt.xlabel('Number of crowdsourced labels')
        plt.ylabel('MCE')
        plt.legend(loc='best')    
    
    # load results for predicting individual data points
    # Brier score
    rmse = load_mean_results(methods, nruns, weak_proportions, "rmse.npy")   
    for p in weak_proportions:
        plt.figure()
        plt.title("Brier Score on Individual Data Points")
        for m in methods:
            plt.plot(Nreps_iter, rmse[p][m], label='%.2f reliable, %s' % (p, m))
        plt.xlabel('Number of crowdsourced labels')
        plt.ylabel('Brier Score')
        plt.ylim(0, 1.0)
        plt.legend(loc='best')    
        
    # AUC
    auc = load_mean_results(methods, nruns, weak_proportions, "auc.npy")
    for p in weak_proportions:
        plt.figure()
        plt.title("AUC when Predicting Individual Data Points")
        for m in methods:
            plt.plot(Nreps_iter, auc[p][m], label='%.2f reliable, %s' % (p, m))
        plt.xlabel('Number of crowdsourced labels')
        plt.ylabel('AUC')
        plt.ylim(0, 1.0)
        plt.legend(loc='best')    
    
    # Mean cross entropy
    mce = load_mean_results(methods, nruns, weak_proportions, "mce.npy")
    for p in weak_proportions:
        plt.figure()
        plt.title("Mean Cross Entropy of Individual Data Points")
        for m in methods:
            plt.plot(Nreps_iter, mce[p][m], label='%.2f reliable, %s' % (p, m))
        plt.xlabel('Number of crowdsourced labels')
        plt.ylabel('MCE')
        plt.legend(loc='best')    
    
    # Variances within a single dataset
    #rmse_var = load_mean_results(nruns, weak_proportions, "rmse_var.npy")
    #auc_var = load_mean_results(nruns, weak_proportions, "auc_var.npy")
    #mce_var = load_mean_results(nruns, weak_proportions, "mce_var.npy")    
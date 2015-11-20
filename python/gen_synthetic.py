"""
Generate synthetic data for testing the heatmap methods. Run two experiments. For each experiment, generate 20
datasets.

1. VARYING SPARSENESS: Generate noisy reports with a mixture of reliabilities. These should be uniformly distributed through the space.
When we call prediction_tests, we will evaluate with decreasing sparseness -- this is already covered by experiment 2
because we test with increasing numbers of labels. However, the sparseness could be due to the spacing of the
reports and the total number of reports from each worker, both of which will affect the
 reliability model of the workers and the GP. So it may still be interesting to test these effects separately.

2. VARYING NOISE: Generate datasets with a fixed sparseness, but ~20 different noise levels. 
  Initialise with perfect reporters. Then
  perturb by increasing probability of random reports. Pi for all reporters is drawn from a beta distribution with
  different parameters in each case. Initially, strong diagonals (almost always perfect classifiers). Subtract from
  diagonals/add pseudo-counts to off-diagonals to increase probability of random classifiers. Need to find an easily-
  interpretable dependent variable: Mean reliability of classifier - think this should be equivalent to expected
  overall accuracy of reports.
  
3. LENGTH SCALE COLLAPSE/STABILITY: what happens when data is heteroskedastic? 
Large scale interpolation needed but in some areas we get confident, alternating reports. Do the alternating reports
cause length scale collapse leading to no interpolation? Does the use of nu0 < 1 affect this? 
Should we show potential problems when t itself is treated as a smooth function? Reason for difference: observations
of rho are very noisy and lowering length scale doesn't lead to extreme values of rho. In fact, keeping length scale higher
means more extreme values of rho where pseudo counts are shared.

4. EFFECT OF UPDATED REPORTS FROM ONE PERSON on a model that assumes all t and c are independent.
  
Plots -- accuracy, heatmaps, accuracy of confusion matrix?  

"""

import sys

sys.path.append("/homes/49/edwin/robots_code/HeatMapBCC/python")
sys.path.append("/homes/49/edwin/robots_code/pyIBCC/python")

from scipy.stats import beta, bernoulli, multivariate_normal as mvn
import prediction_tests
import numpy as np
import logging
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

RESET_ALL_DATA = True

# FUNCTIONS -------------------------------------------------------------------------------------------------------
def dataset_location(dataset_label):
    # ID of the data set
    outputdir = "/homes/49/edwin/robots_code/heatmap_expts/data/output/synth_sparse_%s/" % (dataset_label)
    logging.debug("Using output directory %s" % outputdir)
    
    data_outputdir = outputdir + '/synth_data/'
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    if not os.path.isdir(data_outputdir):
        os.mkdir(data_outputdir)
        
    return outputdir, data_outputdir    

def gen_synth_ground_truth(reset_all_data, Nreports, ls, dataset_label):
    '''
    Generate data.
    '''
    _, data_outputdir = dataset_location(dataset_label)
    
    if not reset_all_data and os.path.isfile(data_outputdir + "x_all.npy"):
        xreports = np.load(data_outputdir + "x_all.npy")[:Nreports]
        yreports = np.load(data_outputdir + "y_all.npy")[:Nreports]
        t_gold = np.load(data_outputdir + "t_all.npy")[:Nreports]
        return xreports, yreports, t_gold
        
    # select random points in the grid to generate reports
    xreports = np.random.rand(Nreports, 1) * float(nx)
    yreports = np.random.rand(Nreports, 1) * float(ny)
    
    # select random points in the grid to test
    xtest = np.random.rand(Ntest, 1) * float(nx)
    ytest = np.random.rand(Ntest, 1) * float(ny)
        
    x_all = np.concatenate((xreports, xtest))
    y_all = np.concatenate((yreports, ytest))
        
    ddx = x_all - x_all.T
    ddy = y_all - y_all.T
    
    K = sq_exp_cov(ddx, ddy, ls) 
    # scale K toward more extreme probabilities
    output_scale = logit(0.9)
    f_mean = np.zeros(len(x_all))
    f_all = mvn.rvs(mean=f_mean, cov=K * output_scale)
    
    x_plot = np.arange(0, nx, 1)
    y_plot = np.arange(0, ny, 1)
    x_plot, y_plot = np.meshgrid(x_plot, y_plot)
    z_plot = np.zeros(x_plot.shape)
    for i in range(z_plot.shape[0]):
        for j in range(z_plot.shape[1]):
            z_plot[i, j] = f_all[ np.argmin(np.abs(x_all[:,0] - x_plot[i,j])**2 + np.abs(y_all[:,0] - y_plot[i,j])**2 ) ]
            z_plot[i, j] = sigmoid(z_plot[i, j]) 
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_plot, y_plot, z_plot, cmap="Spectral", rstride=1, cstride=1)
    plt.title('ground truth density function')    
    
    f_rep = f_all[:Nreports]
    
    rho_rep = sigmoid(f_rep) # class density at report locations
    
    # generate ground truth for the report locations
    t_gold = bernoulli.rvs(rho_rep)

    # generate ground truth for the test locations
    f_test = f_all[Nreports:]
    rho_test = sigmoid(f_test)
    t_test_gold = bernoulli.rvs(rho_test).reshape(-1)

    # Save test data to file    
    np.save(data_outputdir + "x_all.npy", x_all)
    np.save(data_outputdir + "y_all.npy", y_all)
    np.save(data_outputdir + "f_all.npy", f_all)
    np.save(data_outputdir + "t_all.npy", np.concatenate((t_gold, t_test_gold)) )# t_all
    
    return xreports, yreports, t_gold
    
def gen_synth_reports(reset_all_data, Nreports, a, b, xreports, yreports, t_gold, dataset_label):
    
    _, data_outputdir = dataset_location(dataset_label)
    
    if not reset_all_data and os.path.isfile(data_outputdir + "C.npy"):
        return
    
    alpha0 = np.zeros((J, 2, S))
    for s in range(S):
        alpha0 += b[s]
        alpha0[np.arange(J), np.arange(J), s] = a[s]    
    
    #generate confusion matrices
    pi = np.zeros((J, 2, S))
    for j in range(J):
        pi[j, 0, :] = beta.rvs(alpha0[j, 0, 0], alpha0[j, 1, 0], size=S)
        pi[j, 1, :] = 1 - pi[j, 0, :]
    
    # generate reports -- get the correct bit of the conf matrix
    pi_reps = pi[:, 1, :].reshape(J, S)
    reporter_ids = np.random.randint(0, S, (Nreports, 1))
    pi_reps = pi_reps[t_gold[:, np.newaxis], reporter_ids]
    reports = bernoulli.rvs(pi_reps)
    report_loc_idxs = np.arange(len(t_gold))
    C = np.concatenate((reporter_ids, xreports[report_loc_idxs], yreports[report_loc_idxs], reports), axis=1)
    
    np.save(data_outputdir + "C.npy", C)
    np.save(data_outputdir + "pi_all.npy", pi.swapaxes(0,1).reshape((J*2, S), order='F').T )# pi all. Flattened so     
        
    #plt.scatter(xtest[sidxs, 0], t_test_gold[sidxs], color='b')
#     
#     # KNN estimates for plotting purposes
#     windowmean = np.zeros(len(xreports))
#     for i in range(len(xreports)):
#         start = i - 30
#         if start < 0:
#             start = 0
#         finish = i + 30
#         if finish > len(xreports):
#             finish = len(xreports)
#         windowsize = finish - start
#         windowmean[i] = np.sum(reports[sidxs][start:finish]) / float(windowsize)
#     #plt.plot(xreports[sidxs], windowmean, color='y', label='KNN reports')
#    
#    # PLOT RESULTS -----------------------------------------------------------------------------------------------
#     
#     sidxs = np.argsort(xtest[:, 0])
#     if 'HeatmapBCC' in methods:
#         plt.plot(xtest[sidxs], results[Nreps_initial]['heatmapbcc'][0][sidxs], color='magenta', label='BCC predictions')
#         plt.plot(xtest[sidxs], densityresults[Nreps_initial]['heatmapbcc'][0][sidxs], color='cyan', label='BCC density estimation')
#     if 'KDE' in methods:
#         plt.plot(xtest[sidxs], results[Nreps_initial]['KDE'][0][sidxs], color='darkgreen', label='KDE')
#     if "GP" in methods:
#         plt.plot(xtest[sidxs], results[Nreps_initial]['Train_GP_on_Freq'][0][sidxs], color='brown', label='GP')
#     if "IBCC+GP" in methods:
#         plt.plot(xtest[sidxs], results[Nreps_initial]['IBCC_then_GP'][0][sidxs], color='black', label='IBCC+GP')
#     
#     plt.legend(loc='best')

def sq_exp_cov(xvals, yvals, ls):
    Kx = np.exp( -xvals**2 / ls[0] )
    Ky = np.exp( -yvals**2 / ls[1] )
    K = Kx * Ky
    return K

def sigmoid(f):
    g = 1/(1+np.exp(-f))
    return g

def logit(g):
    f = -np.log(1/g - 1)
    return f

"""
 TODO:
 
 5. Calculate mean + variance of statistics across multiple runs. 
 6. Plotting across all runs and all iterations -- option to turn off plotting for individual run/iteration.
 7. Log likelihood measure.
 8. Save plots so we can run on florence.
"""

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

# EXPERIMENT CONFIG ---------------------------------------------------------------------

methods = [
           'KDE',
           'GP',
           'IBCC+GP',
           'HeatmapBCC'
           ]

# GROUND TRUTH
nx = 20.0
ny = 20.0
J = 2 # number of classes

Ntest = 100 # no. points in the grid to test

ls = nx #np.random.randint(1, nx * 2, 1)
ls = np.zeros(2) + ls
print "length scale: "
print ls

nu0 = np.ones(J) # generate ground truth for the report locations
z0 = nu0[1] / np.sum(nu0)

# # VARYING NOISE TESTS -------------------------------------------------------------------------------------------
# This will also test sparseness so we don't need to run the algorithms again to show the effect of sparseness,
# just make different plots!

# Number of datasets
nruns = 1#25
nsteps = 8

# REPORTS
Nreports = 400 # total number of reports in complete data set
Nreps_initial = 50 # number of labels in first iteration data set. 
Nrep_inc = (Nreports - Nreps_initial) / (nsteps - 1) # increment the number of labels at each iteration    
logging.info('Incrementing number of reports by %i in each iteration.' % Nrep_inc)

# REPORTERS
S = 5 # number of reporters

a_reliable = 9999999999
b_reliable = 1

a_weak = 5.5
b_weak = 4.5

nproportions = 6.0 
if nproportions > S:
    nproportions = S
weak_proportions = np.arange(nproportions + 1) / float(nproportions)

for d in range(nruns):
    for p_idx, p in enumerate(weak_proportions):
    
        a = np.ones(S)
        a[:S * p] = a_reliable
        a[S * p:] = a_weak
        b = np.ones(S)
        b[:S*p] = b_reliable
        b[S*p:] = b_weak
    
        dataset_label = "d%i" % d
        logging.info("Generating data/reloading old data for proportion %i, Dataset %d" % (p_idx, d))
        xreports, yreports, t_gold = gen_synth_ground_truth(RESET_ALL_DATA & (p_idx==0), Nreports, ls, dataset_label) # only reset on the first iteration
        dataset_label = "p%i_d%i" % (p_idx, d)
        gen_synth_reports(RESET_ALL_DATA, Nreports, a, b, xreports, yreports, t_gold, dataset_label) # only reset on the first iteration

# RUN TESTS -----------------------------------------------------------------------------------------------------------
for p_idx, p in enumerate(weak_proportions):
    
    a = np.ones(S)
    a[:S * p] = a_reliable
    a[S * p:] = a_weak
    b = np.ones(S)
    b[:S*p] = b_reliable
    b[S*p:] = b_weak
    
    for d in range(nruns):
        dataset_label = "d%i" % d
        logging.info("Running tests for proportion %i, Dataset %d" % (p_idx, d))
        
        outputdir, data_outputdir = dataset_location(dataset_label)    
        x_all = np.load(data_outputdir + "x_all.npy")
        xtest = x_all[Nreports:]
        y_all = np.load(data_outputdir + "y_all.npy")
        ytest = y_all[Nreports:]
        f_all = np.load(data_outputdir + "f_all.npy")
        f_test = f_all[Nreports:]
        rho_test = sigmoid(f_test)
        t_all = np.load(data_outputdir + "t_all.npy" )# t_all
        t_test_gold = t_all[Nreports:]
        
        dataset_label = "p%i_d%i" % (p_idx, d)
        outputdir, data_outputdir = dataset_location(dataset_label) 
        C = np.load(data_outputdir + "C.npy")
        
        alpha0 = np.zeros((J, 2, S))
        for s in range(S):
            alpha0 += b[s]
            alpha0[np.arange(J), np.arange(J), s] = a[s]    
            
        # Run the tests with the current data set
        tester = prediction_tests.Tester(outputdir, methods, Nreports, z0, alpha0, nu0)            
        tester.run_tests(C, nx, ny, xtest.reshape(-1), ytest.reshape(-1), t_test_gold, rho_test, Nreps_initial, Nrep_inc)
        tester.save_self()
 
# # SPARSENESS TESTS ----------------------------------------------------------------------------------------------------
# 
# # Doesn't this come from running the previous experiments anyway? Taking one value of p and looking at the performance
# # as the labels come in... 
# # Number of datasets
# nruns = 20
# nsteps = 10
# 
# # REPORTS
# Nreports = 1000 # total number of reports in complete data set
# Nreps_initial = 100 # number of labels in first iteration data set
# Nrep_inc = (Nreports - Nreps_initial) / (nsteps - 1) # increment the number of labels at each iteration    
# 
# # REPORTERS
# S = 1 # number of reporters
# 
# a = 1000000
# b = 1
# 
# for d in range(nruns):
#     logging.info("Dataset %d" % d)
#     run_synth_test(Nreports, a, b, d) 

# TRUSTED REPORTER TESTS -----------------------------------------------------------------------------------------------
# Show what happens when we supply labels from a few highly reliable workers, the changes should propagate. Can start 
# with noisy labels only, then show what happens when trusted reports are introduced.

"""
Generate synthetic data for testing the heatmap methods. Run two experiments. For each experiment, generate 20
datasets.

1. VARYING SPARSENESS: Generate noisy reports with a mixture of reliabilities. These should be uniformly distributed through the space.
When we call prediction_tests, we will evaluate with decreasing sparseness.

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

import matplotlib.pyplot as plt

def sq_exp_cov(xvals, yvals, ls):
    Kx = np.exp( -xvals**2 / ls[0] )
    Ky = np.exp( -yvals**2 / ls[1] )
    K = Kx * Ky
    return K

def sigmoid(f):
    g = 1/(1+np.exp(-f))
    return g

# SPARSENESS TESTS ----------------------------------------------------------------------------------------------------

"""
 TODO:
 
 1. Ensure we can do a single run through.
 3. Save test data to file.
 4. Ensure result data is saved correctly.
 5. Ensure result stats are saved correctly.
 6. Move settings to top, away from the generation so we can see the set up.
 7. Multiple repeats with different datsets.
"""

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

# EXPERIMENT CONFIG

methods = [
           'KDE',
           'GP',
           'IBCC+GP',
           'HeatmapBCC'
           ]

# GROUND TRUTH
nx = 100.0
ny = 1.0
J = 2 # number of classes

Ntest = 100 # no. points in the grid to test

ls = np.array([np.sqrt(nx) * 2, 100000])
print "length scale: "
print ls

nu0 = np.ones(J) # generate ground truth for the report locations
z0 = nu0[1] / np.sum(nu0)

# REPORTS
Nreports = 1000 # total number of reports in complete data set
Ninitial_labels = Nreports #Nreports / 10 # number of labels in first iteration data set
Nlabel_increment = Nreports / 10 # increment the number of labels at each iteration

# REPORTERS
S = 1 # number of reporters

a = 1000000
b = 1
alpha0 = np.zeros((J, 2)) + b
alpha0[np.arange(J), np.arange(J)] = a 
alpha0 = alpha0[:,:,np.newaxis]

# ID of the data set
dataset_idx = 0
outputdir = "/homes/49/edwin/robots_code/heatmap_expts/data/output/synth_sparse_%d/" % dataset_idx

#generate confusion matrices
pi = np.zeros((J, 2, S))
for j in range(J):
    pi[j, 0, :] = beta.rvs(alpha0[j, 0, 0], alpha0[j, 1, 0], size=S)
    pi[j, 1, :] = 1 - pi[j, 0, :]

# select random points in the grid to generate reports
xreports = np.random.rand(Nreports, 1) * nx
yreports = np.random.rand(Nreports, 1) * ny

# select random points in the grid to test
xtest = np.random.rand(Ntest, 1) * nx
ytest = np.random.rand(Ntest, 1) * ny

x_all = np.concatenate((xreports, xtest))
y_all = np.concatenate((yreports, ytest))

ddx = x_all - x_all.T
ddy = y_all - y_all.T

K = sq_exp_cov(ddx, ddy, ls) / 5.0 # scale it toward more extreme probabilities
f_all = mvn.rvs(np.zeros(len(x_all)), K)

f_rep = f_all[:Nreports]

rho_rep = sigmoid(f_rep) # class density at report locations

# generate ground truth for the report locations
t_gold = bernoulli.rvs(rho_rep)

# generate reports -- get the correct bit of the conf matrix
pi = pi[:, 1, :].reshape(J, S)
reporter_ids = np.random.randint(0, S, (Nreports, 1))
pi = pi[t_gold[:, np.newaxis], reporter_ids]
reports = bernoulli.rvs(pi)
report_loc_idxs = np.arange(len(t_gold))
C = np.concatenate((reporter_ids, xreports[report_loc_idxs], yreports[report_loc_idxs], reports), axis=1)

# generate ground truth for the test locations
f_test = f_all[Nreports:]
rho_test = sigmoid(f_test)
t_test_gold = bernoulli.rvs(rho_test).reshape(-1)

# plot the ground truth
plt.figure()
sidxs = np.argsort(x_all[:, 0])
plt.plot(x_all[sidxs, 0], np.concatenate((rho_rep, rho_test))[sidxs], color='r', label='Ground truth density')
#plt.scatter(xtest[sidxs, 0], t_test_gold[sidxs], color='b')
sidxs = np.argsort(xreports[:, 0])
plt.scatter(xreports[sidxs, 0], reports[sidxs], color='b', label='Reports')

windowmean = np.zeros(len(xreports))
for i in range(len(xreports)):
    start = i - 30
    if start < 0:
        start = 0
    finish = i + 30
    if finish > len(xreports):
        finish = len(xreports)
    windowsize = finish - start
    windowmean[i] = np.sum(reports[sidxs][start:finish]) / float(windowsize)
plt.plot(xreports[sidxs], windowmean, color='y', label='KNN reports')

clusteridxs_all = np.zeros(S, dtype=int)

# Run the tests with the current data set
results, densityresults, heatmapcombiner, gpgrid, gpgrid2, ibcc_combiner = prediction_tests.run_tests(S, C, nx, ny, z0, alpha0,
                                        clusteridxs_all, alpha0, nu0, ls, [t_test_gold], [xtest.reshape(-1)], [ytest.reshape(-1)],
                                        rho_test, Nreports, Ninitial_labels, Nlabel_increment, outputdir, methods)
sidxs = np.argsort(xtest[:, 0])
if 'HeatmapBCC' in methods:
    plt.plot(xtest[sidxs], results[Ninitial_labels]['heatmapbcc'][0][sidxs], color='magenta', label='BCC predictions')
    plt.plot(xtest[sidxs], densityresults[Ninitial_labels]['heatmapbcc'][0][sidxs], color='cyan', label='BCC density estimation')
if 'KDE' in methods:
    plt.plot(xtest[sidxs], results[Ninitial_labels]['KDE'][0][sidxs], color='darkgreen', label='KDE')
if "GP" in methods:
    plt.plot(xtest[sidxs], results[Ninitial_labels]['Train_GP_on_Freq'][0][sidxs], color='brown', label='gp')
if "IBCC+GP" in methods:
    plt.plot(xtest[sidxs], results[Ninitial_labels]['IBCC_then_GP'][0][sidxs], color='black', label='IBCC+GP')

plt.legend(loc='best')

# # VARYING NOISE TESTS -------------------------------------------------------------------------------------------
# reliability_test_density = 0.1 # reports used in the test with varying reporter reliability
# Nrel = int(nx*ny*reliability_test_density)

"""
 TODO: Wait till the sparseness test is done then alter it. 
"""

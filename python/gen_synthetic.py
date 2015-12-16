"""
Generate synthetic data for testing the heatmap methods. Run two experiments. For each experiment, generate 20
datasets.

1. VARYING SPARSENESS: Generate noisy reports with diags mixture of reliabilities. These should be uniformly distributed through the space.
When we call prediction_tests, we will evaluate with decreasing sparseness -- this is already covered by experiment 2
because we test with increasing numbers of labels. However, the sparseness could be due to the spacing of the
reports and the total number of reports from each worker, both of which will affect the
 reliability model of the workers and the GP. So it may still be interesting to test these effects separately.

2. VARYING NOISE: Generate datasets with diags fixed sparseness, but ~20 different noise levels. 
  Initialise with perfect reporters. Then
  perturb by increasing probability of random reports. Pi for all reporters is drawn from diags beta distribution with
  different parameters in each case. Initially, strong diagonals (almost always perfect classifiers). Subtract from
  diagonals/add pseudo-counts to off-diagonals to increase probability of random classifiers. Need to find an easily-
  interpretable dependent variable: Mean reliability of classifier - think this should be equivalent to expected
  overall accuracy of reports.
  
3. LENGTH SCALE COLLAPSE/STABILITY: what happens when data is heteroskedastic? 
Large scale interpolation needed but in some areas we get confident, alternating reports. Do the alternating reports
cause length scale collapse leading to no interpolation? Does the use of nu0 < 1 affect this? 
Should we show potential problems when t itself is treated as diags smooth function? Reason for difference: observations
of rho are very noisy and lowering length scale doesn't lead to extreme values of rho. In fact, keeping length scale higher
means more extreme values of rho where pseudo counts are shared. 
 
4. EFFECT OF UPDATED REPORTS FROM ONE PERSON on diags model that assumes all t and biases are independent.
  
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
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D

RESET_ALL_DATA = False
PLOT_SYNTH_DATA = False
SAVE_RESULTS = True

# FUNCTIONS -------------------------------------------------------------------------------------------------------
    
def run_experiments():
    for cluster_spread in cluster_spreads:
        
        experiment_label = expt_label_template % cluster_spread 
        
        for d in range(nruns):
            for p_idx, p in enumerate(weak_proportions):
            
                diags = np.ones(S)
                diags[:S * p] = diag_reliable
                diags[S * p:] = diag_weak
                off_diags = np.ones(S)
                off_diags[:S * p] = off_diag_reliable
                off_diags[S * p:] = off_diag_weak
                biases = np.ones((S, J))
                biases[:S * p, :] = bias_reliable
                biases[S * p:, :] = bias_weak
            
                dataset_label = "d%i" % d
                logging.info("Generating data/reloading old data for proportion %i, Dataset %d" % (p_idx, d))
                xreports, yreports, t_gold = gen_synth_ground_truth(RESET_ALL_DATA & (p_idx==0), Nreports, Ntest, ls, 
                    experiment_label, dataset_label, 5, cluster_spread * nx / Nreports**0.5) # only reset on the first iteration
                dataset_label = "p%i_d%i" % (p_idx, d)
                gen_synth_reports(RESET_ALL_DATA, Nreports, diags, off_diags, biases, xreports, yreports, t_gold, 
                    experiment_label, dataset_label)
        
        # RUN TESTS -----------------------------------------------------------------------------------------------------------
        for p_idx, p in enumerate(weak_proportions):
            
            diags = np.ones(S)
            diags[:S * p] = diag_reliable
            diags[S * p:] = diag_weak
            off_diags = np.ones(S)
            off_diags[:S*p] = off_diag_reliable
            off_diags[S*p:] = off_diag_weak
            
            for d in range(nruns):
                dataset_label = "d%i" % d
                logging.info("Running tests for proportion %i, Dataset %d" % (p_idx, d))
                
                outputdir, data_outputdir = dataset_location(experiment_label, dataset_label)    
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
                outputdir, data_outputdir = dataset_location(experiment_label, dataset_label) 
                C = np.load(data_outputdir + "C.npy") 
                
                alpha0 = np.ones((J, 2, S)) + 10
                for s in range(S):
                    alpha0[np.arange(J), np.arange(J), s] += 1
                
                # Run the tests with the current data set
                tester = prediction_tests.Tester(outputdir, methods, Nreports, z0, alpha0, nu0, ls[0], optimise=False, 
                                                 verbose=False)            
                tester.run_tests(C, nx, ny, xtest.reshape(-1), ytest.reshape(-1), t_test_gold, rho_test, Nreps_initial, 
                                 Nrep_inc)
                if SAVE_RESULTS:
                    tester.save_separate_results()
        
    return tester

def dataset_location(experiment_label, dataset_label):
    # ID of the data set
    #output 5 was most working one so far
    outputdir = "./data/" + experiment_label + '/'
    
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    
    outputdir = outputdir + "synth_sparse_%s/" % (dataset_label)
    logging.debug("Using output directory %s" % outputdir)
    
    data_outputdir = outputdir + '/synth_data/'
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    if not os.path.isdir(data_outputdir):
        os.mkdir(data_outputdir)
        
    return outputdir, data_outputdir    

def plot_density(nx, ny, x_all, y_all, f_all, title='ground truth density function', apply_sigmoid=True, ax=None, 
                 transparency=0.0):

    cmap = plt.get_cmap('Spectral')                
    cmap._init()
    cmap._lut[:,-1] = np.linspace(1-transparency, 1-transparency, cmap.N+3)   

    xi = np.linspace(min(x_all), max(x_all))
    yi = np.linspace(min(y_all), max(y_all))
    x_plot, y_plot = np.meshgrid(xi, yi)
    if apply_sigmoid:
        f_all = sigmoid(f_all)
    z_plot = griddata(x_all.reshape(-1), y_all.reshape(-1), f_all, xi, yi)
    
    if not ax:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    ax.plot_surface(x_plot, y_plot, z_plot, cmap="Spectral", rstride=1, cstride=1)
    ax.set_zlim3d(0, 1)
    plt.title(title)
    
def plot_report_histogram(reportsx, reportsy, reports, ax=None):
    if not ax:
        fig = plt.figure()
        ax = fig.gca(projection='3d')    
    weights = reports.copy()
    weights[weights==0] = -1
    hist, xedges, yedges = np.histogram2d(reportsx.flatten(), reportsy.flatten(), weights=weights.flatten())
    x_plot, y_plot = np.meshgrid(xedges[:-1], yedges[:-1])
    ax.plot_surface(x_plot, y_plot, hist, cmap='Spectral', rstride=1, cstride=1)

def gen_synth_ground_truth(reset_all_data, Nreports, Ntest, ls, experiment_label, dataset_label, 
                           mean_reps_per_cluster=1, clusterspread=0):
    '''
    Generate data.
    '''
    _, data_outputdir = dataset_location(experiment_label, dataset_label)
    
    if not reset_all_data and os.path.isfile(data_outputdir + "x_all.npy"):
        xreports = np.load(data_outputdir + "x_all.npy")[:Nreports]
        yreports = np.load(data_outputdir + "y_all.npy")[:Nreports]
        t_gold = np.load(data_outputdir + "t_all.npy")[:Nreports]
        return xreports, yreports, t_gold
        
    #reps per cluster controls how concentrated the reports are in an group. mean_reps_per_cluster==1 puts all reports at
    # separate locations.
    nclusters = Nreports / float(mean_reps_per_cluster)
    xclusters = np.random.rand(nclusters, 1) * (float(nx) - clusterspread)
    yclusters = np.random.rand(nclusters, 1) * (float(ny) - clusterspread)    
        
    # assign reports to clusters evenly
    rep_cluster_idxs = np.tile(np.arange(nclusters, dtype=int)[:, np.newaxis], (np.ceil(Nreports/nclusters), 1))
    rep_cluster_idxs = rep_cluster_idxs[:Nreports].flatten()
    
    # select random points in the grid to generate reports
    xreports = xclusters[rep_cluster_idxs] + np.random.rand(Nreports, 1) * float(clusterspread)    
    yreports = yclusters[rep_cluster_idxs] + np.random.rand(Nreports, 1) * float(clusterspread)
    
    # select random points in the grid to test
    #xtest = np.random.rand(Ntest, 1) * float(nx)
    #ytest = np.random.rand(Ntest, 1) * float(ny)
    
#     # select specific report points -- diagonal line
#     Nreports = np.round(Nreports **0.5)
#     xreports = (np.arange(Nreports) * float(nx) / Nreports)[np.newaxis, :]
#     xreports = np.tile(xreports, (Nreports, 1))
#     xreports = xreports.reshape(Nreports**2, 1)
#     yreports = (np.arange(Nreports) * float(ny) / Nreports)[:, np.newaxis]
#     yreports = np.tile(yreports, (1, Nreports))
#     yreports = yreports.reshape(Nreports**2, 1)
#     Nreports = Nreports ** 2

    # Select diags grid of test points Ntest x Ntest
    Ntest = np.round(Ntest **0.5)
    xtest = (np.arange(Ntest) * float(nx) / Ntest)[np.newaxis, :]
    xtest = np.tile(xtest, (Ntest, 1))
    xtest = xtest.reshape(Ntest**2, 1)
    ytest = (np.arange(Ntest) * float(ny) / Ntest)[:, np.newaxis]
    ytest = np.tile(ytest, (1, Ntest))
    ytest = ytest.reshape(Ntest**2, 1)
        
    x_all = np.concatenate((xreports, xtest))
    y_all = np.concatenate((yreports, ytest))
        
    ddx = x_all - x_all.T
    ddy = y_all - y_all.T
    
    K = sq_exp_cov(ddx, ddy, ls) 
    f_mean = np.zeros(len(x_all))
    f_all = mvn.rvs(mean=f_mean, cov=K / output_scale)
    
    if PLOT_SYNTH_DATA:
        plot_density(nx, ny, x_all, y_all, f_all)
    
    f_rep = f_all[:Nreports]
    
    rho_rep = sigmoid(f_rep) # class density at report locations
    
    if PLOT_SYNTH_DATA:
        plot_density(nx, ny, xreports, yreports, rho_rep, "\rho at report locations", False)
    
    # generate ground truth for the report locations
    t_gold = bernoulli.rvs(rho_rep)

    print "Fraction of positive training target points: %.3f" % (np.sum(t_gold) / float(len(t_gold)))
    
    # generate ground truth for the test locations
    f_test = f_all[Nreports:]
    rho_test = sigmoid(f_test)
    t_test_gold = bernoulli.rvs(rho_test).reshape(-1)
    print "Fraction of positive test target points: %.3f" % (np.sum(t_test_gold) / float(len(t_test_gold)))

    # Save test data to file    
    np.save(data_outputdir + "x_all.npy", x_all)
    np.save(data_outputdir + "y_all.npy", y_all)
    np.save(data_outputdir + "f_all.npy", f_all)
    np.save(data_outputdir + "t_all.npy", np.concatenate((t_gold, t_test_gold)) )# t_all
    
    return xreports, yreports, t_gold
    
def gen_synth_reports(reset_all_data, Nreports, diags, off_diags, bias_vector, xreports, yreports, t_gold, 
                      experiment_label, dataset_label):
    
    _, data_outputdir = dataset_location(experiment_label, dataset_label)
    
    if not reset_all_data and os.path.isfile(data_outputdir + "C.npy"):
        return
    
    alpha0 = np.zeros((J, 2, S))
    for s in range(S):
        alpha0[:, :, s] += off_diags[s]
        alpha0[range(J), range(J), s] = diags[s]
        alpha0[:, :, s] += bias_vector[s:s+1, :] # adds diags fixed number of counts to each row so it biases toward
        #same answer regardless of ground truth
    
    #generate confusion matrices
    pi = np.zeros((J, 2, S))
    for s in range(S):
        for j in range(J):
            pi[j, 0, s] = beta.rvs(alpha0[j, 0, s], alpha0[j, 1, s])
            pi[j, 1, s] = 1 - pi[j, 0, s]
        print "Confusion matrix for worker %i: %s" % (s, str(pi[:, :, s]))
    # generate reports -- get the correct bit of the conf matrix
    pi_reps = pi[:, 1, :].reshape(J, S)
    reporter_ids = np.random.randint(0, S, (Nreports, 1))
    
    if PLOT_SYNTH_DATA:
        hist, bins = np.histogram(reporter_ids, S)
        plt.figure()
        plt.bar(bins[:-1], hist)
        plt.xlabel("Reporter IDs")
        plt.ylabel("Number of reports from each reporter")
    
    pi_reps = pi_reps[t_gold[:, np.newaxis], reporter_ids]
    reports = bernoulli.rvs(pi_reps)
    print "Fraction of positive reports: %.3f" % (np.sum(reports) / float(len(reports)))
    
    if PLOT_SYNTH_DATA:
        plot_report_histogram(xreports, yreports, reports)    
    
    C = np.concatenate((reporter_ids, xreports, yreports, reports), axis=1)
    
    np.save(data_outputdir + "C.npy", C)
    np.save(data_outputdir + "pi_all.npy", pi.swapaxes(0,1).reshape((J*2, S), order='F').T )# pi all. Flattened so     

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

# EXPERIMENT CONFIG ---------------------------------------------------------------------------------------------------

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

Ntest = nx*ny # no. points in the grid to test

ls = 40.0 #nx #np.random.randint(1, nx * 2, 1)
ls = np.zeros(2) + ls
print "length scale: "
print ls

# scale K toward more extreme probabilities
output_scale = 1.0 / logit(0.75)**2
print "Output scale for ground truth: " + str(output_scale)

nu0 = np.ones(J) # generate ground truth for the report locations
z0 = nu0[1] / np.sum(nu0)

# # VARYING NOISE TESTS -------------------------------------------------------------------------------------------
# This will also test sparseness so we don't need to run the algorithms again to show the effect of sparseness,
# just make different plots!

# Number of datasets
nruns = 25
nsteps = 4

# REPORTS
Nreports = 500 #400 # total number of reports in complete data set
Nreps_initial = 50#50 #50 # number of labels in first iteration data set. 
Nrep_inc = (Nreports - Nreps_initial) / (nsteps - 1) # increment the number of labels at each iteration    
logging.info('Incrementing number of reports by %i in each iteration.' % Nrep_inc)

# REPORTERS
S = 16 # number of reporters

diag_reliable = 5000.0
off_diag_reliable = 1.0
bias_reliable = np.zeros(J)

# For the unreliable workers to have white noise
#diag_weak = 5000.0
#off_diag_weak = 5000.0
#bias_weak = np.zeros(J)

# For the unreliable workers to have bias
diag_weak = 3.0
off_diag_weak = 1.0
bias_weak = np.zeros(J)
bias_weak[0] = 10.0

nproportions = 4
if nproportions > S:
    nproportions = S
weak_proportions = np.arange(1.0, nproportions+1.0)
weak_proportions /= nproportions

expt_label_template = 'output_cluslocs%.2f_bias_test1'

# values to be used in each experiment
cluster_spreads = [1.0, 0.5, 0.2] # spreads are multiplied by average distance between reports and a random 
#number between 0 and 1 with mean 0.5. So spread of 1 should give the default random placement of reports.

# MAIN SET OF SYNTHETIC DATA EXPERIMENTS ------------------------------------------------------------------------------
if __name__ == '__main__':
    run_experiments()
    
# CASE STUDY: INTERPOLATING BETWEEN GOOD WORKERS ----------------------------------------------------------------------
# Show an example where IBCC infers trusted workers, then interpolates between them, ignoring the reports from noisy 
# workers. This should show for each method:
# 1. GP will weight all reporters equally, so will have errors where the noisy reporters are.
# 2. IBCC grid + GP will not infer the reliability as accurately and will interpolate more poorly if there is a gradual
# change across a grid square.
# Setup: 
# 1. Requires some clustering to discriminate between worker reliabilities. --> pick the middle cluster spread setting
# 2. Two sets of workers, biased and good. Biased workers will pull GP in the same direction, regardless of the ground
# truth. --> pick weak proportion 0.5
# 3. Can test with lots of labels to show that this problem is not avoided by GP with lots of data. --> nReports==1000 
# 4. Plot ground truth (put red dots for biased workers?) and predictions from each method. Zoom in on an area where 
# the biased workers are clustered with no or very few good workers.
# 5. Table with MCE for each method.  
    
# CASE STUDY: TRUSTED REPORTER -----------------------------------------------------------------------------------------
# Show what happens when we supply labels from a highly reliable worker, the changes should propagate. Can start 
# with noisy labels only, then show what happens when trusted reports are introduced -- allows us to learn meaningful 
# confusion matrices even if there is a large amount of noise. This might be better shown with Ushahidi data. 
# This should show for each method:
# 1. GP stuck on very noisy/weak decisions.
# 2. IBCC grid + GP either requires very coarse grid, leading to poor interpolation, or very small grid so that fewer
# reporters coincide, hence fewer reporters detected as reliable.
# Set up: 
# 0. Test this with 20 reporters so that there are several good reporters to detect.
# 1. Take a case with 75% noisy workers where none of the methods do well.
# 2. Can use middle cluster spreading or no clustering.
# 3. nReports == 1000
# 4. Plot ground truth and initial predictions without trusted worker. Plot results when trusted worker is introduced.
# 5. Table for MCE for each method and for HeatmapBCC with and without the trusted worker.
# 6. May want to add a column to the table showing a comparison in MCE over a number of datasets, or a plot with 
# different numbers of trusted reports. 


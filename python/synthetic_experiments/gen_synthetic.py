"""
Generate synthetic data for testing the heatmap methods.
"""

from scipy.stats import beta, bernoulli, multivariate_normal as mvn
import prediction_tests
import numpy as np
import logging
import os

import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D

# HELPER FUNCTIONS -------------------------------------------------------------------------------------------------------

def matern_3_2(xvals, yvals, ls):
    Kx = np.abs(xvals) * 3**0.5 / ls[0]
    Kx = (1 + Kx) * np.exp(-Kx)
    Ky = np.abs(yvals) * 3**0.5 / ls[1]
    Ky = (1 + Ky) * np.exp(-Ky)
    K = Kx * Ky
    return K

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

# EXPERIMENT CONFIG ---------------------------------------------------------------------------------------------------

RESET_ALL_DATA = False
PLOT_SYNTH_DATA = False
SAVE_RESULTS = True

methods = [
           'MV',
           'NN',
           'SVM',
           'oneclassSVM',
           'KDE',
           'IBCC',
           'GP',
           'IBCC+GP',
           'HeatmapBCC'
           ]

# GROUND TRUTH
nx = 40.0#100.0  #20.0
ny = 40.0#100.0  #20.0
J = 2 # number of classes

Ntest = nx*ny # no. points in the grid to test

ls = 20.0 #40.0 #nx #np.random.randint(1, nx * 2, 1)
ls = np.zeros(2) + ls
print "length scale: "
print ls

# scale K toward more extreme probabilities
output_scale = 1.0 / logit(0.75)**2

nu0 = np.ones(J) # generate ground truth for the report locations
z0 = nu0[1] / np.sum(nu0)

# # VARYING NOISE TESTS -------------------------------------------------------------------------------------------
# This will also test sparseness so we don't need to run the algorithms again to show the effect of sparseness,
# just make different plots!

# Number of datasets
nruns = 25
nsteps = 11

# REPORTS
Nreports = 2100#500 #400 # total number of reports in complete data set
Nreps_initial = 100#50 #50 # number of labels in first iteration data set.
Nrep_inc = (Nreports - Nreps_initial) / (nsteps - 1) # increment the number of labels at each iteration
logging.info('Incrementing number of reports by %i in each iteration.' % Nrep_inc)

# REPORTERS
S = 20 # number of reporters

nproportions = 5
if nproportions > S:
    nproportions = S
weak_proportions = np.arange(0.0, nproportions)
weak_proportions /= (nproportions - 1)

# Run only lowest proportion
#weak_proportions = [1]

# values to be used in each experiment
cluster_spreads = [0]#[0.2]#[1.0, 0.5, 0.2] # spreads are multiplied by average distance between reports and a random
#number between 0 and 1 with mean 0.5. So spread of 1 should give the default random placement of reports.

shape_s0 = 1
rate_s0 = 1
# 10, 4 would make it so that the prior expected precision is around 2.58, corresponding to std of a beta-distributed variable
# with hyper-parameters 5, 5.
alpha0_offdiag = 1
alpha0_diag = 2

# See run_experiments: snap-to-grid flag indicates whether we should snap report locations to their nearest grid location.
# Doing so means that we assume reports relate to the whole grid square, and that different sources relate to the same
# t object. We could reduce problems with this discretization step if we use soft snapping based on distance.
# When set to true, the model predicts the state of each grid location, and the latent density of states. Lots of
# reports at same place does not necessarily imply high density, which makes sense if there is only a single emergency.
# When set to false, the model predicts the density of reports at each location, if the reports were accurate,
# and assumes that reports may relate to different events at the same location.

# DATA GENERATION --------------------------------------------------------------------------------------------------

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
    z_plot = griddata(x_all.reshape(-1), y_all.reshape(-1), f_all.reshape(-1), xi, yi, interp='linear')

    if not ax:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    ax.plot_surface(x_plot, y_plot, z_plot, cmap="Spectral", rstride=1, cstride=1)
    #ax.set_zlim3d(0, 1)
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

def gen_synth_ground_truth(reset_all_data, nx, ny, Nreports, Ntest, ls, snap_to_grid, experiment_label, dataset_label,
                           mean_reps_per_cluster=1, clusterspread=0, outputscale_new=-1):
    '''
    Generate data.
    '''
    if outputscale_new != -1:
        output_scale = outputscale_new
    
    _, data_outputdir = dataset_location(experiment_label, dataset_label)

    if not reset_all_data and os.path.isfile(data_outputdir + "x_all.npy"):
        xreports = np.load(data_outputdir + "x_all.npy")[:Nreports]
        yreports = np.load(data_outputdir + "y_all.npy")[:Nreports]
        t_gold = np.load(data_outputdir + "t_all.npy")[:Nreports]
        return xreports, yreports, t_gold

    #reps per cluster controls how concentrated the reports are in an group. mean_reps_per_cluster==1 puts all reports at
    # separate locations.
    nclusters = Nreports / int(mean_reps_per_cluster)
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
    Ntest = int(np.round(Ntest **0.5))
    xtest = (np.arange(Ntest) * float(nx) / Ntest)[np.newaxis, :]
    xtest = np.tile(xtest, (Ntest, 1))
    xtest = xtest.reshape(Ntest**2, 1)
    ytest = (np.arange(Ntest) * float(ny) / Ntest)[:, np.newaxis]
    ytest = np.tile(ytest, (1, Ntest))
    ytest = ytest.reshape(Ntest**2, 1)

    if snap_to_grid:
        # convert xreports and yreports to discrete grid locations.
        xreports = np.floor(xreports)
        yreports = np.floor(yreports)
        xtest = np.floor(xtest)
        ytest = np.floor(ytest)

    x_all = np.concatenate((xreports, xtest))
    y_all = np.concatenate((yreports, ytest))

    ddx = x_all - x_all.T
    ddy = y_all - y_all.T

    #K = sq_exp_cov(ddx, ddy, ls)
    K = matern_3_2(ddx, ddy, ls)
    f_mean = np.zeros(len(x_all))
    f_all = mvn.rvs(mean=f_mean, cov=K / output_scale)

    # generate ground truth
    rho_all = sigmoid(f_all) # class density
    t_all = bernoulli.rvs(rho_all)

    if snap_to_grid:
        # find duplicate locations and use the last t_all value for each.
        for i in range(len(f_all)):
            xi = x_all[i]
            yi = y_all[i] # Geordie variable

            samex = np.argwhere(x_all==xi)[:, 0]
            samexy = samex[y_all[samex, 0]==yi]

            f_all[samexy] = f_all[i]
            t_all[samexy] = t_all[i]
            rho_all[samexy] = rho_all[i]

    if PLOT_SYNTH_DATA:
        plot_density(nx, ny, x_all, y_all, f_all, apply_sigmoid=False)

    # report locations
    t_rep = t_all[:Nreports]
    # test locations
    t_test = t_all[Nreports:]

    if PLOT_SYNTH_DATA:
        plot_density(nx, ny, xreports, yreports, rho_all[:Nreports], "\rho at report locations", apply_sigmoid=False)

    print "Fraction of positive training target points: %.3f" % (np.sum(t_rep) / float(len(t_rep)))
    print "Fraction of positive test target points: %.3f" % (np.sum(t_test) / float(len(t_test)))

    # Save test data to file
    np.save(data_outputdir + "x_all.npy", x_all)
    np.save(data_outputdir + "y_all.npy", y_all)
    np.save(data_outputdir + "f_all.npy", f_all)
    np.save(data_outputdir + "t_all.npy", t_all)

    return xreports, yreports, t_rep

def gen_synth_reports(reset_all_data, Nreports, diags, off_diags, bias_vector, xreports, yreports, t_gold, snap_to_grid,
                      experiment_label, dataset_label, Scurrent=-1):

    if Scurrent != -1:
        S = Scurrent

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

#     if PLOT_SYNTH_DATA:
#         hist, bins = np.histogram(reporter_ids, S)
#         plt.figure()
#         plt.bar(bins[:-1], hist)
#         plt.xlabel("Reporter IDs")
#         plt.ylabel("Number of reports from each reporter")

    pi_reps = pi_reps[t_gold[:, np.newaxis], reporter_ids]
    reports = bernoulli.rvs(pi_reps)
    print "Fraction of positive reports: %.3f" % (np.sum(reports) / float(len(reports)))

    if PLOT_SYNTH_DATA:
        plot_report_histogram(xreports, yreports, reports)

    C = np.concatenate((reporter_ids, xreports, yreports, reports), axis=1)

    np.save(data_outputdir + "C.npy", C)
    np.save(data_outputdir + "pi_all.npy", pi.swapaxes(0,1).reshape((J*2, S), order='F').T )# pi all.
    
"""
 TODO:

 5. Calculate mean + variance of statistics across multiple runs.
 6. Plotting across all runs and all iterations -- option to turn off plotting for individual run/iteration.
 7. Log likelihood measure.
 8. Save plots so we can run on florence.
"""

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

# Global variables ----------------------------------------------------------------------------------------------------

xreports = []
yreports = []
t_gold = []
x_all = []
y_all = []
f_all = []
t_all = []
xtest = []
ytest = []
t_test_gold = []
rho_test = []
C = []
tester = []

# These ones will be redefined in the scripts used to run specific experiments
diag_reliable = 10.0
off_diag_reliable = 1.0
bias_reliable = np.zeros(J)

# For the unreliable workers to have white noise
diag_weak = 5.0
off_diag_weak = 5.0
bias_weak = np.zeros(J)

# MAIN SET OF SYNTHETIC DATA EXPERIMENTS ------------------------------------------------------------------------------
def run_experiments(expt_label_template, dstart=0, dend=nruns, p_idx_start=0, p_idx_end=nproportions, snap_to_grid=False):
    print "Output scale for ground truth: " + str(output_scale)
    
    for cluster_spread in cluster_spreads:

        experiment_label = expt_label_template % cluster_spread

        for d in range(dstart, dend):
            for p_idx, p in enumerate(weak_proportions):

                if p_idx < p_idx_start:
                    continue
                if p_idx >= p_idx_end:
                    continue

                diags = np.ones(S)
                diags[:int(S * p)] = diag_reliable
                diags[int(S * p):] = diag_weak
                off_diags = np.ones(S)
                off_diags[:int(S * p)] = off_diag_reliable
                off_diags[int(S * p):] = off_diag_weak
                biases = np.ones((S, J))
                biases[:int(S * p), :] = bias_reliable
                biases[int(S * p):, :] = bias_weak

                dataset_label = "d%i" % d
                logging.info("Generating data/reloading old data for proportion %i, Dataset %d" % (p_idx, d))
                # only reset on the first iteration
                xreports, yreports, t_gold = gen_synth_ground_truth(RESET_ALL_DATA & (p_idx==0), nx, ny, Nreports,
                    Ntest, ls, snap_to_grid, experiment_label, dataset_label, 1, 0, outputscale_new=output_scale)#5, cluster_spread * nx / Nreports**0.5)
                dataset_label = "p%f_d%i" % (p, d)
                gen_synth_reports(RESET_ALL_DATA, Nreports, diags, off_diags, biases, xreports, yreports, t_gold,
                                  snap_to_grid, experiment_label, dataset_label, S)

        # RUN TESTS -----------------------------------------------------------------------------------------------------------
        for p_idx, p in enumerate(weak_proportions):

            if p_idx < p_idx_start:
                continue
            if p_idx >= p_idx_end:
                continue

            diags = np.ones(S)
            diags[:int(S * p)] = diag_reliable
            diags[int(S * p):] = diag_weak
            off_diags = np.ones(S)
            off_diags[:int(S * p)] = off_diag_reliable
            off_diags[int(S * p):] = off_diag_weak

            for d in range(dstart, dend):
                dataset_label = "d%i" % d
                logging.info("Running tests for proportion %i, Dataset %d" % (p_idx, d))

                outputdir, data_outputdir = dataset_location(experiment_label, dataset_label)
                x_all = np.load(data_outputdir + "x_all.npy")
                y_all = np.load(data_outputdir + "y_all.npy")

                if snap_to_grid:
                    x_all = np.floor(x_all).astype(int)
                    y_all = np.floor(y_all).astype(int)

                xtest = x_all[Nreports:]
                ytest = y_all[Nreports:]

                f_all = np.load(data_outputdir + "f_all.npy")
                f_test = f_all[Nreports:]
                rho_test = sigmoid(f_test)
                t_all = np.load(data_outputdir + "t_all.npy" )# t_all
                t_test_gold = t_all[Nreports:]

                dataset_label = "p%f_d%i" % (p, d)
                outputdir, data_outputdir = dataset_location(experiment_label, dataset_label)
                C = np.load(data_outputdir + "C.npy")

                alpha0 = np.zeros((J, 2, S)) + alpha0_offdiag
                for s in range(S):
                    alpha0[np.arange(J), np.arange(J), s] += alpha0_diag

                # Run the tests with the current data set
                tester = prediction_tests.Tester(outputdir, methods, Nreports, z0, alpha0, nu0, shape_s0, rate_s0,
                                                 ls[0], optimise=False, verbose=False)
                tester.run_tests(C, nx, ny, xtest.reshape(-1), ytest.reshape(-1), t_test_gold, rho_test, Nreps_initial,
                                 Nrep_inc)
                if SAVE_RESULTS:
                    tester.save_separate_results()

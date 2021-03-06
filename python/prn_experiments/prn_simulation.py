"""
Load Planetary response network data and run active learning simulations.
"""

import sys

sys.path.append("./python")
sys.path.append("../HeatMapBCC/python")
sys.path.append("../pyIBCC/python")

from prediction_tests import Tester
import numpy as np
import logging
import os
from scipy.stats import gamma

import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D

import prn

# HELPER FUNCTIONS -------------------------------------------------------------------------------------------------------

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


# DATA GENERATION --------------------------------------------------------------------------------------------------
def dataset_location(experiment_label, dataset_label):
    # ID of the data set
    outputdir = "./data/" + experiment_label + '/'
    
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    
    outputdir = outputdir + "prn_%s/" % (dataset_label)
    logging.debug("Using output directory %s" % outputdir)
    
    data_outputdir = outputdir + '/prn_data/'
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
    z_plot = griddata(x_all.reshape(-1), y_all.reshape(-1), f_all.reshape(-1), xi, yi)
    
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

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

# EXPERIMENT CONFIG ---------------------------------------------------------------------------------------------------

RESET_ALL_DATA = False
PLOT_SYNTH_DATA = False
SAVE_RESULTS = True

methods = [
           'MV',
           'NN',
           'SVM',
           #'oneclassSVM',
           #'KDE',
           #'IBCC',
           'GP',
           'IBCC+GP',
           'HeatmapBCC'
           ]

# Number of datasets
nruns = 20
nsteps = 5

featurenames = ['structural_damage_3']
# , 'tarp', 'structural_damage_2']#[, 'tarp', 'structural_damage_2']

neg_sample_size = 0.2#0.5#0.1 # how many of the "no mark" labels to use?
Nreps_initial_fraction = 0.1

topdir = 'prn4/'
expt_label_template = topdir + '%s'

def load_data(featurename='structural_damage_3'):
    C = np.load(prn.Cfile % (featurename, 0))
    t_all = np.load(prn.goldfile % ('p', featurename))# t_all    

    nneg = np.sum(C[:, 3] == 0)
    npos = np.sum(C[:, 3] > 0)
    
    #0.5 used in successful expmts
    Nreports = 0.5 * np.floor(neg_sample_size * nneg + npos) #C.shape[0] # total number of reports in complete data set - don't use whole dataset, it's too large

    return C, Nreports, prn.nx, prn.ny, t_all, t_all 

# LOAD THE GOLD DATA ---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    for featurename in featurenames:
        C, Nreports, nx, ny, t_all, _ = load_data(featurename)
        
        testidxs = np.load(prn.goldfile % ('idxs', featurename))        
        linear_unique = np.load(prn.goldfile % ('linearidxs', featurename))
        
        # REPORTS
        Nreps_initial = int(Nreports * Nreps_initial_fraction) #60 #50 # number of labels in first iteration data set. 
        Nrep_inc = int((Nreports - Nreps_initial) / (nsteps - 1)) # increment the number of labels at each iteration    
        logging.info('Number of reports = %i. Incrementing number of reports by %i in each iteration.' % (Nreports, Nrep_inc))
        
        experiment_label = expt_label_template % featurename
        if not os.path.isdir('./data/' + experiment_label):
            os.mkdir('./data/' + experiment_label)
        
        J = 2 # number of classes
        L = len(np.unique(C[:, 3])) # number of scores/label types
        nx = prn.nx
        ny = prn.ny
        x_all, y_all = np.unravel_index(linear_unique[testidxs], (int(nx), int(ny)))
        t_all = t_all[testidxs]
        Ntest = len(x_all) # no. points in the grid to test
        
        # HYPERPARAMETERS ------------------------------------------------------------------------------------------------------

        optimize = True
        ls = 20
        lpls = None
        
        # as an alternative to using the standard optimiser, we can test sample length-scale values here
#         optimize = False         
#         ls = [4, 8, 16, 32, 64] # a range of lengthscales that are a very roughly representative sample from a gamma distribution with shape=1 and scale=30
#         lpls = gamma.logpdf(ls, 2, scale=22) # informative prior over the lengthscales
        
        print "initial length scale: "
        print ls
        
        nu0 = np.ones(J) # generate ground truth for the report locations
        nu0[:] = prn.bestnu0
        
        z0 = nu0[1] / np.sum(nu0)
        
        shape_s0 = 0.5
        rate_s0 = 10.0 * 0.5        
         
        alpha0 = np.ones((J, L), dtype=float)
        if J==L:
            alpha0[range(J), range(L)] = 5
        elif featurename=='structural_damage_1':
            alpha0[1, 1] = 2
            alpha0[0, 0] = 2     
            alpha0 *= 3            
        elif featurename=='structural_damage_2':
            alpha0[1, 2] = 2
            alpha0[0, 0] = 2
            alpha0 *= 3
        elif featurename=='structural_damage_3':
            alpha0[1, 3] = 2
            alpha0[0, 0] = 2
            alpha0 *= 3
        else:
            alpha0[0, 0] = 19
            alpha0[1:, 0] = 15
        
        snap_to_grid = True

    # MAIN SET OF SYNTHETIC DATA EXPERIMENTS ---------------------------------------------------------------------------
        # RUN TESTS ----------------------------------------------------------------------------------------------------
        for d in range(nruns):
            dataset_label = "d%i" % d
            logging.info("Running tests for dataset %d" % (d))
                    
            outputdir, data_outputdir = dataset_location(experiment_label, dataset_label)
            
            xtest = x_all
            ytest = y_all
            rho_test = []
            
            # random selection of 1000 test points for debugging
            #ntestd = 1000
            #selection = np.random.choice(len(x_all), ntestd, replace=False)
            xtest = x_all#[selection]
            ytest = y_all#[selection]
            t_test_gold = t_all[:, 1]#[selection, 1]
            
            # select only 10% of negative labels -- we want roughly balanced training data
            negidxs = np.argwhere(C[:, 3] == 0)
            nneg = len(negidxs)
            selected_neg = np.random.choice(nneg, int(nneg * neg_sample_size), replace=False)
            selected_neg_idxs = np.zeros(len(C), dtype=bool)
            selected_neg_idxs[negidxs[selected_neg]] = True
            pos_idxs = C[:, 3] > 0
            C_d = C[selected_neg_idxs | pos_idxs, :]
             
            # shuffle the order of the C data points
            shuffle_idxs = np.random.permutation(C_d.shape[0])
            C_d = C_d[shuffle_idxs, :]
            if snap_to_grid:
                C_d = C_d.astype(int)
            # Run the tests with the current data set
            tester = Tester(outputdir, methods, Nreports, z0, alpha0, nu0, shape_s0, rate_s0, 
                                             ls, optimise=optimize, verbose=False, lpls=lpls)
            tester.ignore_report_point_density = True
            t_test_gold_density = np.copy(t_test_gold)
            # ignore locations where the gold is undecided, even for the density estimator
            t_test_gold[(t_test_gold > 0.1) & (t_test_gold < 0.9)] = -1 
            t_test_gold_density[(t_test_gold_density > 0.1) & (t_test_gold_density < 0.9)] = -1          
            tester.run_tests(C_d, nx, ny, xtest.reshape(-1), ytest.reshape(-1), t_test_gold, t_test_gold_density, 
                             Nreps_initial, Nrep_inc)
            #use this line to recompute metrics but keep same results
#             tester.reevaluate(C_d, nx, ny, xtest.reshape(-1), ytest.reshape(-1), t_test_gold, t_test_gold_density, Nreps_initial, 
#                              Nrep_inc)
            if SAVE_RESULTS:
                tester.save_separate_results()

"""
Load Planetary response network data and run active learning simulations.
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

import prn
from heatmapbcc import HeatMapBCC
from gpgrid import GPGrid

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
    #output 5 was most working one so far
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
            'IBCC',
            'KDE',
            'GP',
            'IBCC+GP',
            'HeatmapBCC'
           ]

# Number of datasets
nruns = 20
nsteps = 5

featurenames = ['structural_damage_3', 'tarp', 'structural_damage_2']

# LOAD THE GOLD DATA ---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    for featurename in ['structural_damage_3']:#featurenames:
        C = np.load(prn.Cfile % (featurename, 0))
        linear_unique = np.load(prn.goldfile % ('linearidxs', featurename))
        testidxs = np.load(prn.goldfile % ('idxs', featurename))
        t_all = np.load(prn.goldfile % ('p', featurename))# t_all
        
        # REPORTS
        nneg = np.sum(C[:, 3] == 0)
        npos = np.sum(C[:, 3] > 0)
        neg_sample_size = 1#1.0#0.1 # how many of the "no mark" labels to use?
        Nreports = np.floor(neg_sample_size * nneg) + npos #C.shape[0] # total number of reports in complete data set - don't use whole dataset, it's too large
        Nreps_initial = 0.65 * Nreports #60 #50 # number of labels in first iteration data set. 
        Nrep_inc = (Nreports - Nreps_initial) / (nsteps - 1) # increment the number of labels at each iteration    
        logging.info('Number of reports = %i. Incrementing number of reports by %i in each iteration.' % (Nreports, Nrep_inc))
        
        C_sample = C[np.random.choice(C.shape[0], Nreps_initial, replace=False), :]
        
#         C1 = np.random.choice(10000, Nreps_initial, replace=False)[:, np.newaxis]
#         C2 = np.random.choice(10000, Nreps_initial, replace=False)[:, np.newaxis]
#         C3 = np.random.randint(0, 2, (Nreps_initial, 1))
#         C0 = np.zeros((Nreps_initial, 1))
#         C_sample = np.concatenate((C0, C1, C2, C3), axis=1)
        
        expt_label_template = 'prn/%s' % featurename
        
        J = 2 # number of classes
        L = len(np.unique(C[:, 3])) # number of scores/label types
        nx = prn.nx
        ny = prn.ny
        x_all, y_all = np.unravel_index(linear_unique[testidxs], (nx, ny))
        t_all = t_all[testidxs]
        Ntest = len(x_all) # no. points in the grid to test
        
        # HYPERPARAMETERS ------------------------------------------------------------------------------------------------------
        nu0 = np.ones(J) # generate ground truth for the report locations
        nu0[:] = prn.bestnu0
        #nu0[0] = 50
        
        z0 = nu0[1] / np.sum(nu0)
        
        shape_s0 = 0.5
        rate_s0 = shape_s0  * 10.0 # 2.58 #2.83 * shape_s0 # 3.5         
        
        #shape_s0 = 10
        #rate_s0 = 4 # chosen so that the prior expected precision is 2.58, corresponding to std of a beta-distributed variable
        # with hyper-parameters 5, 5. 
        alpha0 = np.ones((J, L), dtype=float)
        if J==L:
            alpha0[range(J), range(L)] = 5
        
#         if featurename=='structural_damage_1':
#             alpha0[1, 1] = 2
#             alpha0[0, 0] = 2                 
#         elif featurename=='structural_damage_2':
#             alpha0[1, 2] = 2
#             alpha0[0, 0] = 2
#         elif featurename=='structural_damage_3':
#             alpha0[1, 3] = 2
#             alpha0[0, 0] = 2
#         else:
        alpha0[0, 0] = 19
        alpha0[1:, 0] = 15
            
        snap_to_grid = True
    
        print "number of reports: %i" % Nreports
        
        lengthscales =[5]#[128, 64, 32, 20, 16, 8, 4, 2, 1]#, 256, 512]
        
        lml_h = np.zeros(len(lengthscales))
        lml_g = np.zeros(len(lengthscales))
        
        for i, ls in enumerate(lengthscales):
            # default hyper-parameter initialisation points for all the GPs used below
            shape_ls = 2.0
            rate_ls = shape_ls / ls
                              
            #HEATMAPBCC OBJECT
            heatmapcombiner = HeatMapBCC(nx, ny, J, L, alpha0, np.unique(C_sample[:,0]).shape[0], z0=z0, shape_s0=shape_s0, 
                          rate_s0=rate_s0, shape_ls=shape_ls, rate_ls=rate_ls, force_update_all_points=True)
            heatmapcombiner.min_iterations = 4
            heatmapcombiner.max_iterations = 200
            heatmapcombiner.verbose = True
            heatmapcombiner.conv_threshold = 1e-3
            heatmapcombiner.uselowerbound = True
                    
            logging.info("Running HeatmapBCC... length scale = %f" % ls)
               
            heatmapcombiner.combine_classifications(C_sample)
            results, _, _ = heatmapcombiner.predict(C_sample[:, 1], C_sample[:, 2], variance_method='sample')
            results = results[1, :]
                
            lml_h[i] = heatmapcombiner.lowerbound()
        
#             gpgrid = GPGrid(nx, ny, z0=z0, shape_s0=shape_s0, rate_s0=rate_s0, shape_ls=2.0, rate_ls=rate_ls)
#             gpgrid.min_iter_VB = 5
#             gpgrid.verbose = True
#             gpgrid.max_iter_G = 10
#             gpgrid.conv_threshold = 1e-5
#             gpgrid.conv_check_freq = 1
#             countsize = 1.0
#             gpgrid.fit((C_sample[:, 1], C_sample[:, 2]), C_sample[:, 3] * countsize, totals=np.zeros((C_sample.shape[0], 1)) + countsize,
#                        update_s = True)
#             results, _ = gpgrid.predict((C_sample[:, 1], C_sample[:, 2]), variance_method='sample')
#             results = results[:, 0]
#             
#             lml_g[i] = gpgrid.lowerbound()
             
            plt.figure()
            plt.title('Length scale %i' % ls)
            #cmap = plt.get_cmap('jet')                
            #cmap._init()
            plt.scatter(C_sample[:, 1], C_sample[:, 2], c=results)
            
            print "LL = %.3f" % np.sum(C_sample[:, 3] * np.log(results) + (1-C_sample[:, 3]) * np.log(1 - results))
            
             
            #logging.debug("output scale: %.5f" % heatmapcombiner.heatGP[1].s)
            logging.info("HeatmapBCC Lower bound: %.4f with length scale %f" % (lml_h[i], ls))
            logging.info("GP Lower bound: %.4f with length scale %f" % (lml_g[i], ls))
        
        for i, ls in enumerate(lengthscales):
            logging.info("HeatmapBCC Lower bound: %.4f with length scale %f" % (lml_h[i], ls))
            logging.info("GP Lower bound: %.4f with length scale %f" % (lml_g[i], ls))
        
        plt.figure()
        plt.plot(lengthscales, lml_h, label='HeatmapBCC')
        plt.plot(lengthscales, lml_g, label='GP')
        plt.title("Lower bound variation with Length Scale on PRN Data")
        plt.xlabel("Length scale")
        plt.ylabel("Variational Lower Bound")
        plt.legend(loc='best')
        plt.savefig("./lengthscale_prn_test_%s.png" % featurename)
        
        plt.figure()
        plt.title('Data')
        #cmap = plt.get_cmap('jet')                
        #cmap._init()
        cols = np.array(['r', 'y'])
        plt.scatter(C_sample[:, 1], C_sample[:, 2], c=cols[C_sample[:, 3]])
        
        print "Optimal lengthscale for heatmapBCC: %f" % lengthscales[np.argmax(lml_h)]
        print "Optimal lengthscale for GP: %f" % lengthscales[np.argmax(lml_g)]
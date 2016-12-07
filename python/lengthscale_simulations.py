'''
Created on 6 Dec 2016

@author: edwin
'''

"""
Experiment 2a. VARYING SPARSENESS: Generate noisy reports with a mixture of reliabilities (different levels of biased workers).
These should be uniformly distributed through the space.
When we call prediction_tests, we will evaluate with decreasing sparseness in each iteration. 
The sparseness can be due to either the spacing of the reports or the total number of reports from each worker, 
both of which will affect the reliability model of the workers and the GP. We are more interested in the former, as 
standard IBCC deals with the latter. 

Experiment 2b: Possible alternative view to plot. Keep a fixed sparseness, but different numbers of biased workers.

"""

import sys
from gen_synthetic import dataset_location

import numpy as np
from gen_synthetic import gen_synth_ground_truth, gen_synth_reports
from gpgrid import GPGrid
#import gen_synthetic as gs

from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import Matern

RESET_ALL_DATA = True

nx = 5
ny = 5
Nreports = 100
Ntest = nx * ny
ls = 10
snap_to_grid = False
experiment_label = "lengthscale_sim_expt"
dataset_label = "lengthscale_sim_data"

J = 2
S = 20
diags = np.ones(S)
diags[:5] = 200
diags[5:] = 1
off_diags = np.ones(S)
off_diags[:5] = 1
off_diags[5:] = 1
biases = np.ones((S, J))
biases[:5, :] = 0
biases[5:, :] = 0

if __name__ == '__main__':    
    xreports, yreports, t_gold = gen_synth_ground_truth(RESET_ALL_DATA, nx, ny, Nreports,
                    Ntest, [ls, ls], snap_to_grid, experiment_label, dataset_label, 1, 0)
    
    C, xreports, yreports, reports = gen_synth_reports(RESET_ALL_DATA, Nreports, diags, off_diags, biases, xreports, 
                                                   yreports, t_gold, snap_to_grid, experiment_label, dataset_label)

    outputdir, data_outputdir = dataset_location(experiment_label, dataset_label)
    x_all = np.load(data_outputdir + "x_all.npy")
    y_all = np.load(data_outputdir + "y_all.npy")
    f_all = np.load(data_outputdir + "f_all.npy")
    t_all = np.load(data_outputdir + "t_all.npy" )# t_all
    
    x_test = x_all[Nreports:]
    y_test = y_all[Nreports:]
    
    #gpgrid = GPGrid(nx, ny, z0=0.5, shape_s0=100, rate_s0=10, shape_ls=2.0, rate_ls=rate_ls)
    #gpgrid.fit([xreports, yreports], reports)    
    #gp_preds, gp_var = gpgrid.predict([x_all, y_all], variance_method='sample')
    
    kernel = Matern(length_scale=1)
    gpc = GPC(kernel=kernel, warm_start=False, n_restarts_optimizer=2)
    gpc.fit(np.concatenate((xreports, yreports), axis=1), reports)
    ls_found = gpc.kernel_.theta[0]
#     os_found = gpc.kernel_.theta[1]
    print "length scale: %f" % ls_found
#     print "output scale: %f" % os_found
    params = gpc.get_params(deep=True)
    gpc.predict_proba(np.concatenate((x_test, y_test), axis=1))
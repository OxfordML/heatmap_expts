'''

Plot the ground truth density and the predictions from each method.

Created on 26 Nov 2015

@author: edwin
'''

import numpy as np
import matplotlib.pyplot as plt
import gen_synthetic
import logging

if __name__ == '__main__':
    # import settings from where the experiments were run    
    nruns = gen_synthetic.nruns
    weak_proportions = gen_synthetic.weak_proportions
    Nreps_initial = gen_synthetic.Nreps_initial
    Nreports = gen_synthetic.Nreports 
    Nrep_inc = gen_synthetic.Nrep_inc
    Nsteps = gen_synthetic.nsteps
    
    # get a set of x-coordinates for the number of reports at each iteration
    Nreps_iter = np.arange(Nsteps) * Nrep_inc + Nreps_initial
    
    steps = [50, 200, 350, 500]#[gen_synthetic.Nreports] 
    experiment_label = gen_synthetic.expt_label_template % gen_synthetic.cluster_spreads[0]
    
    for Nreps in steps:
        for d in range(nruns):
            for p_idx, p in enumerate(weak_proportions):
                for cs in gen_synthetic.cluster_spreads:
                    expt_label = gen_synthetic.expt_label_template % cs
                    dataset_label = "d%i" % d
                    outputdir, data_outputdir = gen_synthetic.dataset_location(experiment_label, dataset_label)
                    # load the ground truth
                    f = np.load(data_outputdir + "f_all.npy")
                    x = np.load(data_outputdir + "x_all.npy")
                    y = np.load(data_outputdir + "y_all.npy")
                    ftest = f[gen_synthetic.Nreports:]
                    xtest = x[gen_synthetic.Nreports:]
                    ytest = y[gen_synthetic.Nreports:]
                    
                    # Load the posteriors
                    dataset_label = "p%i_d%i" % (p_idx, d)
                    logging.info("Loading results for proportion %i, Dataset %d" % (p_idx, d))
                    outputdir, _ = gen_synthetic.dataset_location(experiment_label, dataset_label)
                    post = np.load(outputdir + "density_results.npy").item() # posterior densities
                    ### CHANGE THIS TO CORRECT FILENAME AND REMOVE SQRT WHEN RERUNNING
                    var = np.load(outputdir + "density_var.npy").item() # posterior standard deviations of density
                    
                    methods = post[Nreps].keys()
                    
                    
                    fig = plt.figure()
                    nrows = 2.0
                    # plot ground truth
                    subplot = fig.add_subplot(nrows, np.ceil((len(methods) + 1) / nrows), 1,  projection='3d')
                    gen_synthetic.plot_density(gen_synthetic.nx, gen_synthetic.ny, xtest, ytest, ftest, ax=subplot)
                    
                    for i, m in enumerate(methods):
                        subplot = fig.add_subplot(nrows, np.ceil((len(methods) + 1) / nrows), 2 + i,  projection='3d')
                        sd = var[Nreps][m] ** 0.5
                        gen_synthetic.plot_density(gen_synthetic.nx, gen_synthetic.ny, xtest, ytest, post[Nreps][m], title=m, 
                                                   apply_sigmoid=False, ax=subplot, transparency=0.2)
                        gen_synthetic.plot_density(gen_synthetic.nx, gen_synthetic.ny, xtest, ytest, post[Nreps][m] + sd, 
                                                   title=m, apply_sigmoid=False, ax=subplot, transparency=0.7)
                        gen_synthetic.plot_density(gen_synthetic.nx, gen_synthetic.ny, xtest, ytest, post[Nreps][m] - sd, 
                                                   title=m, apply_sigmoid=False, ax=subplot, transparency=0.7) 
                        
                    fig.suptitle('Densities for dataset %i with %i reports, when %.2f%% of the reporters are reliable' % 
                                 (d, Nreps, p*100.0))

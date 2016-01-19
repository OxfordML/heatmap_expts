'''
Created on 18 Jan 2016

@author: edwin
'''

from heatmapbcc import HeatMapBCC
import logging
import numpy as np
from ushahididata import UshahidiDataHandler
from gen_synthetic import dataset_location
from scipy.stats import gamma

nx = 500
ny = 500

def load_data():
    
    datahandler = UshahidiDataHandler(nx,ny, './data/')
    
    datahandler.discrete = False # do not snap-to-grid
    datahandler.load_data()
    # Could see how the methods vary with number of messages, i.e. when the messages come in according to the real
    # time line. Can IBCC do better early on?
    C = datahandler.C[1]
    # Put the reports into grid squares
    C[:, 1] = np.round(C[:, 1])
    C[:, 2] = np.round(C[:, 2])

    # number of available data points
    Nreports =  C.shape[0]
    
    return C, Nreports

def load_synth_data():
    expt_label_template = 'synth/output_cluslocs%.2f_bias_grid1'
    experiment_label = expt_label_template % 0.2
    dataset_label = "p%i_d%i" % (2, 0)
    _, data_outputdir = dataset_location(experiment_label, dataset_label)
    C = np.load(data_outputdir + "C.npy")
    Nreports = C.shape[0]
    return C, Nreports

if __name__ == '__main__':
    # default hyper-parameters
    alpha0 = np.array([[3.0, 1.0], [1.0, 3.0]])[:,:,np.newaxis]
    alpha0 = np.tile(alpha0, (1,1,3))
    # set stronger priors for more meaningful categories
    alpha0[:,:,1] = np.array([[5.0, 1.0], [1.0, 5.0]]) # confident agents
    alpha0[:,:,2] = np.array([[1.0, 1.0], [1.0, 1.0]]) # agents with no prior knowledge of correlations
    clusteridxs_all = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2])
    alpha0_all = alpha0
    # set an uninformative prior over the spatial GP
    z0 = 0.5

    shape_s0 = 10
    rate_s0 = 4
    
    C, Nreports = load_data() 
    
    lengthscales = [8, 16, 32, 64, 128, 256, 512]
    
    lml = np.zeros(len(lengthscales))
    
    for i, ls in enumerate(lengthscales):
        # default hyper-parameter initialisation points for all the GPs used below
        shape_ls = 2.0
        rate_ls = shape_ls / ls
            
        #HEATMAPBCC OBJECT
        heatmapcombiner = HeatMapBCC(nx, ny, 2, 2, alpha0, np.unique(C[:,0]).shape[0], z0=z0, shape_s0=shape_s0, 
                      rate_s0=rate_s0, shape_ls=shape_ls, rate_ls=rate_ls, force_update_all_points=True)
        heatmapcombiner.min_iterations = 4
        heatmapcombiner.max_iterations = 200
        heatmapcombiner.verbose = False
        heatmapcombiner.uselowerbound = True
        
        logging.info("Running HeatmapBCC... length scale = %f" % ls)
        # to do:
        # make sure optimise works
        # make sure the optimal hyper-parameters are passed to the next iteration
        #heatmapcombiner.clusteridxs_alpha0 = clusteridxs
        heatmapcombiner.combine_classifications(C)
        lml[i] = heatmapcombiner.lowerbound()
        logging.debug("output scale: %.5f" % heatmapcombiner.heatGP[1].s)
        logging.info("Lower bound: %.4f" % lml[i])
        
        # bias toward ls=100
        shape_ls = 10
        rate_ls = 0.1
        logmodelprior = gamma.logpdf(ls, a=shape_ls, scale=1.0/rate_ls)
        logjoint = lml[i] + logmodelprior
        logging.debug("Lower bound + logmodelprior: %.4f" % logjoint)
        
        
    from matplotlib import pyplot as plt
    
    plt.figure()
    plt.plot(lengthscales, lml)
    plt.title("Lower bound variation with Length Scale on Ushahidi Data")
    plt.xlabel("Length scale")
    plt.ylabel("Variational Lower Bound")
    plt.savefig("./lengthscale_test.png")
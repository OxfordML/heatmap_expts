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
from gpgrid import GPGrid
#from sklearn import gaussian_process


nx = 100
ny = 100

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

def load_damage_data():
    import ushahidi_loader_damage as uld
    _, _, nx, ny, gold_labels, gold_density = uld.load_data(100, 100)
    # Use the gold labels to train the model so we can learn the correct length scale
        
    fraction = 1.0
        
    xtestgrid = np.arange(nx* fraction)[np.newaxis, :] + (nx - nx * fraction) * 0.5
    xtestgrid = np.tile(xtestgrid, (ny* fraction, 1))
    xtestgrid = xtestgrid.flatten()
      
    ytestgrid = np.arange(ny* fraction)[:, np.newaxis] + (ny - ny * fraction) * 0.5
    ytestgrid = np.tile(ytestgrid, (1, nx* fraction))
    ytestgrid = ytestgrid.flatten()
    
    #random selection
    selectionsize = 0.2
    idxs = np.random.choice(len(xtestgrid), selectionsize * len(xtestgrid))
    xtestgrid = xtestgrid[idxs]
    ytestgrid = ytestgrid[idxs]
       
    C = np.zeros((len(ytestgrid), 4))
    C[:, 1] = xtestgrid
    C[:, 2] = ytestgrid
    C[:, 3] = gold_labels[xtestgrid.astype(int), ytestgrid.astype(int)].flatten()
    
    return C, C.shape[0]

if __name__ == '__main__':
#     # default hyper-parameters
#     alpha0 = np.array([[3.0, 1.0], [1.0, 3.0]])[:,:,np.newaxis]
#     alpha0 = np.tile(alpha0, (1,1,3))
#     # set stronger priors for more meaningful categories
#     alpha0[:,:,1] = np.array([[5.0, 1.0], [1.0, 5.0]]) # confident agents
#     alpha0[:,:,2] = np.array([[1.0, 1.0], [1.0, 1.0]]) # agents with no prior knowledge of correlations
#     clusteridxs_all = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2])
#     alpha0_all = alpha0
    alpha0 = np.array([[100000.0, 1.0], [1.0, 100000.0]])
    # set an uninformative prior over the spatial GP
    z0 = 0.5

    shape_s0 = 0.5
    rate_s0 = 3.31 * 0.5 # the variance in the log space of a beta distributed variable with parameters 1, 1, is 3.31. 
    
    C, Nreports = load_damage_data()#load_data() 
    print "number of reports: %i" % Nreports
    
    lengthscales = [64, 32, 24, 16, 8, 4, 2]#, 256, 512]
    
    lml_h = np.zeros(len(lengthscales))
    lml_g = np.zeros(len(lengthscales))
    
#     gp = gaussian_process.GaussianProcess(corr='squared_exponential', theta0=10, thetaL=1, thetaU=200, verbose=True)
#     C[C[:, 3]<1e-6, 3] = 1e-6
#     C[C[:, 3]>1-1e-6, 3] = 1 - 1e-6
#     gp.fit(C[:, 1:3], np.log(C[:, 3] / (1 - C[:, 3]) ) )
#     print gp.theta_
        
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
        heatmapcombiner.conv_threshold = 1e-3
        heatmapcombiner.uselowerbound = True
            
        logging.info("Running HeatmapBCC... length scale = %f" % ls)
        # to do:
        # make sure optimise works
        # make sure the optimal hyper-parameters are passed to the next iteration
        #heatmapcombiner.clusteridxs_alpha0 = clusteridxs
        results = heatmapcombiner.combine_classifications(C)
        lml_h[i] = heatmapcombiner.lowerbound()
   
        gpgrid = GPGrid(nx, ny, z0=z0, shape_s0=shape_s0, rate_s0=rate_s0, shape_ls=2.0, rate_ls=rate_ls)
        gpgrid.min_iter_VB = 3
        gpgrid.verbose = False
        gpgrid.max_iter_G = 10
        gpgrid.conv_threshold = 1e-5
        gpgrid.conv_check_freq = 1
        countsize = 1.0
        results = gpgrid.fit((C[:, 1], C[:, 2]), C[:, 3] * countsize, totals=np.zeros((C.shape[0], 1)) + countsize)
   
        lml_g[i] = gpgrid.lowerbound()
         
        #logging.debug("output scale: %.5f" % heatmapcombiner.heatGP[1].s)
        logging.info("HeatmapBCC Lower bound: %.4f with length scale %f" % (lml_h[i], ls))
        logging.info("GP Lower bound: %.4f with length scale %f" % (lml_g[i], ls))
         
#         # bias toward ls=100
#         shape_ls = 10
#         rate_ls = 0.1
#         logmodelprior = gamma.logpdf(ls, a=shape_ls, scale=1.0/rate_ls)
#         logjoint = lml[i] + logmodelprior
#         logging.debug("Lower bound + logmodelprior: %.4f" % logjoint)
        
    from matplotlib import pyplot as plt
    
    for i, ls in enumerate(lengthscales):
        logging.info("HeatmapBCC Lower bound: %.4f with length scale %f" % (lml_h[i], ls))
        logging.info("GP Lower bound: %.4f with length scale %f" % (lml_g[i], ls))
    
    plt.figure()
    plt.plot(lengthscales, lml_h, label='HeatmapBCC')
    plt.plot(lengthscales, lml_g, label='GP')
    plt.title("Lower bound variation with Length Scale on Ushahidi Data")
    plt.xlabel("Length scale")
    plt.ylabel("Variational Lower Bound")
    plt.legend(loc='best')
    plt.savefig("./lengthscale_test.png")
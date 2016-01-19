'''

Building damage density estimation, compared against "gold standard".

'''

__author__ = 'edwin'

import logging
import numpy as np
from gen_synthetic import dataset_location
from ushahididata import UshahidiDataHandler
import prediction_tests


expt_label_template = "/ushahidi_damage/"

nruns = 20 # can use different random subsets of the reports C

# number of labels in first iteration dataset
Nreps_initial = 50
# increment the number of labels at each iteration
Nrep_inc = 50

def load_data():
    #Load up some ground truth
    datadir = './data/'
    goldfile_grid = datadir + 'haiti_unosat_target_grid.npy'
    building_density = np.load(goldfile_grid)
    gold_labels = building_density > 0

    # to determine gold density, take a fraction of this and surrounding areas to determine fraction affected
    gold_sums = building_density.copy()
    gold_sums[0:-1, :] += building_density[1:, :]
    gold_sums[1:, :] += building_density[0:-1, :]
    gold_sums[:, 1:] += building_density[:, 0:-1]
    gold_sums[:, 0:-1] += building_density[:, 1:]
    gold_sums[0:-1, 0:-1] += building_density[1:, 1:]
    gold_sums[1:, 1:] += building_density[0:-1, 0:-1]
    gold_sums[0:-1, 1:] += building_density[1:, 0:-1]
    gold_sums[1:, 0:-1] += building_density[0:-1, 1:]
    
    gold_sums[1:-1, 1:-1] /= 9.0
    gold_sums[0:1, 1:-1] /= 6.0
    gold_sums[-1, 1:-1] /= 6.0
    gold_sums[1:-1, 0:1] /= 6.0
    gold_sums[1:-1, -1] /= 6.0
    gold_sums[0, 0] /= 4.0
    gold_sums[-1, -1] /= 4.0
    gold_sums[0, -1] /= 4.0
    gold_sums[-1, 0] /= 4.0
    
    gold_density = gold_sums     

    nx = gold_density.shape[0]
    ny = gold_density.shape[1]
       
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
    
    return C, Nreports, nx, ny, gold_labels, gold_density    

if __name__ == '__main__':
    print "Run tests to determine the accuracy of the bccheatmaps method."

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    methods = [
               'KDE',
               'GP',
               'IBCC+GP',
               'HeatmapBCC'
               ]

    #LOAD THE DATA to run unsupervised learning tests ------------------------------------------------------------------
    C, Nreports, nx, ny, gold_labels, gold_density = load_data()

    # LIMIT THE NUMBER OF REPORTS
    Nreports = 1000

    # Test at the grid squares only
    xtest = np.arange(nx)[np.newaxis, :]
    xtest = np.tile(xtest, (ny, 1))
    xtest = xtest.flatten()
     
    ytest = np.arange(ny)[:, np.newaxis]
    ytest = np.tile(ytest, (1, nx))
    ytest = ytest.flatten()
    
    # default hyper-parameters
    alpha0 = np.array([[3.0, 1.0], [1.0, 3.0]])[:,:,np.newaxis]
    alpha0 = np.tile(alpha0, (1,1,3))
    # set stronger priors for more meaningful categories
    alpha0[:,:,1] = np.array([[5.0, 1.0], [1.0, 5.0]]) # confident agents
    alpha0[:,:,2] = np.array([[1.0, 1.0], [1.0, 1.0]]) # agents with no prior knowledge of correlations
    clusteridxs_all = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2, 0, 0, 2, 2])
    alpha0_all = alpha0
    # set an uninformative prior over the spatial GP
    nu0 = np.array([3.0, 1.0])
    z0 = nu0[1] / np.sum(nu0)

    shape_s0 = 10
    rate_s0 = 4
    
    ls = 10

    # Run the tests with the current dataset
    for d in range(nruns):
        
        # for each run, use different subsets of labels
        shuffle_idxs = np.random.permutation(C.shape[0])
        C = C[shuffle_idxs, :]
        C = C[:Nreports, :] # limit number of reports!
        
        dataset_label = "d%i" % (d)
        outputdir, _ = dataset_location(expt_label_template, dataset_label)         
        
        # Run the tests with the current data set
        tester = prediction_tests.Tester(outputdir, methods, Nreports, z0, alpha0, nu0, shape_s0, rate_s0, 
                ls, optimise=False, verbose=False)            
                
        tester.run_tests(C, nx, ny, xtest, ytest, gold_labels.flatten(), gold_density.flatten(),
                         Nreps_initial, Nrep_inc)
        tester.save_separate_results()
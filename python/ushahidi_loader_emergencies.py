'''

Emergency density estimation. There is no ground truth, so we first run all methods with all data, and find the points
they agree on. 

'''

__author__ = 'edwin'

import logging, os
import numpy as np
from gen_synthetic import dataset_location
from ushahididata import UshahidiDataHandler
import prediction_tests

RELOAD_GOLD = False # load the gold data from file, or compute from scratch?

expt_label_template = "/ushahidi_emergencies/"
    
nruns = 1 # can use different random subsets of the reports C

# number of labels in first iteration dataset
Nreps_initial = 65

# increment the number of labels at each iteration
Nrep_inc = 65

nx = 500.0
ny = 500.0

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
    
    accepted_categories = [0, 1, 2, 3, 11, 12, 15, 17]
    C = C[np.in1d(C[:, 0], accepted_categories), :]

    # number of available data points
    Nreports =  C.shape[0]
    
    return C, Nreports, nx, ny, [], []
    
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
    #Load up some ground truth
    datadir = './data/'

    # Test at the grid squares only
    xtest = np.arange(nx)[np.newaxis, :]
    xtest = np.tile(xtest, (ny, 1))
    xtest = xtest.flatten()
     
    ytest = np.arange(ny)[:, np.newaxis]
    ytest = np.tile(ytest, (1, nx))
    ytest = ytest.flatten()
       
    C, Nreports, _, _, _, _ = load_data()

    # default hyper-parameters
    alpha0 = np.array([[3.0, 1.0], [1.0, 3.0]])[:,:,np.newaxis]
    alpha0 = np.tile(alpha0, (1,1,3))
    # set stronger priors for more meaningful categories
    alpha0[:,:,1] = np.array([[5.0, 1.0], [1.0, 5.0]]) # confident agents
    alpha0[:,:,2] = np.array([[1.0, 1.0], [1.0, 1.0]]) # agents with no prior knowledge of correlations
    clusteridxs_all = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2])
    alpha0_all = alpha0
    # set an uninformative prior over the spatial GP
    nu0 = np.array([2.0, 1.0])
    z0 = nu0[1] / np.sum(nu0)

    shape_s0 = 10
    rate_s0 = 4
    
    ls = 10

    outputdir, _ = dataset_location(expt_label_template, "gold")
    
    if os.path.exists(outputdir + "results.npy") and RELOAD_GOLD:
        logging.info("LOADING GOLD DATA")
        results = np.load(outputdir + "results.npy").item()[Nreports]
        density_results = np.load(outputdir + "density_results.npy").item()[Nreports]
    else:
        logging.info("RUNNING METHODS WITH ALL LABELS TO ESTIMATE GOLD")
        tester = prediction_tests.Tester(outputdir, methods, Nreports, z0, alpha0, nu0, shape_s0, rate_s0, 
                ls, optimise=False, verbose=False)            
        tester.run_tests(C, nx, ny, xtest, ytest, [], [], Nreports, 1)
        tester.save_separate_results()
        
        results = tester.results_all[Nreports]
        density_results = tester.densityresults_all[Nreports]
    
    # find indices where all methods agree to within 0.1
    for i, m in enumerate(results):
        if i==0:
            mlabels = np.zeros((len(results[m]), len(results)))
            mdensity = np.zeros((len(results[m]), len(results)))
        mlabels[:, i] = results[m].flatten()
        mdensity[:, i] = density_results[m].flatten()
        
    gold_labels = np.mean(mlabels, axis=1)
    gold_density = np.mean(mdensity, axis=1)     
    
    gold_tol = 0.1
    
    for i in range(mlabels.shape[1]):       
        # check whether mlabels agree: gold labels are within tolerance, densities are within tolerance, and gold labels
        # have not flipped to the other side of 0.5.
        disagree_idxs = (np.abs(mlabels[:, i] - gold_labels) > gold_tol) | \
            (np.abs(mdensity[:, i] - gold_density) > gold_tol) | ((gold_labels>0.5) != (mlabels[:, i]>0.5)) \
                        | ((mlabels[:, i] > 0.4) & (mlabels[:, i] < 0.6))
            # We could also include only locations with confident classifications, but this biases away from places with
            # uncertain densities. Alternative is to select only when confident + there are reports. This might help
            # with AUC estimates -- but the models are not really confident enough to evaluate AUCs. 
        gold_labels[disagree_idxs] = - 1
        gold_density[disagree_idxs] = - 1
        
    goldidxs = gold_labels != - 1
    xtest = xtest[goldidxs]
    ytest= ytest[goldidxs]
    gold_labels = gold_labels[goldidxs]
    gold_density = gold_density[goldidxs]

    logging.info("NOW RUNNING TESTS")

    # Run the tests with the current dataset
    for d in range(nruns):
        
        # for each run, use different subsets of labels
        shuffle_idxs = np.random.permutation(C.shape[0])
        C = C[shuffle_idxs, :]
        
        dataset_label = "d%i" % (d)
        outputdir, _ = dataset_location(expt_label_template, dataset_label)         
        
        # Run the tests with the current data set
        tester = prediction_tests.Tester(outputdir, methods, Nreports, z0, alpha0, nu0, shape_s0, rate_s0, 
                ls, optimise=False, verbose=False)            
                
        tester.run_tests(C, nx, ny, xtest, ytest, gold_labels.flatten(), gold_density.flatten(),
                         Nreps_initial, Nrep_inc)
        tester.save_separate_results()
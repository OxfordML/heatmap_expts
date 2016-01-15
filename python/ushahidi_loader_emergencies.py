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

RELOAD_GOLD = True # load the gold data from file, or compute from scratch?

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
    
    nruns = 20 # can use different random subsets of the reports C

    nx = 100.0
    ny = 100.0
    
    # Test at the grid squares only
    xtest = np.arange(nx)[np.newaxis, :]
    xtest = np.tile(xtest, (ny, 1))
    xtest = xtest.flatten()
     
    ytest = np.arange(ny)[:, np.newaxis]
    ytest = np.tile(ytest, (1, nx))
    ytest = ytest.flatten()
       
    datahandler = UshahidiDataHandler(nx,ny, './data/')
    datahandler.discrete = False # do not snap-to-grid
    datahandler.load_data()
    # Could see how the methods vary with number of messages, i.e. when the messages come in according to the real
    # time line. Can IBCC do better early on?
    C = datahandler.C[1]
    # Put the reports into grid squares
    C[:, 1] = np.round(C[:, 1])
    C[:, 2] = np.round(C[:, 2])
    C_all = C # save for later

    K = datahandler.K
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

    # number of labels in first iteration dataset
    Nreps_initial = 700
    # increment the number of labels at each iteration
    Nrep_inc = 500

    # number of available data points
    Nreports =  C.shape[0]

    outputdir, _ = dataset_location("/ushahidi_emergencies/", "gold")
    
    if os.path.exists(outputdir + "results.npy") and RELOAD_GOLD:
        results = np.load(outputdir + "results.npy")
        density_results = np.load(outputdir + "density_results.npy")
    else:
        logging.info("RUNNING METHODS WITH ALL LABELS TO ESTIMATE GOLD")
        tester = prediction_tests.Tester(outputdir, methods, Nreports, z0, alpha0, nu0, shape_s0, rate_s0, 
                ls, optimise=False, verbose=False)            
        tester.run_tests(C, nx, ny, xtest, ytest, [], [], Nreps_initial, Nrep_inc)    
        tester.save_separate_results()
    
    # initialise the label sets.
    gold_labels = np.zeros(len(xtest)) - 1
    gold_density = np.zeros(len(xtest)) - 1
    
    # find indices where all methods agree to within 0.1
    for m in methods:
        mlabels = tester.results_all[m]
        mdensity = tester.densityresults_all[m]
        if not np.any(gold_labels): # first iteration
            gold_labels = mlabels 
            gold_density = mdensity
            continue
        # check whether mlabels agree
        disagree_idxs = (np.abs(mlabels - gold_labels) > 0.1) or (np.abs(mdensity - gold_density) > 0.1) 
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
        outputdir, _ = dataset_location("/ushahidi_damage/", dataset_label)         
        
        # Run the tests with the current data set
        tester = prediction_tests.Tester(outputdir, methods, Nreports, z0, alpha0, nu0, shape_s0, rate_s0, 
                ls, optimise=False, verbose=False)            
                
        tester.run_tests(C, nx, ny, xtest, ytest, gold_labels.flatten(), gold_density.flatten(),
                         Nreps_initial, Nrep_inc)
        tester.save_separate_results()
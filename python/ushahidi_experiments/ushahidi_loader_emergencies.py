'''

Emergency density estimation. There is no ground truth, so we first run all methods with all data, and find the points
they agree on. 

'''

__author__ = 'edwin'

import logging, os
import numpy as np
from gen_synthetic import dataset_location
from ushahididata import UshahidiDataHandler
from prediction_tests import Tester
# from reevaluate import Tester
from scipy.sparse import coo_matrix

RELOAD_GOLD = False # load the gold data from file, or compute from scratch?

expt_label_template = "/ushahidi_emergencies4/"
    
nruns = 20 # can use different random subsets of the reports C

# number of labels in first iteration dataset
Nreps_initial = 75

# increment the number of labels at each iteration
Nrep_inc = 50

nx = 100.0
ny = 100.0

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
           'MV',
           'NN',
           'oneclassSVM',
           'KDE',
           'IBCC',
           'GP',
           'IBCC+GP',
           'HeatmapBCC'
               ]

    #LOAD THE DATA to run unsupervised learning tests ------------------------------------------------------------------
    #Load up some ground truth
    datadir = './data/'

    # Test at the grid squares only
    xtestgrid = np.arange(nx)[np.newaxis, :]
    xtestgrid = np.tile(xtestgrid, (ny, 1))
    xtestgrid = xtestgrid.flatten()
      
    ytestgrid = np.arange(ny)[:, np.newaxis]
    ytestgrid = np.tile(ytestgrid, (1, nx))
    ytestgrid = ytestgrid.flatten()
       
    C, Nreports, _, _, _, _ = load_data()

    # default hyper-parameters
    alpha0 = np.array([[100.0, 1.0], [1.0, 100.0]])[:,:,np.newaxis]
    #alpha0 = np.tile(alpha0, (1,1,3))
    # set stronger priors for more meaningful categories
    #alpha0[:,:,1] = np.array([[5.0, 1.0], [1.0, 5.0]]) # confident agents
    #alpha0[:,:,2] = np.array([[1.0, 1.0], [1.0, 1.0]]) # agents with no prior knowledge of correlations
    #clusteridxs_all = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2])
    # set an uninformative prior over the spatial GP
    nu0 = np.array([2000.0, 1000.0])
    z0 = nu0[1] / np.sum(nu0)

    shape_s0 = np.sum(nu0) / 2.0 #2.0
    from scipy.stats import beta
    rho_samples = beta.rvs(nu0[1], nu0[0]+C.shape[0], size=50000)
    f_var = np.var(np.log(rho_samples / (1 - rho_samples)))
    rate_s0 =  0.5 * shape_s0 * f_var 
    rate_s0 = shape_s0   
    print "Rate s0 = %f" % rate_s0 
    
    ls = 16 # worked well with 5 but should be 16?

    outputdir, _ = dataset_location(expt_label_template, "gold")
    
    from ibcc import IBCC
    
    ibcc_combiner = IBCC(2, 2, alpha0, nu0, len(np.unique(C[:, 0])))                
#     ibcc_combiner.clusteridxs_alpha0 = clusteridxs_all
    ibcc_combiner.verbose = True
    ibcc_combiner.min_iterations = 5
    
    report_coords = (C[:,1].astype(int), C[:, 2].astype(int))
    linearIdxs = np.ravel_multi_index(report_coords, dims=(nx, ny))
    C_flat = C[:,[0,1,3]]
    C_flat[:,1] = linearIdxs
    bcc_pred = ibcc_combiner.combine_classifications(C_flat, optimise_hyperparams=False)
    
    gold_coords_grid = coo_matrix((np.ones(C.shape[0]), report_coords), (nx, ny))
    test_coords = np.argwhere(gold_coords_grid.toarray()>0)
    xtest = test_coords[:, 0]
    ytest = test_coords[:, 1]
    
    gold_labels = bcc_pred[np.ravel_multi_index((xtest, ytest), dims=(nx, ny)), 1]
    gold_density = gold_labels
    
#     if os.path.exists(outputdir + "results.npy") and RELOAD_GOLD:
#         logging.info("LOADING GOLD DATA")
#         results = np.load(outputdir + "results.npy").item()[Nreports]
#         density_results = np.load(outputdir + "density_results.npy").item()[Nreports]
#     else:
#         logging.info("RUNNING METHODS WITH ALL LABELS TO ESTIMATE GOLD")
#         tester = prediction_tests.Tester(outputdir, methods, Nreports, z0, alpha0, nu0, shape_s0, rate_s0, 
#                 ls, optimise=False, verbose=False)            
#         tester.run_tests(C, nx, ny, xtest, ytest, [], [], Nreports, 1)
#         tester.save_separate_results()
#         
#         results = tester.results_all[Nreports]
#         density_results = tester.densityresults_all[Nreports]
#     
#     # find indices where all methods agree to within 0.1
#     for i, m in enumerate(results):
#         if i==0:
#             mlabels = np.zeros((len(results[m]), len(results)))
#             mdensity = np.zeros((len(results[m]), len(results)))
#         mlabels[:, i] = results[m].flatten()
#         mdensity[:, i] = density_results[m].flatten()
#         
#     gold_labels = np.mean(mlabels, axis=1)
#     gold_density = np.mean(mdensity, axis=1)     
#     
#     print "No. neg gold labels = %i" % np.sum(gold_labels<0.5)
#     
#     gold_tol = 0.1
#     
#     for i in range(mlabels.shape[1]):       
#         # check whether mlabels agree: gold labels are within tolerance, densities are within tolerance, and gold labels
#         # have not flipped to the other side of 0.5.
#         disagree_idxs = (np.abs(mlabels[:, i] - gold_labels) > gold_tol) | \
#             (np.abs(mdensity[:, i] - gold_density) > gold_tol) | ((gold_labels>0.5) != (mlabels[:, i]>0.5)) \
#                         | ((mlabels[:, i] > 0.4) & (mlabels[:, i] < 0.6))
#             # We could also include only locations with confident classifications, but this biases away from places with
#             # uncertain densities. Alternative is to select only when confident + there are reports. This might help
#             # with AUC estimates -- but the models are not really confident enough to evaluate AUCs. 
#         gold_labels[disagree_idxs] = - 1
#         gold_density[disagree_idxs] = - 1
#     
#     print "No. neg gold labels after disagreements removed = %i" % np.sum(gold_labels<0.5)
#         
#     goldidxs = gold_labels != - 1
#     xtest = xtest[goldidxs]
#     ytest= ytest[goldidxs]
#     gold_labels = gold_labels[goldidxs]
#     gold_density = gold_density[goldidxs]

    logging.info("NOW RUNNING TESTS")

    # Run the tests with the current dataset
    for d in range(nruns):
        
        # random selection of 1000 test points
        ntestd = 1000
        selection = np.random.choice(len(xtestgrid), ntestd, replace=False)
        xtest_d = xtestgrid[selection]
        ytest_d = ytestgrid[selection]
        
        xtest_d = np.concatenate((xtest, xtest_d)).astype(int)
        ytest_d = np.concatenate((ytest, ytest_d)).astype(int)
        
        gold_d = np.concatenate((gold_labels.flatten(), np.zeros(ntestd)-1))
        golddensity_d = np.concatenate((gold_density.flatten(), np.zeros(ntestd)-1))
        
        # for each run, use different subsets of labels
        shuffle_idxs = np.random.permutation(C.shape[0])
        C = C[shuffle_idxs, :]
        
        dataset_label = "d%i" % (d)
        outputdir, _ = dataset_location(expt_label_template, dataset_label)         
        
        # Run the tests with the current data set
        tester = Tester(outputdir, methods, Nreports, z0, alpha0, nu0, shape_s0, rate_s0, 
                ls, optimise=False, verbose=False)            
                
        tester.run_tests(C, nx, ny, xtest_d, ytest_d, gold_d, golddensity_d, Nreps_initial, Nrep_inc)
#         tester.reevaluate(C, nx, ny, xtest_d, ytest_d, gold_d, golddensity_d, Nreps_initial, Nrep_inc)

        tester.save_separate_results()
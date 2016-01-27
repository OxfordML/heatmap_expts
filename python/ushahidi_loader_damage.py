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

nruns = 50 # can use different random subsets of the reports C

# number of labels in first iteration dataset
Nreps_initial = 10
# increment the number of labels at each iteration
Nrep_inc = 10

def load_data(nx=500, ny=500):
    #Load up some ground truth
    datadir = './data/'

    goldfile_grid = datadir + 'haiti_unosat_target_grid%i_%i.npy' % (nx, ny)
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
       
    datahandler = UshahidiDataHandler(nx,ny, './data/')
    datahandler.discrete = False # do not snap-to-grid
    datahandler.load_data()
    # Could see how the methods vary with number of messages, i.e. when the messages come in according to the real
    # time line. Can IBCC do better early on?
    C = datahandler.C[1]
    # Put the reports into grid squares
    C[:, 1] = np.round(C[:, 1])
    C[:, 2] = np.round(C[:, 2])
    # number of available data points - LIMIT THE NUMBER OF REPORTS
    Nreports =  500 # C.shape[0]
    
    return C, Nreports, nx, ny, gold_labels, gold_density    

if __name__ == '__main__':
    print "Run tests to determine the accuracy of the bccheatmaps method."

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    methods = [
               'KDE',
               'IBCC',
               'GP',
               'IBCC+GP',
               'HeatmapBCC'
               ]

    #LOAD THE DATA to run unsupervised learning tests ------------------------------------------------------------------
    C, Nreports, nx, ny, gold_labels, gold_density = load_data()

    # Test at the grid squares only
    xtest = np.arange(nx)[np.newaxis, :]
    xtest = np.tile(xtest, (ny, 1))
    xtest = xtest.flatten()
     
    ytest = np.arange(ny)[:, np.newaxis]
    ytest = np.tile(ytest, (1, nx))
    ytest = ytest.flatten()
    
    # default hyper-parameters
    alpha0 = np.array([[60.0, 1.0], [1.0, 60.0]])[:,:,np.newaxis]
    alpha0 = np.tile(alpha0, (1,1,3))
    # set stronger priors for more meaningful categories
    alpha0[:,:,1] = np.array([[100.0, 1.0], [1.0, 100.0]]) # confident agents
    alpha0[:,:,2] = np.array([[1.0, 1.0], [1.0, 1.0]]) # agents with no prior knowledge of correlations
    clusteridxs_all = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0])
    alpha0_all = alpha0
    # set an uninformative prior over the spatial GP
    nu0 = np.array([100.0, 50.0])
    z0 = nu0[1] / np.sum(nu0)

    #shape_s0 = 10
    #rate_s0 = 4
    shape_s0 = np.sum(nu0) / 2.0 #2.0
    from scipy.stats import beta
    rho_samples = beta.rvs(nu0[1], nu0[0], size=50000)
    f_var = np.var(np.log(rho_samples / (1 - rho_samples)))
    rate_s0 =  0.5 * shape_s0 * f_var
    print "Rate s0 = %f" % rate_s0 
    
    ls = 16.0

    # Run the tests with the current dataset
    for d in range(nruns):
        
        # random selection of 1000 test points
        ntestd = 1000
        selection = np.random.choice(len(xtest), ntestd, replace=False)
        xtest_d = xtest[selection]
        ytest_d = ytest[selection]
        
        # for each run, use different subsets of labels
        shuffle_idxs = np.random.permutation(C.shape[0])
        C = C[shuffle_idxs, :]
        C_d = C[:Nreports, :] # limit number of reports!
        
#         # next, generate some simulated reports from the ground truth
#         new_agent = len(clusteridxs_all) - 2
#         new_agent_idxs = np.random.randint(0, 2, size=500)
#         
#         locs_pos = gold_labels >= 0.5
#         locs_pos = np.argwhere(locs_pos)[np.random.choice(len(locs_pos), 250, replace=False), :]
#         locs_neg = gold_labels < 0.5
#         locs_neg = np.argwhere(locs_neg)[np.random.choice(len(locs_neg), 250, replace=False), :]
#         Csim = np.zeros((500, 4))
#         Csim[:, 0] = new_agent + new_agent_idxs
#         Csim[:250, 1:3] = locs_pos 
#         Csim[250:, 1:3] = locs_neg
#         
#         accuracies = np.array([0, 1])
#         Csim[:250, 3] = np.random.rand(250) < accuracies[Csim[:250, 0].astype(int) - new_agent]
#         accuracies = np.array([1, 1])
#         Csim[250:, 3] = np.random.rand(250) > accuracies[Csim[:250, 0].astype(int) - new_agent] 
#         
#         C_d = np.concatenate((C_d, Csim), axis=0)
#         Nreports_plus_sim = C_d.shape[0]
        
        dataset_label = "d%i" % (d)
        outputdir, _ = dataset_location(expt_label_template, dataset_label)         
        
        # Run the tests with the current data set
        tester = prediction_tests.Tester(outputdir, methods, Nreports, z0, alpha0, nu0, shape_s0, rate_s0, 
                ls, optimise=False, verbose=False)            
                
        tester.run_tests(C_d, nx, ny, xtest_d, ytest_d, gold_labels.flatten()[selection], 
                                                    gold_density.flatten()[selection], Nreps_initial, Nrep_inc)
        tester.save_separate_results()
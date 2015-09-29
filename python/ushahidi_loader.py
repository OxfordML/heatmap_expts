__author__ = 'edwin'

import logging
import numpy as np
from ushahididata import UshahidiDataHandler
import prediction_tests

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
#     goldfile = "/home/edwin/Datasets/haiti_unosat/haiti_unosat_target_182_188_726_720_100_100.csv"
#     tgrid = np.genfromtxt(goldfile).astype(int)
    datadir = './data/'# '/home/edwin/Datasets/haiti_unosat/'
    goldfile = datadir + 'haiti_unosat_target_list.npy' #"./data/haiti_unosat_target3.npy"
    targets_all = np.load(goldfile)

    labels = {}
    targetsx = {}
    targetsy = {}

    # SUBSET FOR TESTING - whole dataset is too long (~300,000 data points). Create 10 subsets.
    for testsubset in range(10):
        testidxs = np.random.randint(targets_all.shape[0], size=1000)
        targets = targets_all[testidxs,:]

        targetsx[testsubset] = targets[:,0]
        targetsy[testsubset] = targets[:,1]
        labels[testsubset] = targets[:,2].astype(int)

    goldfile_grid = datadir + 'haiti_unosat_target_grid.npy'
    gold_density = np.load(goldfile_grid).astype(int)

    nx = 100
    ny = 100
    datahandler = UshahidiDataHandler(nx,ny, './data/')
    datahandler.discrete = False # do not snap-to-grid
    datahandler.load_data()
    # Need to see how the methods vary with number of messages, i.e. when the messages come in according to the real
    # time line. Can IBCC do better early on?
    C = datahandler.C[1]
    C_all = C # save for later

    K = datahandler.K
    # default hyper-parameters
    # default hyper-parameters
    alpha0 = np.array([[3.0, 1.0], [1.0, 3.0]])[:,:,np.newaxis]
    alpha0 = np.tile(alpha0, (1,1,3))
    # set stronger priors for more meaningful categories
    alpha0[:,:,1] = np.array([[5.0,1.0],[1.0,5.0]]) # confident agents
    alpha0[:,:,2] = np.array([[2.0,1.0],[1.0,2.0]]) # agents with no prior knowledge of correlations
    clusteridxs_all = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2, 0, 0, 2, 2])
    alpha0_all = alpha0
    # set an uninformative prior over the spatial GP
    nu0 = np.array([2.0, 1.0])
    z0 = nu0[1] / np.sum(nu0)
#
# #     #subset for testing
#     C = C[1:100,:]
#     # alpha0 = alpha0[:, :, np.sort(np.unique(C[:,0].astype(int)))]
#     K = len(np.unique(C[:,0]))

    # number of labels in first iteration dataset
    Ninitial_labels = 100
    # increment the number of labels at each iteration
    Nlabel_increment = 100

    # number of available data points
    navailable =  C.shape[0]

    # Run the tests with the current dataset
    outputdir = "./data/output/ush_"
    heatmapcombiner, gpgrid, gpgrid2, ibcc_combiner = prediction_tests.run_tests(K, C, nx, ny, z0, alpha0,
                                                        clusteridxs_all, alpha0_all, nu0, labels, targetsx, targetsy,
                                                        gold_density, navailable, Ninitial_labels, Nlabel_increment, outputdir, methods)
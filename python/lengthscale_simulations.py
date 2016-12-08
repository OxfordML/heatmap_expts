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

import numpy as np
from gpgrid import GPGrid
from scipy.stats import beta
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn, bernoulli

from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import Matern

from sklearn.model_selection import KFold

lsrange = np.array([1, 2, 10, 100])#np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) / float(10) * ls[0] * 2   

nx = 100
ny = 1
Nreports = 200
Ntest = nx * ny
ls = 10
ls = [ls, ls]
output_scale = 0.1
experiment_label = "lengthscale_sim_expt"
dataset_label = "lengthscale_sim_data"

J = 2
S = 1
nReliable = 7
diags = np.ones(S)
diags[:nReliable] = 2000
diags[nReliable:] = 1
off_diags = np.ones(S)
off_diags[:nReliable] = 1
off_diags[nReliable:] = 1
biases = np.ones((S, J))
biases[:nReliable, :] = 0
biases[nReliable:, :] = 0

def matern_3_2(xvals, yvals, ls):
    Kx = np.abs(xvals) * 3**0.5 / ls[0]
    Kx = (1 + Kx) * np.exp(-Kx)
    Ky = np.abs(yvals) * 3**0.5 / ls[1]
    Ky = (1 + Ky) * np.exp(-Ky)
    K = Kx * Ky
    return K

def sq_exp_cov(xvals, yvals, ls):
    Kx = np.exp( -xvals**2 / ls[0] )
    Ky = np.exp( -yvals**2 / ls[1] )
    K = Kx * Ky
    return K

def sigmoid(f):
    g = 1/(1+np.exp(-f))
    return g

def logit(g):
    f = -np.log(1/g - 1)
    return f

# Use the examples from Sklearn, e.g. the example given by plot_gpc.
# What is different in our dataset?
# Why is a short length-scale favoured on this dataset?
# Why does the lower bound on the VB implementation not seem to give good results?

def gen_synth_ground_truth(nx, ny, Nreports, ls, output_scale=1):
    '''
    Generate data.
    '''
    
    # create all x and y points
    x_all = np.arange(nx)[np.newaxis, :]
    x_all = np.tile(x_all, (ny, 1))
    y_all = np.arange(ny)[:, np.newaxis]
    y_all = np.tile(y_all, (1, nx))
    
    x_all = x_all.flatten()
    y_all = y_all.flatten()
    
    N = x_all.size
       
    ddx = x_all - x_all.T
    ddy = y_all - y_all.T

    #K = sq_exp_cov(ddx, ddy, ls)
    K = matern_3_2(ddx, ddy, ls)
    f_mean = np.zeros(N)
    f_all = mvn.rvs(mean=f_mean, cov=K * output_scale)

    # generate ground truth
    rho_all = sigmoid(f_all) # class density
    t_all = bernoulli.rvs(rho_all)

    print t_all

    return N, x_all, y_all, f_all, t_all

def split_data(N, Ntest):
    grididxs = np.arange(N)
    # select random points in the grid to test
    testidxs = np.random.choice(grididxs, Ntest, replace=False)
    trainidxs = np.ones(N)
    trainidxs[testidxs] = 0
    trainidxs = np.argwhere(trainidxs).flatten()
    
    return trainidxs, testidxs

def gen_synth_reports(N, Nreports, diags, off_diags, bias_vector, x_all, y_all, t_all, S=1):

    alpha0 = np.zeros((J, 2, S))
    for s in range(S):
        alpha0[:, :, s] += off_diags[s]
        alpha0[range(J), range(J), s] = diags[s]
        alpha0[:, :, s] += bias_vector[s:s+1, :] # adds diags fixed number of counts to each row so it biases toward
        #same answer regardless of ground truth

    #generate confusion matrices
    pi = np.zeros((J, 2, S))
    for s in range(S):
        for j in range(J):
            pi[j, 0, s] = beta.rvs(alpha0[j, 0, s], alpha0[j, 1, s])
            pi[j, 1, s] = 1 - pi[j, 0, s]
        print "Confusion matrix for worker %i: %s" % (s, str(pi[:, :, s]))
    # generate reports -- get the correct bit of the conf matrix
    pi_reps = pi[:, 1, :].reshape(J, S)
    reporter_ids = np.random.randint(0, S, (Nreports))[:, np.newaxis]

    dataidxreports = np.random.choice(N, Nreports, replace=True)
    xreports = x_all[dataidxreports][:, np.newaxis]
    yreports = y_all[dataidxreports][:, np.newaxis]

    pi_reps = pi_reps[t_all[dataidxreports], reporter_ids.flatten()]
    reports = bernoulli.rvs(pi_reps)[:, np.newaxis]
    print "Fraction of positive reports: %.3f" % (np.sum(reports) / float(len(reports)))

    C = np.concatenate((reporter_ids, xreports, yreports, reports), axis=1)
    
    return C, reporter_ids, reports, pi, xreports, yreports, dataidxreports

def nlpd_beta(gold, est_mean, est_var):
    '''
    This should be the same as cross entropy. Gives the negative log probability density of a ground-truth density value
    according to a beta distribution with given mean and variance.
    '''
    a_plus_b = (1.0 / est_var) * est_mean * (1-est_mean) - 1
    a = est_mean * a_plus_b
    b = (1-est_mean) * a_plus_b
    
    # gold density will break if it's actually set to zero or one.
    minval = 1e-6
    gold[gold > 1.0 - minval] = 1.0 - minval
    gold[gold < minval] = minval
        
    a[a<minval] = minval
    b[b<minval] = minval
    
    nlpd = np.sum(- beta.logpdf(gold, a, b)) 
    return nlpd / len(gold) # we return the mean per data point

def test_gp(ls_i, train, test):
    trainreps = np.in1d(dataidxreports, train)
    xrep_train = xreports[trainreps, :]
    yrep_train = yreports[trainreps, :]
    rep_train = reports[trainreps]

    x_test = x_all[test]
    y_test = y_all[test]
    
    shape_s0 = 1000000.0
    rate_s0 = shape_s0 * output_scale
    shape_ls = 2.0
    rate_ls = 2.0 / ls[0]
    gpgrid = GPGrid(dims=(nx, ny), z0=0.5, shape_s0=shape_s0, rate_s0=rate_s0, shape_ls=shape_ls, rate_ls=rate_ls)
    train_coords = np.concatenate((xrep_train.astype(int), yrep_train.astype(int)), axis=1)
    gpgrid.fit(train_coords, rep_train)
    test_coords = np.concatenate((x_test[:, np.newaxis], y_test[:, np.newaxis]), axis=1)
    gp_preds, _ = gpgrid.predict(test_coords, variance_method='sample')
    lb = gpgrid.lowerbound()
    
    return np.round(gp_preds), gp_preds, lb

def test_sklearn_gp(ls_i, train, test):
    trainreps = np.in1d(dataidxreports, train)
    xrep_train = xreports[trainreps]
    yrep_train = yreports[trainreps]
    rep_train = reports[trainreps]
    
    x_test = x_all[test][:, np.newaxis]
    y_test = y_all[test][:, np.newaxis]
    
    kernel = Matern(length_scale=ls_i)
    gpc = GPC(kernel=kernel, warm_start=False, n_restarts_optimizer=0, optimizer=None)
    trainX = np.concatenate((xrep_train, yrep_train), axis=1)
    gpc.fit(trainX, rep_train)
    testX = np.concatenate((x_test, y_test), axis=1)
    preds = gpc.predict(testX)
    probs = gpc.predict_proba(testX)[:, 1]
    score = gpc.log_marginal_likelihood()
    return preds, probs, score

if __name__ == '__main__':    
#     N, x_all, y_all, f_all, t_all = gen_synth_ground_truth(nx, ny, Nreports, ls, output_scale)
    N = nx
    x_all = np.arange(nx)
    y_all = np.zeros(nx)
    f_all = np.zeros(nx)
    f_all[:nx/2] = -10
    f_all[nx/2:] = 10
    rho_all = sigmoid(f_all)
    t_all = bernoulli.rvs(rho_all)
    
    C, reporter_ids, reports, pi, xreports, yreports, dataidxreports = gen_synth_reports(
                                    N, Nreports, diags, off_diags, biases, x_all, y_all, t_all, S)

    sortidxs = np.argsort(xreports.flatten())
    xreports = xreports[sortidxs]
    yreports = yreports[sortidxs]
    reports = reports[sortidxs]

    dataidxs = np.arange(N)
    
    kfold_random_state = np.random.randint(0, 100)
    kf = KFold(n_splits=10, shuffle=True, random_state = kfold_random_state)
    
    lb_results = np.zeros(len(lsrange))
    acc_results = np.zeros(len(lsrange))
    #nlpd_results = np.zeros(len(lsrange))
    roc_results = np.zeros(len(lsrange))
    for i, ls_i in enumerate(lsrange): 
        t_pred = np.zeros(N)
        rho_mean = np.zeros(N)
        
        # reset the shuffling
        kf.random_state = kfold_random_state
        for train, test in kf.split(dataidxs):     
        
            t_pred[test], rho_mean[test], lb_k = test_gp(ls_i, train, test)
            lb_results[i] += lb_k
            
        #nlpd_results[i] = nlpd_beta(rho_test, rho_mean, rho_var)
        acc_results[i] = accuracy_score(t_all, t_pred)
        roc_results[i] = roc_auc_score(t_all, rho_mean)
            
        #print "Cross entropy between densities: %.2f" % nlpd_results[i]
        print "Accuracy: %.2f" % acc_results[i] 

    p1 = plt.figure()
    plt.plot(lsrange, lb_results, label='VB')
    plt.title('Lower bound')
    
    p2 = plt.figure()
    plt.plot(lsrange, acc_results, label='VB')
    plt.title('Accuracy')
    
    p3 = plt.figure()
    plt.plot(lsrange, roc_results, label='VB')
    plt.title('ROC AUC')
    
    #plt.figure()
    #plt.plot(lsrange, nlpd_results)
    #plt.title('NLPD')
    
    lb_results = np.zeros(len(lsrange))
    acc_results = np.zeros(len(lsrange))
    #nlpd_results = np.zeros(len(lsrange))
    roc_results = np.zeros(len(lsrange))
    
    for i, ls_i in enumerate(lsrange): 
        t_pred = np.zeros(N)
        rho_mean = np.zeros(N)
        
        # reset the shuffling
        kf.random_state = kfold_random_state        
        for train, test in kf.split(dataidxs):        
        
            t_pred[test], rho_mean[test], lb_k = test_sklearn_gp(ls_i, train, test)
            lb_results[i] += lb_k
            
        #nlpd_results[i] = nlpd_beta(rho_test, rho_mean, rho_var)
        acc_results[i] = accuracy_score(t_all, t_pred)
        roc_results[i] = roc_auc_score(t_all, rho_mean)
            
        #print "Cross entropy between densities: %.2f" % nlpd_results[i]
        print "Accuracy: %.2f" % acc_results[i] 

    p1 = plt.figure(p1.number)
    plt.plot(lsrange, lb_results, label='Laplace')
    plt.title('Lower bound')
    plt.legend(loc='best')
    
    p2 = plt.figure(p2.number)
    plt.plot(lsrange, acc_results, label='Laplace')
    plt.title('Accuracy')
    plt.legend(loc='best')
    
    p3 = plt.figure(p3.number)
    plt.plot(lsrange, roc_results, label='Laplace')
    plt.title('ROC AUC')
    plt.legend(loc='best')
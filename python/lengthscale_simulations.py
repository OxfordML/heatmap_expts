'''
Created on 6 Dec 2016

@author: edwin
'''
from heatmapbcc import HeatMapBCC

"""
Experiment 2a. VARYING SPARSENESS: Generate noisy reports with a mixture of reliabilities (different levels of biased workers).
These should be uniformly distributed through the space.
When we call prediction_tests, we will evaluate with decreasing sparseness in each iteration. 
The sparseness can be due to either the spacing of the reports or the total number of reports from each worker, 
both of which will affect the reliability model of the workers and the GP. We are more interested in the former, as 
standard IBCC deals with the latter. 

Experiment 2b: Possible alternative view to plot. Keep a fixed sparseness, but different numbers of biased workers.


TO CHECK

3. LB seems wrong on HeatmapBCC? Trying with stronger alpha. Try viewing f with a fixed length scale -- this shows an
unexpected function shape. It should be basically identical to the GP without BCC, since the worker is 100% reliable.
To solve this, let's visualise the f function at each iteration of the heatmapbcc vb loop. 
First step -- rerunning without a fixed number of G updates per iteration. Perhaps this was the problem. 

4. Permit Output scale to vary
5. Different data: noisy labels. How well do we model the information source reliability?
6. New ground truth.

COMPLETE-ISH
1. This seems okay on a simple test dataset -- check on larger/different data. Does the VB GB still work now we have removed the terms that should have cancelled out? D-tr(K^-1C) = D - invK_expecF
2. What is going on with heatmapBCC convergence? == fixed to remove checking of fractions and use lower bound in this expt

"""

import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
from gpgrid import GPGrid
from scipy.stats import beta
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn, bernoulli

from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import Matern, RBF

from sklearn.model_selection import KFold

theta0 = np.logspace(0, 8, 30)
theta1 = np.logspace(-1, 1, 29)
#Theta0, Theta1 = np.meshgrid(theta0, theta1)

lsrange = theta1#np.array([0.01, 0.1, 1, 10, 100])#np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) / float(10) * ls[0] * 2   
loglsrange = np.log10(lsrange)

nx = 100
ny = 1
Nreports = 25
Ntest = nx * ny
ls = 10
ls = [ls, ls]
output_scale = 10000
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

def test_vb_heatmapbcc(ls_i, train, test, vm='rough'):
    trainreps = np.in1d(dataidxreports, train)
    xrep_train = xreports[trainreps, :]
    yrep_train = yreports[trainreps, :]
    rep_train = reports[trainreps, :]

    x_test = x_all[test]
    y_test = y_all[test]
    
    shape_s0 = 1000000000.0
    rate_s0 = shape_s0 * output_scale
    shape_ls = 2.0
    rate_ls = 2.0 / ls_i
    
    alpha0 = np.array([[10000, 0.1], [0.1, 10000]])
    hbcc = HeatMapBCC(nx, ny, nclasses=2, nscores=2, alpha0=alpha0, K=1, z0=0.5, 
                        shape_s0=shape_s0, rate_s0=rate_s0, shape_ls=shape_ls, rate_ls=rate_ls)
    #hbcc.conv_threshold_G = 1e-6
    hbcc.conv_threshold = 1e-6
    hbcc.conv_check_freq = 1
    hbcc.verbose = True
    hbcc.max_iterations = 500
    hbcc.uselowerbound = True
    
    train_coords = np.concatenate((xrep_train, yrep_train), axis=1)
    crowdlabels = np.concatenate((np.zeros((np.sum(trainreps), 1)), train_coords, rep_train), axis=1)
    hbcc.combine_classifications(crowdlabels)
    preds, rho, _ = hbcc.predict(x_test[:, np.newaxis], y_test[:, np.newaxis], variance_method=vm)
    lb = hbcc.lowerbound()
    
    rho = rho[1, :]
    
    return np.round(preds), rho, lb, hbcc.heatGP[1].obs_f

def test_vb_gp(ls_i, train, test, vm='sample'):
    trainreps = np.in1d(dataidxreports, train)
    xrep_train = xreports[trainreps, :]
    yrep_train = yreports[trainreps, :]
    rep_train = reports[trainreps]

    x_test = x_all[test]
    y_test = y_all[test]
    
    shape_s0 = 1000000000.0
    rate_s0 = shape_s0 * output_scale
    shape_ls = 2.0
    rate_ls = 2.0 / ls_i
    gpgrid = GPGrid(dims=(nx, ny), z0=0.5, shape_s0=shape_s0, rate_s0=rate_s0, shape_ls=shape_ls, rate_ls=rate_ls)
    gpgrid.conv_threshold_G = 1e-6
    gpgrid.conv_threshold = 1e-6
    gpgrid.conv_check_freq = 1
    gpgrid.verbose = True
    
    train_coords = np.concatenate((xrep_train, yrep_train), axis=1)
    gpgrid.fit(train_coords, rep_train)
    test_coords = np.concatenate((x_test[:, np.newaxis], y_test[:, np.newaxis]), axis=1)
    gp_preds, _ = gpgrid.predict(test_coords, variance_method=vm)
    lb, dll, _, _, _, _ = gpgrid.lowerbound(True)
    
    return np.round(gp_preds), gp_preds, lb, gpgrid.obs_f, dll

def test_sklearn_gp(ls_i, train, test):
    trainreps = np.in1d(dataidxreports, train)
    xrep_train = xreports[trainreps]
    yrep_train = yreports[trainreps]
    rep_train = reports[trainreps]
    
    x_test = x_all[test][:, np.newaxis]
    y_test = y_all[test][:, np.newaxis]
    
    #kernel1 = RBF(length_scale=ls_i)#length_scale_bounds=[ls_i, ls_i])#Matern(length_scale=ls_i)
    kernel1 = Matern(length_scale=ls_i)
    gpc = GPC(kernel=output_scale * kernel1, optimizer=None)#, warm_start=False, n_restarts_optimizer=0, optimizer=None)
    trainX = np.concatenate((xrep_train, yrep_train), axis=1)
    gpc.fit(trainX, rep_train.flatten())
    score = gpc.log_marginal_likelihood(np.log([output_scale, ls_i]))#[ls_i, 1])
    
    logging.debug('Kernel: %s, score=%.3f' % (gpc.kernel_, score))
    
    testX = np.concatenate((x_test, y_test), axis=1)
    preds = gpc.predict(testX)
    probs = gpc.predict_proba(testX)[:, 1]
        
    # Specify Gaussian Processes with fixed and optimized hyperparameters
    # gp_fix = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0),
    #                                    optimizer=None)
    # gp_fix.fit(X[:train_size], y[:train_size])    
    # gp_fix.log_marginal_likelihood(np.log([Theta0[i, j], Theta1[i, j]]))    
    return preds, probs, score, gpc.base_estimator_.f_cached

if __name__ == '__main__':    
#     N, x_all, y_all, f_all, t_all = gen_synth_ground_truth(nx, ny, Nreports, ls, output_scale)
    N = nx
#     x_all = np.arange(nx)
#     y_all = np.zeros(nx)
# #     f_all = np.zeros(nx)
# #     f_all[:nx/2] = -1
# #     f_all[nx/2:] = 1
# #     rho_all = sigmoid(f_all / np.sqrt(output_scale))
# #     t_all = bernoulli.rvs(rho_all)
#     t_all = np.zeros(nx, dtype=int)
#     t_all[:nx/2] = 0
#     t_all[nx/2:] = 1
    
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 5, nx)
    y = np.array(X > 2.5, dtype=int)
    
    x_all = X
    y_all = np.zeros(nx)
    t_all = y    
    
#     C, reporter_ids, reports, pi, xreports, yreports, dataidxreports = gen_synth_reports(
#                                     N, Nreports, diags, off_diags, biases, x_all, y_all, t_all, S)

    dataidxreports = np.argsort(np.argsort(x_all))[:, np.newaxis]
    xreports = x_all[:, np.newaxis]#[dataidxreports]
    yreports = y_all[:, np.newaxis]#[dataidxreports]
    reports = t_all[:, np.newaxis]#[dataidxreports]
    Nreports = len(xreports)

    sortidxs = np.argsort(dataidxreports.flatten())
    dataidxreports = dataidxreports[sortidxs]
    xreports = xreports[sortidxs]
    yreports = yreports[sortidxs]
    reports = reports[sortidxs]

    dataidxs = np.arange(N)
    
    kfold_random_state = np.random.randint(0, 100)
    kf = KFold(n_splits=10, shuffle=True, random_state = kfold_random_state)
    
    lb_results = np.zeros(len(lsrange), dtype=float)
    dll_results = np.zeros(len(lsrange), dtype=float)
    acc_results = np.zeros(len(lsrange), dtype=float)
    #nlpd_results = np.zeros(len(lsrange))
    roc_results = np.zeros(len(lsrange), dtype=float)
    
    for i, ls_i in enumerate(lsrange): 
        t_pred = np.zeros(N)
        rho_mean = np.zeros(N)
        f_train = np.zeros(Nreports) # inferred latent function values at the training locations
        
        # reset the shuffling
        kf.random_state = kfold_random_state
        for train, test in kf.split(dataidxs):     
        
            t_pred[test], rho_mean[test], lb_k, f_train[np.in1d(dataidxreports, train)], dll_k = test_vb_gp(ls_i, train, test)
            lb_results[i] += lb_k
            dll_results[i] += dll_k
        
        if np.sum(lb_results[i] >= lb_results[:i]) == i:
            chosen_rho_mean = rho_mean
            chosen_ls = ls_i
            chosen_t_pred = t_pred 
            chosen_f_train = f_train           
        
        #nlpd_results[i] = nlpd_beta(rho_test, rho_mean, rho_var)
        acc_results[i] = accuracy_score(t_all, t_pred)
        roc_results[i] = roc_auc_score(t_all, rho_mean)
            
        #print "Cross entropy between densities: %.2f" % nlpd_results[i]
        print "Accuracy: %.2f" % acc_results[i] 

    lb_results /= kf.n_splits
    dll_results /= kf.n_splits  

    p1 = plt.figure()
    plt.plot(loglsrange, lb_results, label='GPC VB', color='b')
    plt.scatter(loglsrange[lsrange==chosen_ls], lb_results[lsrange==chosen_ls], marker='x', label='GPC VB optimal length scale', color='b')
    plt.title('Lower bound')
    
    p2 = plt.figure()
    plt.plot(loglsrange, acc_results, label='GPC VB', color='b')
    plt.title('Accuracy')
    
    p3 = plt.figure()
    plt.plot(loglsrange, roc_results, label='GPC VB', color='b')
    plt.title('ROC AUC')
    
    p4 = plt.figure()
    sorteddataidxs = np.argsort(x_all)[:, np.newaxis]
    plt.plot(x_all[sorteddataidxs], chosen_rho_mean[sorteddataidxs], label='GPC VB, ls=%.2f' % chosen_ls, color='b')
    plt.title('rho and t')
    
#     plt.scatter(x_all, chosen_t_pred, color='b', marker='o')
    
    p5 = plt.figure()
    plt.plot(xreports, chosen_f_train, label='GPC VB, ls=%.2f' % chosen_ls, color='b')
    plt.title('f at the training points')
        
#     p1 = plt.figure(p1.number)
#     plt.plot(loglsrange, dll_results, label='GPC VB DLL', color='b', linestyle='dashed')
#     plt.title('Lower bound')
        
    #plt.figure()
    #plt.plot(lsrange, nlpd_results)
    #plt.title('NLPD')
    
    lb_results = np.zeros(len(lsrange), dtype=float)
    dll_results = np.zeros(len(lsrange), dtype=float)
    acc_results = np.zeros(len(lsrange), dtype=float)
    #nlpd_results = np.zeros(len(lsrange))
    roc_results = np.zeros(len(lsrange), dtype=float)
    
    for i, ls_i in enumerate(lsrange): 
        t_pred = np.zeros(N)
        rho_mean = np.zeros(N)
        f_train = np.zeros(Nreports) # inferred latent function values at the training locations
        
        # reset the shuffling
        kf.random_state = kfold_random_state
        for train, test in kf.split(dataidxs):     
        
            t_pred[test], rho_mean[test], lb_k, f_train[np.in1d(dataidxreports, train)], dll_k = test_vb_gp(
                                                                                     ls_i, train, test, vm='rough')
            lb_results[i] += lb_k
            dll_results[i] += dll_k      
        
        if np.sum(lb_results[i] >= lb_results[:i]) == i:
            chosen_rho_mean = rho_mean
            chosen_ls = ls_i
            chosen_t_pred = t_pred 
            chosen_f_train = f_train           
        
        #nlpd_results[i] = nlpd_beta(rho_test, rho_mean, rho_var)
        acc_results[i] = accuracy_score(t_all, t_pred)
        roc_results[i] = roc_auc_score(t_all, rho_mean)
            
        #print "Cross entropy between densities: %.2f" % nlpd_results[i]
        print "Accuracy: %.2f" % acc_results[i]     
    
    lb_results /= kf.n_splits
    dll_results /= kf.n_splits  
    
#     p1 = plt.figure(p1.number)
#     plt.plot(loglsrange, lb_results, label='GPC VB, analytical pred.', color='g')
#     plt.scatter(loglsrange[lsrange==chosen_ls], lb_results[lsrange==chosen_ls], marker='x', 
#                 label='GPC VB, analytical pred., optimal length scale', color='g')    
#     plt.title('Lower bound')
#     plt.legend(loc='best')
    
    p2 = plt.figure(p2.number)
    plt.plot(loglsrange, acc_results, label='GPC VB, analytical pred.', color='g')
    plt.title('Accuracy')
    plt.legend(loc='best')
    plt.ylim(-0.01, 1.01)
    
    p3 = plt.figure(p3.number)
    plt.plot(loglsrange, roc_results, label='GPC VB, analytical pred.', color='g')
    plt.title('ROC AUC')
    plt.legend(loc='best')
    plt.ylim(-0.01, 1.01)
    
    p4 = plt.figure(p4.number)
    sorteddataidxs = np.argsort(x_all)[:, np.newaxis]
    plt.plot(x_all[sorteddataidxs], chosen_rho_mean[sorteddataidxs], label='GPC VB, analytical pred., ls=%.2f' % chosen_ls,
              color='g')
    plt.title('rho and t')
    plt.legend(loc='best')
    
#     plt.scatter(x_all, chosen_t_pred, color='g', marker='x')
#     plt.scatter(x_all, t_all, color='black', marker='*')
    
#     p5 = plt.figure(p5.number)
#     plt.plot(xreports, chosen_f_train, label='GPC VB, analytical pred., ls=%.2f' % chosen_ls, color='g')
#     plt.title('f at the training points')    
#     plt.legend(loc='best')    
    
#     p1 = plt.figure(p1.number)
#     plt.plot(loglsrange, dll_results, label='GPC VB, analytical pred., DLL', color='g', linestyle='dashed')
#     plt.title('Lower bound')    
 
    lb_results = np.zeros(len(lsrange), dtype=float)
    dll_results = np.zeros(len(lsrange), dtype=float)
    acc_results = np.zeros(len(lsrange), dtype=float)
    #nlpd_results = np.zeros(len(lsrange))
    roc_results = np.zeros(len(lsrange), dtype=float)
    
    for i, ls_i in enumerate(lsrange): 
        t_pred = np.zeros(N)
        rho_mean = np.zeros(N)
        f_train = np.zeros(Nreports) # inferred latent function values at the training locations
         
        # reset the shuffling
        kf.random_state = kfold_random_state
        for train, test in kf.split(dataidxs):     
         
            t_pred[test], rho_mean[test], lb_k, f_train[np.in1d(dataidxreports, train)] = test_vb_heatmapbcc(
                                                                                     ls_i, train, test, vm='rough')
            lb_results[i] += lb_k
         
        if (ls_i < 1.65) and (ls_i > 1.63): #np.sum(lb_results[i] >= lb_results[:i]) == i:
            chosen_rho_mean = rho_mean
            chosen_ls = ls_i
            chosen_t_pred = t_pred 
            chosen_f_train = f_train           
         
        #nlpd_results[i] = nlpd_beta(rho_test, rho_mean, rho_var)
        acc_results[i] = accuracy_score(t_all, t_pred)
        roc_results[i] = roc_auc_score(t_all, rho_mean)
             
        #print "Cross entropy between densities: %.2f" % nlpd_results[i]
        print "Accuracy: %.2f" % acc_results[i]     
     
    lb_results /= kf.n_splits
    dll_results /= kf.n_splits  
     
    p1 = plt.figure(p1.number)
    plt.plot(loglsrange, lb_results, label='HeatmapBCC VB, analytical pred.', color='k')
    plt.scatter(loglsrange[lsrange==chosen_ls], lb_results[lsrange==chosen_ls], marker='x', 
                label='HeatmapBCC VB, analytical pred., optimal length scale', color='k')
    plt.title('Lower bound')
    plt.legend(loc='best')
     
    p2 = plt.figure(p2.number)
    plt.plot(loglsrange, acc_results, label='HeatmapBCC VB, analytical pred.', color='k')
    plt.title('Accuracy')
    plt.legend(loc='best')
    plt.ylim(-0.01, 1.01)
     
    p3 = plt.figure(p3.number)
    plt.plot(loglsrange, roc_results, label='HeatmapBCC VB, analytical pred.', color='k')
    plt.title('ROC AUC')
    plt.legend(loc='best')
    plt.ylim(-0.01, 1.01)
     
    p4 = plt.figure(p4.number)
    sorteddataidxs = np.argsort(x_all)[:, np.newaxis]
    plt.plot(x_all[sorteddataidxs], chosen_rho_mean[sorteddataidxs], label='HeatmapBCC VB, analytical pred., ls=%.2f' % chosen_ls,
              color='k')
    plt.title('rho and t')
    plt.legend(loc='best')
     
    p5 = plt.figure(p5.number)
    plt.plot(xreports, chosen_f_train, label='HeatmapBCC VB, analytical pred., ls=%.2f' % chosen_ls, color='k')
    plt.title('f at the training points')    
    plt.legend(loc='best')    
     
#     p1 = plt.figure(p1.number)
#     plt.plot(loglsrange, dll_results, label='HeatmapBCC VB, analytical pred., DLL', color='g', linestyle='dashed')
#     plt.title('Lower bound')     
    
    lb_results = np.zeros(len(lsrange), dtype=float)
    acc_results = np.zeros(len(lsrange), dtype=float)
    #nlpd_results = np.zeros(len(lsrange))
    roc_results = np.zeros(len(lsrange), dtype=float)
    
    for i, ls_i in enumerate(lsrange): 
        t_pred = np.zeros(N)
        rho_mean = np.zeros(N)
        f_train = np.zeros(Nreports)
        
        # reset the shuffling
        kf.random_state = kfold_random_state        
        for train, test in kf.split(dataidxs):        
        
            t_pred[test], rho_mean[test], lb_k, f_train[np.in1d(dataidxreports, train)] = test_sklearn_gp(ls_i, train, test)
            lb_results[i] += lb_k
        
        if np.sum(lb_results[i] >= lb_results[:i]) == i:
            chosen_t_pred = t_pred
            chosen_rho_mean = rho_mean
            chosen_ls = ls_i            
            chosen_f_train = f_train                        
                        
        #nlpd_results[i] = nlpd_beta(rho_test, rho_mean, rho_var)
        acc_results[i] = accuracy_score(t_all, t_pred)
        roc_results[i] = roc_auc_score(t_all, rho_mean)
            
        #print "Cross entropy between densities: %.2f" % nlpd_results[i]
        print "Accuracy: %.2f" % acc_results[i] 

    lb_results /= kf.n_splits

    p1 = plt.figure(p1.number)
    plt.plot(loglsrange, lb_results, label='GPC Laplace', color='r')
    plt.scatter(loglsrange[lsrange==chosen_ls], lb_results[lsrange==chosen_ls], marker='x', 
                label='GPC Laplace optimal length scale', color='r')    
    plt.title('Lower bound')
    plt.legend(loc='best')
    
    p2 = plt.figure(p2.number)
    plt.plot(loglsrange, acc_results, label='GPC Laplace', color='r')
    plt.title('Accuracy')
    plt.legend(loc='best')
    plt.ylim(-0.01, 1.01)
    
    p3 = plt.figure(p3.number)
    plt.plot(loglsrange, roc_results, label='GPC Laplace', color='r')
    plt.title('ROC AUC')
    plt.legend(loc='best')
    plt.ylim(-0.01, 1.01)
    
    p4 = plt.figure(p4.number)
    sorteddataidxs = np.argsort(x_all)[:, np.newaxis]
    plt.plot(x_all[sorteddataidxs], chosen_rho_mean[sorteddataidxs], label='GPC Laplace, ls=%.2f' % chosen_ls, color='r')
    plt.title('rho and t')
    plt.legend(loc='best')
    
#     plt.scatter(x_all, chosen_t_pred, color='r', marker='x')
    plt.scatter(x_all, t_all, color='black', marker='*', label='ground truth')
    
    p5 = plt.figure(p5.number)
    plt.plot(xreports, chosen_f_train, label='GPC Laplace, ls=%.2f' % chosen_ls, color='r')
    plt.title('f at the training points')    
    plt.legend(loc='best')
    
    outputpath = './output/lengthscale_plots/plot_gpc/%s'
    plt.figure(p1.number)
    plt.savefig(outputpath % 'lowerbound.eps')
    plt.figure(p2.number)
    plt.savefig(outputpath % 'accuracy.eps')
    plt.figure(p3.number)
    plt.savefig(outputpath % 'auc.eps')
    plt.figure(p4.number)
    plt.savefig(outputpath % 'rho.eps')
    plt.figure(p5.number)
    plt.savefig(outputpath % 'f.eps')            
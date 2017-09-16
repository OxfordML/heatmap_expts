'''
Tests using synthetic data to show that it is possible to learn the length-scale using HeatmapBCC, GPClassifierVB or 
the SKLearn implementation.

TODO: why does the plot of f look wrong? Is the order of the predictions wrong?
TODO: test with noisy reports

Created on 6 Dec 2016

@author: edwin
'''
from heatmapbcc import HeatMapBCC

import logging
logging.basicConfig(level=logging.DEBUG)
import os
import numpy as np
from gp_classifier_vb import GPClassifierVB
from scipy.stats import beta
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn, bernoulli
#import alan
from run_synthetic_case_studies import plot_heatmap
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold

lsrange = np.logspace(-1, 2, 20)#29)
loglsrange = np.log10(lsrange)

nx = 10
ny = 10
Nreports = 50
ls = 10
ls = [ls, ls]
output_scale = 3

J = 2
S = 5
nReliable = 3
diags = np.ones(S)
diags[:nReliable] = 10
diags[nReliable:] = 1
off_diags = np.ones(S)
off_diags[:nReliable] = 1
off_diags[nReliable:] = 1
biases = np.ones((S, J))
biases[:nReliable, :] = 0.5
biases[nReliable:, :] = 0.5

methods = ['heatmapbcc', 'gbvb', 'gplaplace'] #'gpvbanal', 

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

def gen_synth_ground_truth(nx, ny, ls, output_scale=1):
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

    return N, x_all, y_all, f_all, t_all, rho_all

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
    
    shape_s0 = 2.0
    rate_s0 = shape_s0
    shape_ls = 2.0
    rate_ls = 2.0 / ls_i
    
    alpha0 = np.array([[2, 1], [1, 2]]) #[1, 0.5], [0.5, 1]])
    hbcc = HeatMapBCC(nx, ny, nclasses=2, nscores=2, alpha0=alpha0, K=1, z0=0.5, 
                        shape_s0=shape_s0, rate_s0=rate_s0, shape_ls=shape_ls, rate_ls=rate_ls)
    #hbcc.conv_threshold_G = 1e-6
    hbcc.conv_threshold = 1e-6
    hbcc.conv_check_freq = 1
    hbcc.verbose = True
    hbcc.max_iterations = 20
    hbcc.uselowerbound = True
    
    train_coords = np.concatenate((xrep_train, yrep_train), axis=1)
    crowdlabels = np.concatenate((np.zeros((np.sum(trainreps), 1)), train_coords, rep_train), axis=1)
    hbcc.combine_classifications(crowdlabels)
    preds, rho, _ = hbcc.predict(x_test[:, np.newaxis], y_test[:, np.newaxis], variance_method=vm)
    lb = hbcc.lowerbound()
    
    rho = rho[1, :]
    
    f_train, _ = hbcc.heatGP[1].predict_f((xrep_train, yrep_train))
    return np.round(preds[1, :]), rho, lb, f_train.flatten()

def test_vb_gp(ls_i, train, test, vm='sample'):
    trainreps = np.in1d(dataidxreports, train)
    xrep_train = xreports[trainreps, :]
    yrep_train = yreports[trainreps, :]
    rep_train = reports[trainreps]

    x_test = x_all[test]
    y_test = y_all[test]
    
    shape_s0 = 2.0
    rate_s0 = shape_s0
    shape_ls = 2.0
    rate_ls = 2.0 / ls_i
    gpgrid = GPClassifierVB(2, z0=0.5, shape_s0=shape_s0, rate_s0=rate_s0, shape_ls=shape_ls, rate_ls=rate_ls)
    gpgrid.conv_threshold_G = 1e-6
    gpgrid.conv_threshold = 1e-6
    gpgrid.conv_check_freq = 1
    gpgrid.verbose = True
    
    train_coords = np.concatenate((xrep_train, yrep_train), axis=1)
    gpgrid.fit(train_coords, rep_train)
    test_coords = np.concatenate((x_test[:, np.newaxis], y_test[:, np.newaxis]), axis=1)
    gp_preds, _ = gpgrid.predict(test_coords, variance_method=vm)
    lb, dll, _, _, _, _ = gpgrid.lowerbound(True)
    
    f_train, _ = gpgrid.predict_f((xrep_train, yrep_train))
    
    gp_preds = gp_preds.flatten()
    
    return np.round(gp_preds), gp_preds, lb, f_train.flatten(), dll

def test_sklearn_gp(ls_i, train, test):
    trainreps = np.in1d(dataidxreports, train)
    xrep_train = xreports[trainreps]
    yrep_train = yreports[trainreps]
    rep_train = reports[trainreps]
    
    x_test = x_all[test][:, np.newaxis]
    y_test = y_all[test][:, np.newaxis]
    
    #kernel1 = RBF(length_scale=ls_i)#length_scale_bounds=[ls_i, ls_i])#Matern(length_scale=ls_i)
    kernel1 = ConstantKernel(1.0) * Matern(length_scale=ls_i, length_scale_bounds="fixed")
    gpc = GPC(kernel=kernel1)#, warm_start=False, n_restarts_optimizer=0, optimizer=None)
    trainX = np.concatenate((xrep_train, yrep_train), axis=1)
    gpc.fit(trainX, rep_train.flatten())
    score = gpc.log_marginal_likelihood()
    
    logging.debug('Kernel: %s, score=%.3f' % (gpc.kernel_, score))
    
    testX = np.concatenate((x_test, y_test), axis=1)
    preds = gpc.predict(testX)
    probs = gpc.predict_proba(testX)[:, 1]
        
    return preds, probs, score, gpc.base_estimator_.f_cached

def plot_results(methodlabel, nmethods, methodidx, col, figures, lb_results, kf, chosen_f_train, chosen_rho_mean, 
                 acc_results, roc_results): 
    lb_results /= kf.n_splits
     
    p1 = plt.figure(figures[0].number)
    plt.plot(loglsrange, lb_results, label=methodlabel, color=col)
    plt.scatter(loglsrange[lsrange==chosen_ls], lb_results[lsrange==chosen_ls], marker='x', color=col)#,
#                 label='%s, optimal length scale' % methodlabel, )
    plt.xlabel('log_10(lengthscale)')
    plt.ylabel('estimated ln(marginal likelihood)')
    plt.title('Lower bound')
    plt.legend(loc='best')
     
    p2 = plt.figure(figures[1].number)
    plt.plot(loglsrange, acc_results, label=methodlabel, color=col)
    plt.scatter(loglsrange[lsrange==chosen_ls], acc_results[lsrange==chosen_ls], marker='x', color=col)
    plt.xlabel('log_10(lengthscale)')
    plt.ylabel('accuracy')        
    plt.title('Accuracy')
    plt.legend(loc='best')
    plt.ylim(-0.01, 1.01)
     
    p3 = plt.figure(figures[2].number)
    plt.plot(loglsrange, roc_results, label=methodlabel, color=col)
    plt.scatter(loglsrange[lsrange==chosen_ls], roc_results[lsrange==chosen_ls], marker='x', color=col)    
    plt.title('ROC AUC')
    plt.xlabel('log_10(lengthscale)')
    plt.ylabel('ROC AUC')        
    plt.legend(loc='best')
    plt.ylim(-0.01, 1.01)
     
    p4 = plt.figure(figures[3].number)
    if experiment_name == '2D_toy' or experiment_name=='2D_toy_noisyworkers':
        ax = p4.add_subplot(np.ceil(nmethods/2.0), 2, methodidx)
        #alan.plot_density(nx, ny, x_all, y_all, chosen_rho_mean, ax=ax)
        plot_heatmap(nx, ny, x_all, y_all, chosen_rho_mean, methodlabel, fig=ax, colorbar_on=False)
    else:
        plt.subplot(np.ceil(nmethods/2.0), 2, methodidx)
        sorteddataidxs = np.argsort(x_all)[:, np.newaxis]
        plt.plot(x_all[sorteddataidxs], chosen_rho_mean[sorteddataidxs], label='%s, ls=%.2f' % (methodlabel, chosen_ls),
              color=col)
    plt.xlabel('input coordinate')
    plt.ylabel('p(+ve label)')        
    plt.legend(loc='best')
     
    p5 = plt.figure(figures[4].number)
    if experiment_name == '2D_toy' or experiment_name=='2D_toy_noisyworkers':
        ax = p5.add_subplot(np.ceil(nmethods/2.0), 2, methodidx)
        #alan.plot_density(nx, ny, x_all, y_all, chosen_f_train, ax=ax)
        plot_heatmap(nx, ny, xreports, yreports, chosen_f_train, methodlabel, fig=ax, colorbar_on=False)
    else:
        plt.subplot(np.ceil(nmethods/2.0), 2, methodidx)        
        plt.plot(xreports, chosen_f_train, label='%s, ls=%.2f' % (methodlabel, chosen_ls), color=col)
    plt.xlabel('input coordinate')
     
    plt.legend(loc='best')
    
    return p1, p2, p3, p4, p5

if __name__ == '__main__':
    # select one of these experiment names to generate appropriate data
    experiment_name = '2D_toy'
#     experiment_name = 'one_changepoint'
# experiment_name = '2D_toy_noisyworkers'

    # Generate the ground truth first
    if experiment_name == '2D_toy' or experiment_name == '2D_toy_noisyworkers':    
        N, x_all, y_all, f_all, t_all, rho_all = gen_synth_ground_truth(nx, ny, ls, output_scale)
    elif experiment_name == 'one_changepoint':
        # create a very simple dataset where X values larger than 2.5 have positive y values, those lower than 2.5 have 0s
        nx = 100
        ny = 1
        N = nx
        rng = np.random.RandomState(0)
        x_all = rng.uniform(0, 5, nx)
        y_all = np.zeros(nx)
        f_all = np.array(x_all > 2.5, dtype=int)
        # now introduce a little noise    
        rho_all = sigmoid((f_all - 0.5)*2)
        t_all = bernoulli.rvs(rho_all)
         
    # generate the reports
    if experiment_name == '2D_toy_noisyworkers':
        C, _, reports, _, xreports, yreports, dataidxreports = gen_synth_reports(N, Nreports, diags, 
                                                                            off_diags, biases, x_all, y_all, t_all, S)
    else: # otherwise assume we have one perfect trainer
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
    
    p1 = plt.figure()
    p2 = plt.figure()
    p3 = plt.figure()
    p4 = plt.figure()
    plt.title(r'$\rho$') # needs an 'r' in front of string to treat as raw text so that \r is not an escape sequence    
    p5 = plt.figure()
    plt.title('f at the training points')
    figures = [p1, p2, p3, p4, p5]
    
# GPVB ----------------------------------------------------------------------------------------------------------------
    if 'gbvb' in methods:
        
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
                trainreps = np.in1d(dataidxreports, train)
                t_pred[test], rho_mean[test], lb_k, f_train[trainreps], dll_k = test_vb_gp(ls_i, train, test)
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
         
        p1, p2, p3, p4, p5 = plot_results('GPC VB', len(methods) + 1, 4, 'b', figures, lb_results, kf, chosen_f_train, 
                                          chosen_rho_mean, acc_results, roc_results)        
            
# GPVB with analytical approximation -----------------------------------------------------------------------------------
    if 'gpvbanal' in methods: #use analytical 'rough' approximation instead of sampling to estimate output expected value
        
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
                trainreps = np.in1d(dataidxreports, train)  
                t_pred[test], rho_mean[test], lb_k, f_train[trainreps], dll_k = test_vb_gp(ls_i, train, test, vm='rough')
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
                 
        p1, p2, p3, p4, p5 = plot_results('GPC VB, analytical pred.', len(methods) + 1, 5, 'y', figures, lb_results, kf, chosen_f_train, 
                                          chosen_rho_mean, acc_results, roc_results)   
        
# HeatmapBCC -----------------------------------------------------------------------------------------------------------
    if 'heatmapbcc' in methods:
     
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
                trainreps = np.in1d(dataidxreports, train)
                t_pred[test], rho_mean[test], lb_k, f_train[trainreps] = test_vb_heatmapbcc(
                                                                                         ls_i, train, test, vm='rough')
                lb_results[i] += lb_k
             
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
         
        p1, p2, p3, p4, p5 = plot_results('HeatmapBCC VB', len(methods) + 1, 2, 'g', figures, lb_results, kf, 
                                          chosen_f_train, chosen_rho_mean, acc_results, roc_results)
         
# GP Laplace -----------------------------------------------------------------------------------------------------------
    if 'gplaplace' in methods:
        
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
                trainreps = np.in1d(dataidxreports, train)
                t_pred[test], rho_mean[test], lb_k, f_train[trainreps] = test_sklearn_gp(ls_i, train, test)
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
    
        p1, p2, p3, p4, p5 = plot_results('GPC Laplace', len(methods) + 1, 3, 'r', figures, lb_results, kf, chosen_f_train, chosen_rho_mean, 
                                          acc_results, roc_results)


    #ground truth
    plt.figure(p4.number)
    if experiment_name == '2D_toy' or experiment_name=='2D_toy_noisyworkers':
        ax = p4.add_subplot(np.ceil((len(methods)+1)/2.0), 2, 1)
        plot_heatmap(nx, ny, x_all, y_all, t_all, 'ground truth class labels', fig=ax)

#         ax.scatter(x_all[t_all==1], y_all[t_all==1], np.ones(np.sum(t_all==1)), color='black', marker='x', 
#                     label='ground truth -- positive values')
#         ax.scatter(x_all[t_all==0], y_all[t_all==0], np.zeros(np.sum(t_all==0)), color='black', marker='o', 
#                     label='ground truth -- negative values')
    else:
        plt.subplot(np.ceil((len(methods)+1)/2.0), 2, 1)
        plt.scatter(x_all, t_all, color='black', marker='*', label='ground truth')
        
    plt.legend(loc='best')
            
    p5 = plt.figure(p5.number)
    if experiment_name == '2D_toy' or experiment_name=='2D_toy_noisyworkers':
        ax = p5.add_subplot(np.ceil((len(methods)+1)/2.0), 2, 1)
        #alan.plot_density(nx, ny, x_all, y_all, f_all, ax=ax)
        plot_heatmap(nx, ny, x_all, y_all, f_all, 'ground truth latent f', fig=ax)
    else:        
        plt.subplot(np.ceil((len(methods)+1)/2.0), 2, 1)        
        plt.plot(x_all, f_all, label='ground truth latent function', color='k')        
        
    plt.legend(loc='best')
    
    outputpath = './output/lengthscale_plots/'
    if not os.path.isdir(outputpath):
        os.mkdir(outputpath)
    outputpath += '%s' % experiment_name
    if not os.path.isdir(outputpath):
        os.mkdir(outputpath)
    outputpath += '/%s'
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
    
    plt.show()     
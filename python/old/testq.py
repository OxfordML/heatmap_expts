'''
Created on 11 Feb 2016

@author: edwin
'''

from gpgrid import GPGrid
import numpy as np
from scipy.stats import norm

from gpgrid import sigmoid

if __name__ == '__main__':
    
    mu0 = 0
    s = 0.5/5
    
    f_samples = norm.rvs(loc=mu0, scale=np.sqrt(1.0/s), size=50000)
    rho_samples = sigmoid(f_samples)
    rho_mean = np.mean(rho_samples)
    rho_var = np.var(rho_samples - rho_mean)
    # find the beta parameters
    a_plus_b = (1.0 / rho_var) * (rho_mean*(1 - rho_mean)) - 1
    a = a_plus_b * rho_mean
    b = a_plus_b * (1 - rho_mean)
    
    gp = GPGrid(1, 1, rho_mean, 1.0, 1.0/s, s, 1, 1, np.array([1, 1]))
    
    l = 1
    x = np.zeros(l)#np.array([0, 0, 0, 0, 0])
    y = np.zeros(l)#array([0, 0, 0, 0, 0])
    obs = np.ones(l)#np.array([1, 1, 1, 1, 1])
    gp.fit((x, y), obs, update_s = True)
    
    mean, var = gp.predict((np.array([0]), np.array([0])), variance_method='sample')
    
    a_plus_b2 = (1.0 / var) * (mean*(1 - mean)) - 1
    a2 = a_plus_b2 * mean
    b2 = a_plus_b2 * (1 - mean)
    
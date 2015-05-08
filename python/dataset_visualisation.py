'''
Created on 6 May 2015

@author: edwin
'''
import numpy as np
import matplotlib.pyplot as plt
from prediction_tests import UshahidiDataHandler as UDH

# Plot the distribution of gold labels

goldfile = "./data/haiti_unosat_target3.npy"
targets = np.load(goldfile).astype(int)
targets = targets[0:1000,:]
targetsx = targets[:,0]
targetsy = targets[:,1]
labels = targets[:,2]

plt.figure()
# Damage
didxs = labels==1
plt.scatter(targetsx[didxs], targetsy[didxs], c='red')
# No damage
ndidxs = labels==0
plt.scatter(targetsx[ndidxs], targetsy[ndidxs], c='blue')
plt.show()

# Plot the reports
dh = UDH(1000, 1000, './data/')
dh.load_ush_data()
C = dh.C[1]
C = C[0:200, :]

# plt.figure()
plt.scatter(C[:,1], C[:,2], c='yellow')
plt.show()
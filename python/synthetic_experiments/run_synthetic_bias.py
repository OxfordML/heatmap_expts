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
import gen_synthetic as gs

nruns = gs.nruns
Nreports = gs.Nreports
Nreps_initial = gs.Nreps_initial
Nrep_inc = gs.Nrep_inc
nsteps = gs.nsteps
weak_proportions = gs.weak_proportions

# EXPERIMENT CONFIG ---------------------------------------------------------------------------------------------------

# REPORTERS
diag_reliable = 10.0
off_diag_reliable = 1.0
bias_reliable = np.zeros(gs.J)

# For the unreliable workers to have bias
diag_weak = 2.0
off_diag_weak = 1.0
bias_weak = np.zeros(gs.J)
bias_weak[0] = 5.0

expt_label_template = 'synth/output_cluslocs%.2f_bias_grid17'
dataset_location = gs.dataset_location

if __name__ == '__main__':    
    gs.run_experiments(expt_label_template, snap_to_grid=True)#, 1, gs.nruns, 2, gs.nproportions)

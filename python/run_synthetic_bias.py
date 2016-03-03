"""
Experiment 2a. VARYING SPARSENESS: Generate noisy reports with a mixture of reliabilities (different levels of biased workers).
These should be uniformly distributed through the space.
When we call prediction_tests, we will evaluate with decreasing sparseness in each iteration. 
The sparseness can be due to either the spacing of the reports or the total number of reports from each worker, 
both of which will affect the reliability model of the workers and the GP. We are more interested in the former, as 
standard IBCC deals with the latter. 

Experiment 2b: Possible alternative view to plot. Keep a fixed sparseness, but different numbers of biased workers.

"""

import sys
from gen_synthetic import dataset_location

sys.path.append("/homes/49/edwin/robots_code/HeatMapBCC/python")
sys.path.append("/homes/49/edwin/robots_code/pyIBCC/python")

import numpy as np
import gen_synthetic as gs

nruns = gs.nruns
Nreports = gs.Nreports
Nreps_initial = gs.Nreps_initial
Nrep_inc = gs.Nrep_inc
nsteps = gs.nsteps
weak_proportions = gs.weak_proportions

# EXPERIMENT CONFIG ---------------------------------------------------------------------------------------------------

RESET_ALL_DATA = False
PLOT_SYNTH_DATA = False
SAVE_RESULTS = True

# REPORTERS
diag_reliable = 10.0
off_diag_reliable = 1.0
bias_reliable = np.zeros(gs.J)

# For the unreliable workers to have bias
diag_weak = 2.0
off_diag_weak = 1.0
bias_weak = np.zeros(gs.J)
bias_weak[0] = 5.0

expt_label_template = 'synth/output_cluslocs%.2f_bias_grid5'

# Flag indicates whether we should snap report locations to their nearest grid location.
# Doing so means that we assume reports relate to the whole grid square, and that different sources relate to the same
# t object. We could reduce problems with this discretization step if we use soft snapping based on distance.
# When set to true, the model predicts the state of each grid location, and the latent density of states. Lots of 
# reports at same place does not necessarily imply high density, which makes sense if there is only a single emergency. 
# When set to false, the model predicts the density of reports at each location, if the reports were accurate,
# and assumes that reports may relate to different events at the same location.   
snap_to_grid = True

if __name__ == '__main__':    
    gs.run_experiments(expt_label_template, 1, gs.nruns, 2, gs.nproportions)

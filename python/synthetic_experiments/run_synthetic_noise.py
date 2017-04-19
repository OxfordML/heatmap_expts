"""
Experiment 1a. VARYING SPARSENESS: Generate noisy reports with a mixture of reliabilities (different levels of noisy workers).
These should be uniformly distributed through the space.
When we call prediction_tests, we will evaluate with decreasing sparseness in each iteration.
The sparseness can be due to either the spacing of the reports or the total number of reports from each worker,
both of which will affect the reliability model of the workers and the GP. We are more interested in the former, as
standard IBCC deals with the latter.

Experiment 1b: Possible alternative view to plot. Keep a fixed sparseness, but different noise levels.

Possible alternative setup is to keep same number of reliable workers, but just add noisy workers (this would increase
total number of labels though).

"""

import sys

sys.path.append("/homes/49/edwin/robots_code/HeatMapBCC/python")
sys.path.append("/homes/49/edwin/robots_code/pyIBCC/python")

import numpy as np
import gen_synthetic as gs
#from gen_synthetic import dataset_location

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

diag_reliable = 10.0
off_diag_reliable = 1.0
bias_reliable = np.zeros(gs.J)

# For the unreliable workers to have white noise
diag_weak = 5.0
off_diag_weak = 5.0
bias_weak = np.zeros(gs.J)

#expt_label_template = 'synth/output_cluslocs%.2f_noise_free5'
expt_label_template = 'synth/output_cluslocs%.2f_noise_grid5'

if __name__ == '__main__':
    gs.run_experiments(expt_label_template)#, 0, gs.nruns, 2, gs.nproportions)

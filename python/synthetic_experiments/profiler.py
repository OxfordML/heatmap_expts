import cProfile

import numpy as np
import gen_synthetic as gs

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

expt_label_template = 'synth/output_cluslocs%.2f_noise_free_profilertest'

command = 'gs.run_experiments(expt_label_template, dend=1)'
cProfile.run( command, filename="../gen_synthetic.profile") 
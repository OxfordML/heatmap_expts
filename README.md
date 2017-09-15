# heatmap_expts

This project also depends on:

   * scipy, numpy, matplotlib
   * https://github.com/edwinrobots/HeatMapBCC
   * https://github.com/edwinrobots/pyIBCC 
   
You will need to add the 'python' subdirectory of each project to the Python path.
   
The experiments in the paper can be run as follows. For synthetic data:

   * synthetic_experiments/run_synthetic_noise.py
   * synthetic_experiments/run_synthetic_noise_nogrid.py
   * synthetic_experiments/run_synthetic_bias.py
   
Then run plot_performance.py to plot the performance metrics. These will be saved in the 
output/synth folder.

For the PRN satellite image dataset:
   * prn_experiments/prn_simulation
   * plot_performance.py produces images in output/prn
   
For Ushahidi dataset:
   * ushahidi_experiments/ushahidi_loader_emergencies.py
   
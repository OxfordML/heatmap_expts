# heatmap_expts

This project also depends on:

   * scipy (tested with 0.19.1), scikit-learn (tested with 0.19.0), numpy (tested with 1.13), matplotlib (tested with 2.0.0)
   * https://github.com/edwinrobots/HeatMapBCC (master branch)
   * https://github.com/edwinrobots/pyIBCC (master branch)
   
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
   
For plotting heatmaps, from the repository HeatMapBCC run python/crowdscanner/ushahidiheatmap.py
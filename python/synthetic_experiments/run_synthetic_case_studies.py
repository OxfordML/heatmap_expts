''' 

Out of the things below, we focus on case study 4.

3. LENGTH SCALE COLLAPSE/STABILITY: what happens when data is heteroskedastic? -- drop this as it is a GP problem.
Large scale interpolation needed but in some areas we get confident, alternating reports. Do the alternating reports
cause length scale collapse leading to no interpolation? Does the use of nu0 < 1 affect this as it would lower noise
in observations of rho? 

4. CASE STUDY: INTERPOLATING BETWEEN GOOD WORKERS
Show an example where IBCC infers trusted workers, then interpolates between them, ignoring the reports from noisy 
workers. This should show for each method:
1. GP will weight all reporters equally, so will have errors where the noisy reporters are.
2. IBCC grid + GP will not infer the reliability as accurately and will interpolate more poorly if there is a gradual
change across a grid square.
Setup: 
1. Requires some clustering to discriminate between worker reliabilities. --> pick the middle cluster spread setting
2. Two sets of workers, biased and good. Biased workers will pull GP in the same direction, regardless of the ground
truth. --> pick weak proportion 0.5
3. Can test with lots of labels to show that this problem is not avoided by GP with lots of data. --> nReports==1000 
4. Plot ground truth (put red dots for biased workers?) and predictions from each method. Zoom in on an area where 
the biased workers are clustered with no or very few good workers.
5. Table with MCE for each method.  -- can be avoided if we simply pick out a sample from one of the experiment 1/2 tests.

5. CASE STUDY: TRUSTED REPORTER -- probably drop this, or show the simulation from ORCHID/Ushahidi.
Show what happens when we supply labels from a highly reliable worker, the changes should propagate. Can start 
with noisy labels only, then show what happens when trusted reports are introduced -- allows us to learn meaningful 
confusion matrices even if there is a large amount of noise. This might be better shown with Ushahidi data. 
This should show for each method:
1. GP stuck on very noisy/weak decisions.
2. IBCC grid + GP either requires very coarse grid, leading to poor interpolation, or very small grid so that fewer
reporters coincide, hence fewer reporters detected as reliable.
Set up: 
0. Test this with 20 reporters so that there are several good reporters to detect.
1. Take a case with 75% noisy workers where none of the methods do well.
2. Can use middle cluster spreading or no clustering.
3. nReports == 1000
4. Plot ground truth and initial predictions without trusted worker. Plot results when trusted worker is introduced.
5. Table for MCE for each method and for HeatmapBCC with and without the trusted worker.
6. May want to add a column to the table showing a comparison in MCE over a number of datasets, or a plot with 
different numbers of trusted reports. 

 
Created on 1 Mar 2016

@author: edwin
'''

# Experiment data folder to load from
# from run_synthetic_noise import expt_label_template
from run_synthetic_bias import expt_label_template
from gen_synthetic import cluster_spreads, dataset_location, Nreports, nx, ny, J, S, sigmoid, nruns, nproportions
cluster_spread = cluster_spreads[0]
experiment_label = expt_label_template % cluster_spread

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix

def plot_heatmap(nx, ny, x, y, t, title='density', transparency=0.0, fig=None, colorbar_on=True):

    cmap = plt.get_cmap('seismic')                
    cmap._init()
    cmap._lut[:,-1] = np.linspace(1-transparency, 1-transparency, cmap.N+3)   

    z_plot = np.zeros((nx, ny)) - 1
    z_plot[x.astype(int).reshape(-1), y.astype(int).reshape(-1)] = t
    
    if not fig:
        fig = plt.figure()

    cax = plt.imshow(z_plot, cmap=cmap, aspect=None, origin='lower', vmin=0, vmax=1, interpolation='none', filterrad=0.01)
    if colorbar_on:
        plt.colorbar(cax, orientation='horizontal', pad=0.05, shrink=0.9)
    
    #ax.set_zlim3d(0, 1)
    plt.title(title)
    
    return fig

# Plot the reports. Colour the workers according to their ground truth reliability.
def plot_reports(nx, ny, x, y, z, transparency=0.0, title='Reports', fig=None):
    cmap = plt.get_cmap('seismic')                
    cmap._init()
    cmap._lut[:,-1] = np.linspace(1-transparency, 1-transparency, cmap.N+3)
    
    if not fig:
        fig = plt.figure()
    z[z==0] = -1
    z_plot = coo_matrix((z.astype(int).flatten(), (x.astype(int).flatten(), y.astype(int).flatten())), shape=(nx, ny))
    z_plot = z_plot.toarray()

    cax = plt.imshow(z_plot, cmap=cmap, aspect=None, origin='lower', vmin=-5.0, vmax=5.0, interpolation='none', filterrad=0.01)
    plt.colorbar(cax, orientation='horizontal', pad=0.05, shrink=0.9)
    plt.title(title)
    
    return fig

def case_study_4(d, p_idx, nlabels_arr=[]):
    dataset_label = "d%i" % d
    _, datadir_targets = dataset_location(experiment_label, dataset_label)
    
    dataset_label = "p%i_d%i" % (p_idx, d)
    datadir_results, datadir_reports = dataset_location(experiment_label, dataset_label)
    
    # Load the data from the specified folder
    x_all = np.load(datadir_targets + "x_all.npy")
    y_all = np.load(datadir_targets + "y_all.npy")
    f_all = np.load(datadir_targets + "f_all.npy")
    r_all = sigmoid(f_all)
    
    xtest = x_all[Nreports:]
    ytest = y_all[Nreports:]
    rtest = r_all[Nreports:]
    
    C = np.load(datadir_reports + "C.npy")
    pi_all = np.load(datadir_reports + "pi_all.npy")
    pi_all = pi_all.reshape((J, J, S))
    
    xreps = C[:, 1]
    yreps = C[:, 2]
    reps = C[:, 3]

    #accs = np.mean(pi_all[range(J), range(J), :], axis=0)
    
    #results = np.load(datadir_results + "results.npy")
    densityresults = np.load(datadir_results + "density_results.npy")
    if not np.any(nlabels_arr):
        nlabels_arr = densityresults.item().keys()
    for nlabels in nlabels_arr:
        
        #t_pred = results.item()[nlabels]['HeatmapBCC']
        
        r_pred = densityresults.item()[nlabels]['HeatmapBCC'].flatten()
        
        # Plot the ground truth density or t function.
        fig = plt.figure()
        splt = plt.subplot(2, 3, 1)
        plot_heatmap(nx, ny, xtest, ytest, rtest, title="Ground Truth Density Function", fig=splt)
        
        splt = plt.subplot(2, 3, 2)
        plot_reports(nx, ny, xreps[:nlabels], yreps[:nlabels], reps[:nlabels], title='Histogram of Reports \n (positive label counts minus negative label counts)', fig=splt)
        
        # New figure. Plot the reports with inferred reliability.
        splt = plt.subplot(2, 3, 3)
        plot_heatmap(nx, ny, xtest, ytest, r_pred, title='HeatmapBCC Inferred Density', fig=splt)
        # Plot the inferred density/t function. 
                        
        # Plot for all other methods
        #plt.figure()
        nmethods = len(densityresults.item()[nlabels].keys()) - 2
        mcount = 0
        for m in densityresults.item()[nlabels].keys():
            if m == 'HeatmapBCC' or m == 'IBCC':
                continue
            else:
                mcount += 1
            
            r_pred = densityresults.item()[nlabels][m].flatten()
        
            splt = plt.subplot(2, nmethods, mcount + 3)
            plot_heatmap(nx, ny, xtest, ytest, r_pred, title='%s Inferred Density' % m, fig=splt)
        
        fig.set_figheight(14)
        fig.set_figwidth(20)
        
        fig.tight_layout(pad=0,w_pad=0,h_pad=0)
                
        # Save plot. Provide identifiers for which run it was. 
        plt.savefig(datadir_results + 'casestudy4_main_%i.pdf' % nlabels)
        
    

if __name__=='__main__':
    for p_idx in range(nproportions):
        for d in range(nruns):
            case_study_4(d, p_idx, [1000, 1500])
            plt.close('all')

    
#     case_study_4(d, 1, [1000, 1500])
'''
Created on 8 Jan 2015

@author: edwin


Notes/TO DO:

See if the data points have been extracted correctly. We currently have 294170 test points, which is too many -- let's 
run this one 5 different sub-samples if this is the correct number. Make sure we are using the original house locations,
not the discrete grid values.

'''

import numpy as np
import logging
from heatmapbcc import HeatMapBCC
from ibccperformance import Evaluator
from ibcc import IBCC
from gpgrid import GPGrid
from scipy.sparse import coo_matrix
from scipy.stats import gaussian_kde, kendalltau
from scipy.optimize import fmin#_cobyla,

results_all = {}
auc_all = {}
mce_all = {}
rmse_all = {}

auc_all_var = {}
mce_all_var = {}
rmse_all_var = {}

tau_all = {}
mced_all = {}
rmsed_all = {}

gpgrid = None
heatmapcombiner = None
gpgrid2 = None
ibcc_combiner = None

def run_tests(K, C_all, nx, ny, z0, alpha0, clusteridxs_all, alpha0_all, nu0, labels, targetsx, targetsy, gold_density,
              navailable, nlabels, stepsize, outputdir, methods):
    C = C_all

    print "No. report types = %i" % K

    # Initialize the optimal grid size for the separate IBCC method.
    opt_nx = float(nx)
    opt_ny = float(ny)

    #GPGRID OBJECT
    gpgrid = GPGrid(nx, ny, z0=z0, shape_ls=10, rate_ls=10.0/100)
    gpgrid.verbose = False

    #HEATMAPBCC OBJECT
    heatmapcombiner = HeatMapBCC(nx, ny, 2, 2, alpha0, K, z0=z0, shape_ls=10, rate_ls=10.0/100, force_update_all_points=True)
    heatmapcombiner.min_iterations = 5
    heatmapcombiner.max_iterations = 200
    heatmapcombiner.conv_threshold = 0.1
    heatmapcombiner.verbose = False

    nout = nx * ny
    gridoutputx = np.tile(np.arange(nx, dtype=np.float).reshape(nx, 1), (1, ny)).reshape(nout)
    gridoutputy = np.tile(np.arange(ny, dtype=np.float).reshape(1, ny), (nx, 1)).reshape(nout)

    while nlabels <= navailable:
        C = C_all[0:nlabels, :]
        agents = np.unique(C[:,0])
        K = int(np.max(agents)+1)
        clusteridxs = clusteridxs_all[0:K]
        alpha0 = alpha0_all

        # containers for results
        if not 'results' in globals():
            results = {}

        # indicator array to show whether reports are positive or negative
        posreports = (C[:, 3] == 1).astype(float)
        negreports = (C[:, 3] == 0).astype(float)

        # Report coords for this round
        reportsx = C[:, 1]
        reportsy = C[:, 2]

        # KERNEL DENSITY ESTIMATION ---------------------------------------------------------------------------------------
        if 'KDE' in methods:
            results['KDE'] = {}

            # Method used here performs automatic bandwidth determination - see help docs
            posinputdata = np.vstack((reportsx[posreports>0], reportsy[posreports>0]))
            neginputdata = np.vstack((reportsx[negreports>0], reportsy[negreports>0]))
            logging.info("Running KDE... pos data size: %i, neg data size: %i " % (posinputdata.shape[1], neginputdata.shape[1]) )
            kdepos = gaussian_kde(posinputdata)
            if neginputdata.shape[1] != 0:
                kdeneg = gaussian_kde(neginputdata)
            #else: # Treat the points with no observation as negatives
                # no_obs_coords = np.argwhere(obs_grid.toarray()==0)
                # neginputdata = np.vstack((no_obs_coords[:,0], no_obs_coords[:,1]))

            def kde_prediction(x, y):
                grid_lower = np.vstack((x - 0.5, y - 0.5))
                grid_upper = np.vstack((x + 0.5, y + 0.5))
                p_loc_giv_damage = np.zeros(len(x))
                p_loc_giv_nodamage = np.zeros(len(x))

                for i in range(len(x)):
                    if i%1000 == 0:
                        logging.debug("Processing %i of %i" % (i,len(x)))
                    p_loc_giv_damage[i] = kdepos.integrate_box(grid_lower[:, i], grid_upper[:, i])
                    if neginputdata.shape[1] != 0:
                        p_loc_giv_nodamage[i] = kdeneg.integrate_box(grid_lower[:, i], grid_upper[:, i])
                    else:
                        p_loc_giv_nodamage[i] = 1.0 / (nx*ny)

                p_damage_loc = p_loc_giv_damage * z0
                p_nodamage_loc = p_loc_giv_nodamage * (1.0-z0)
                p_damage_giv_loc  = p_damage_loc / (p_damage_loc + p_nodamage_loc)
                return p_damage_giv_loc

            # Repeat for each of the test datasets
            for ts in range(len(labels)):
                results['KDE'][ts] = kde_prediction(targetsx[ts], targetsy[ts])
            results['KDE']['grid'] = kde_prediction(gridoutputx, gridoutputy)

            logging.info("KDE complete.")

        # TRAIN GP WITHOUT BCC ---------------------------------------------------------------------------------------------
        if 'GP' in methods:
            logging.info("Using a density GP without BCC...")
            # Get values for training points by taking frequencies -- only one report at each location, so give 1 for
            # positive reports, 0 otherwise
            gpgrid.optimize([reportsx, reportsy],
                            np.concatenate((posreports[:,np.newaxis]*0.8, (posreports+negreports)[:,np.newaxis]), axis=1))

            results['Train_GP_on_Freq'] = {}
            for ts in range(len(labels)):
                gp_preds = gpgrid.predict([targetsx[ts], targetsy[ts]])
                results['Train_GP_on_Freq'][ts] = gp_preds
            results['Train_GP_on_Freq']['grid'] = gpgrid.predict([gridoutputx, gridoutputy]).reshape(nx, ny)

        # RUN SEPARATE IBCC AND GP STAGES ----------------------------------------------------------------------------------
        if 'IBCC+GP' in methods:
            # Should this be dropped from the experiments, as it won't work without gridding the points? --> then run into
            # question of grid size etc. Choose grid so that squares have an average of 3 reports?
            logging.info("Running separate IBCC and GP...")

            # run standard IBCC
            ibcc_combiner = IBCC(2, 2, alpha0, nu0, K)
            ibcc_combiner.clusteridxs_alpha0 = clusteridxs
            ibcc_combiner.verbose = False
            ibcc_combiner.min_iterations = 5
            ibcc_combiner.max_iterations = 200
            ibcc_combiner.conv_threshold = 0.1

            #initialise a GP
            initialguess = [opt_nx, opt_ny]
            # noinspection PyTypeChecker
            constraints = [lambda hp: 1 if np.all(np.asarray(hp[0:2]) >= 1) else -1, lambda hp: 1 if not np.isnan(hp[0]) and hp[0] - int(hp[0]) == 0 else -1,
                           lambda hp: 1 if not np.isnan(hp[1]) and hp[1] - int(hp[1]) == 0 else -1]
            gpgrid2 = GPGrid(opt_nx, opt_ny, z0=z0, shape_ls=10, rate_ls=10.0/100)
            # noinspection PyTypeChecker
            def train_gp_on_ibcc_output(hyperparams):
                logging.debug("fmin gridx and gridy values: %f, %f" % (hyperparams[0], hyperparams[1]))
                opt_nx = int(hyperparams[0])
                opt_ny = int(hyperparams[1])
                if opt_nx < 0 or opt_ny < 0 or np.isnan(opt_nx) or np.isnan(opt_ny):
                    return np.inf
                gpgrid2.nx = opt_nx
                gpgrid2.ny = opt_ny

                # grid indicating where we made observations
                reportsx_grid = (reportsx * opt_nx/nx).astype(int)
                reportsy_grid = (reportsy * opt_ny/ny).astype(int)

                obs_grid = coo_matrix((np.ones(reportsx.shape[0]), (reportsx_grid, reportsy_grid)), (opt_nx, opt_ny))
                # Coordinates where we made observations, i.e. without duplicates due to multiple reports at some points
                obs_coords = np.argwhere(obs_grid.toarray()>0)
                obsx = obs_coords[:, 0]
                obsy = obs_coords[:, 1]

                #flatten the input data so it can be used with standard IBCC
                linearIdxs = np.ravel_multi_index((reportsx_grid, reportsy_grid), dims=(opt_nx, opt_ny))
                C_flat = C[:,[0,1,3]]
                C_flat[:,1] = linearIdxs
                bcc_pred = ibcc_combiner.combine_classifications(C_flat, optimise_hyperparams=False)
                bcc_pred = bcc_pred[np.ravel_multi_index((obsx, obsy), dims=(opt_nx, opt_ny)), 1]

                # use IBCC output to train GP
                _, ls = gpgrid2.optimize([obsx, obsy], bcc_pred)
                logging.debug("fmin param value for lengthscale: %f" % ls[0])
                nlml = gpgrid2.neg_marginal_likelihood(np.log(ls))
                return nlml

            opt_hyperparams = fmin(train_gp_on_ibcc_output, initialguess, maxfun=200,
                                                     full_output=False, xtol=10, ftol=0.5)
            #fmin_cobyla(train_gp_on_ibcc_output, initialguess, constraints, rhobeg=500, rhoend=100)
            # use optimized grid size to make predictions
            results['IBCC_then_GP'] = {}

            for ts in range(len(labels)):
                gp_preds = gpgrid2.predict([targetsx[ts], targetsy[ts]])
                results['IBCC_then_GP'][ts] = gp_preds
            gp_preds = gpgrid2.predict([gridoutputx, gridoutputy])
            results['IBCC_then_GP']['grid'] = gp_preds.reshape(nx, ny)

        # RUN HEAT MAP BCC --------------------------------------------------------------------------------------------------
        if 'HeatmapBCC' in methods:
            logging.info("Running HeatmapBCC...")
            # to do:
            # make sure optimise works
            # make sure the optimal hyper-parameters are passed to the next iteration
            heatmapcombiner.clusteridxs_alpha0 = clusteridxs
            heatmapcombiner.combine_classifications(C, optimise_hyperparams=True)

            results['heatmapbcc'] = {}
            for ts in range(len(labels)):
                bcc_pred = heatmapcombiner.predict(targetsx[ts], targetsy[ts])
                bcc_pred = bcc_pred[1, :] # only interested in positive "damage class"
                results['heatmapbcc'][ts] = bcc_pred

            bcc_pred = heatmapcombiner.predict_grid()
            results['heatmapbcc']['grid'] = bcc_pred[1, :]

        # EVALUATE ALL RESULTS ---------------------------------------------------------------------------------------------
        evaluator = Evaluator("", "BCCHeatmaps", "Ushahidi_Haiti_Building_Damage")

        gold_density = gold_density.flatten()

        for method in results:
            print 'Results for %s with %i labels' % (method, nlabels)
            pred = results[method]

            mce = []
            rmse = []
            auc = []
            mced = []
            rmsed = []
            tau = []
            for ts in range(len(labels)):
                testresults = pred[ts].flatten()

                mce_ts = - np.sum(labels[ts] * np.log(testresults)) - np.sum(labels[ts] * np.log(1 - testresults))
                mce_ts = mce_ts / float(len(labels[ts]))
                mce.append( mce_ts )

                rmse.append( np.sqrt( np.sum((testresults - labels[ts])**2) / float(len(labels[ts])) ) )

                auc_by_class, _ = evaluator.eval_auc(testresults, labels[ts])
                auc.append( np.sum(np.bincount(labels[ts]) * auc_by_class) / len(labels[ts]) )

            mce_mean = np.sum(mce) / float(len(labels))
            rmse_mean = np.sum(rmse) / float(len(labels))
            auc_mean = np.sum(auc) / float(len(labels))

            mce_var = np.sum((mce - mce_mean)**2) / float(len(labels))
            rmse_var = np.sum((rmse - rmse_mean)**2) / float(len(labels))
            auc_var = np.sum((auc - auc_mean)**2) / float(len(labels))

            print "Cross entropy (individual data points): %.4f with SD %.4f" % (mce_mean, mce_var**0.5)
            print "RMSE (individual data points): %.4f with SD %.4f" % (rmse_mean, rmse_var**0.5)
            print "AUC (individual data points): %.4f with SD %.4f" % (auc_mean, auc_var**0.5)

            # assume gold density and est density have 1 row for each class
            est_density = pred['grid'].flatten()

            mced = - np.sum(gold_density * np.log(est_density)) - np.sum(gold_density * np.log(1-est_density))
            mced = mced / float(len(gold_density))
            print "Cross entropy (density estimation): %.4f" % mced

            rmsed = np.sqrt( np.sum((est_density - gold_density)**2) / float(len(gold_density)) )
            print "RMSE (density estimation): %.4f" % rmsed

            tau, _ = kendalltau(est_density, gold_density)
            print "Kendall's Tau (density estimation): %.4f" % tau

            if method not in auc_all:
                auc_all[method] = [auc_mean]
                rmse_all[method] = [rmse_mean]
                mce_all[method] = [mce_mean]

                auc_all_var[method] = [auc_var]
                rmse_all_var[method] = [rmse_var]
                mce_all_var[method] = [mce_var]

                tau_all[method] = [tau]
                rmsed_all[method] = [rmsed]
                mced_all[method] = [mced]
            else:
                auc_all[method].append(auc_mean)
                rmse_all[method].append(rmse_mean)
                mce_all[method].append(mce_mean)

                auc_all_var[method].append(auc_var)
                rmse_all_var[method].append(rmse_var)
                mce_all_var[method].append(mce_var)

                tau_all[method].append(tau)
                rmsed_all[method].append(rmsed)
                mced_all[method].append(mced)

        # set up next iteration
        results_all[nlabels] = results
        if nlabels==C_all.shape[0]:
            break
        elif C_all.shape[0]-nlabels<100:
            nlabels = C_all.shape[0]
        else:
            nlabels += stepsize

    np.save(outputdir + "results.npy", results_all)
    np.save(outputdir + "rmse.npy", rmse_all)
    np.save(outputdir + "auc.npy", auc_all)
    np.save(outputdir + "mce.npy", mce_all)
    np.save(outputdir + "rmse_var.npy", rmse_all_var)
    np.save(outputdir + "auc_var.npy", auc_all_var)
    np.save(outputdir + "mce_var.npy", mce_all_var)
    np.save(outputdir + "rmsed.npy", rmsed_all)
    np.save(outputdir + "tau.npy", tau_all)
    np.save(outputdir + "mced.npy", mced_all)

    return heatmapcombiner, gpgrid, gpgrid2, ibcc_combiner
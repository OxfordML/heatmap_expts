'''
Created on 29 Feb 2016

@author: edwin
'''
import numpy as np
import pandas as pd

gridsize_km = 0.5

def matern_3_2(xvals, yvals, ls):
    Kx = np.abs(xvals) * 3**0.5 / ls
    Kx = (1 + Kx) * np.exp(-Kx)
    Ky = np.abs(yvals) * 3**0.5 / ls
    Ky = (1 + Ky) * np.exp(-Ky)
    K = Kx * Ky
    return K

def logit(g):
    f = -np.log(1/g - 1)
    return f

def sigmoid(f):
    g = 1/(1+np.exp(-f))
    return g

def convert_lat_lon_to_grid(lat, lon, minlat, minlon, maxlat, maxlon, nx, ny):
    x = (lat - minlat) / (maxlat - minlat) * nx
    y = (lon - minlon) / (maxlon - minlon) * ny    
    return x, y

def allocate_tasks(classification_data, available_workers, available_images, nperworker=2):
    #classifcation_data columns: worker_id, click_lat, click_lon, image_id, classification.
    # available_workers: just a list of worker_ids
    # available_images may include images that have not yet been seen by any workers. Therefore, provide location information for these images using the following 
    # columns: image_id, image_bottomleft_lat, image_bottomleft_lon, image_topright_lat, image_topright_lon
    
    # TODO: check that the correct interface is known to BMT, i.e. pass in the full set of available_images data with lat/lon
    # TODO: replace image candidate assignment with real code
    # TODO: put in the real expected information gain calculation 
    
    C = classification_data # convenient label

    minlat = np.min(available_images[:, 1])
    minlon = np.min(available_images[:, 2])
    maxlat = np.max(available_images[:, 3])
    maxlon = np.max(available_images[:, 4])

    nx = np.ceil((maxlat - minlat) * 2500.0) / gridsize_km # this is intended to give grid squares approximately 200mx200m
    ny = np.ceil((maxlon - minlon) * 2500.0) / gridsize_km 
    
    #get the click locations in terms of our grid and discretize
    click_lat = classification_data[:, 1]
    click_lon = classification_data[:, 2]
    x, y = convert_lat_lon_to_grid(click_lat, click_lon, minlat, minlon, maxlat, maxlon, nx, ny)
    
    # use a discrete grid 
    x = np.floor(x).astype(int)
    y = np.floor(y).astype(int)   

    # Iterative Method: 
    nworkers = len(available_workers)
    ncandidates_per_worker = 20 
    candidate_imgs = np.zeros((nworkers, ncandidates_per_worker)) - 1 # candidate images available to each worker
    ncandidates = np.zeros(nworkers)
    workers = np.arange(nworkers)
    need_more_locations = True # flag to indicate that we should find more candidate locations
    
    # 1. Find set of most uncertain locations as follows:
    i = 0
    while need_more_locations and i < 100:
        #  a) Select most uncertain location
        # do this at random for now
        new_img = np.random.randint(0, len(available_images))
        while new_img in candidate_imgs:
            new_img = np.random.randint(0, len(available_images))
        
        for w in workers:
            if ncandidates[w] == ncandidates_per_worker:
                continue
            if not new_img in C[C[:, 0]==available_workers[w], 3]:
                candidate_imgs[w, ncandidates[w]] = new_img
                ncandidates[w] += 1
        if np.sum(ncandidates < ncandidates_per_worker) == 0:
            need_more_locations = False
        i += 1 # put this in as an easy way to avoid infinite loops if a worker has seen all images
        
        #  b) Simulate full knowledge of that location and run IBCC
        #  c) Repeat from (a) to determine more uncertain locations. This process avoids selecting multiple places that are closely connected if using HeatmapBCC.
        #  d) Ensure that we chose locations that have not yet been labeled by the available workers so each worker has 20 possibilities. 

    # 2. For each available worker, calculate the EIG for each allowable assignment.
    I = np.zeros((ncandidates_per_worker, nworkers))
    #skip the actual calculation for now
    
    # 3. Select the pairs in order of maximum EIG.  
    allocs = np.zeros((nperworker, nworkers)) - 1
    while np.sum(allocs == -1):
        # get the index of the optimum assignment
        maxidx = np.argmax(I)
        maxcand, maxworker = np.unravel_index(maxidx, dims=(ncandidates_per_worker, nworkers))
        
        # get allocation number for this worker 
        alloc_to_worker_idxs = nperworker - np.sum(allocs[:, maxworker]==-1)
        # retrieve the image
        img_to_assign = candidate_imgs[maxworker, maxcand]
        # assign the candidate
        allocs[alloc_to_worker_idxs, maxworker] = img_to_assign

        print "Assigning worker %i to image %i" % (maxworker, img_to_assign)

        # if the worker has enough allocations, disqualify the worker 
        if alloc_to_worker_idxs == nperworker - 1:
            I[:, maxworker] = -1
            candidate_imgs[maxworker, :] = -1

        # replace the candidate so we don't run out of options, and so that the candidate is no longer availabe to others
        new_img = np.random.randint(0, len(available_images))
        while new_img in candidate_imgs:
            new_img = np.random.randint(0, len(available_images))
        # disqualify the candidate from being assigned again in this round
        I[candidate_imgs[:, :].T == img_to_assign] = 0 # replace this with the correct value for new_img
        candidate_imgs[candidate_imgs == img_to_assign] = new_img
      
    workers = np.tile(workers[np.newaxis, :], (2, 1)).flatten() # workers to allocate tasks to
    images = allocs.flatten() # images to allocate to the workers
    return workers, images

if __name__ == '__main__':
    from scipy.stats import beta, bernoulli, multivariate_normal as mvn, norm
    from scipy.linalg import cholesky
    import time
    import os.path
    
    # Run a simulation to test the method.  
    # 1. Load the gold standard for the PRN dataset for structural damage 3. 
    datafile = "./data/prn_pilot_classifications_onerowpermark_withsubjectinfo.csv"
    dataframe = pd.read_csv(datafile, usecols=["subject_id", "user_id", "resolution_level", "lat_mark", "long_mark", "lat_lowerleft", 
                               "long_lowerleft", "lat_upperright", "long_upperright", "mark_type", "damage_assessment"])
    
    img_ids = dataframe["subject_id"]
    img_ids, img_idxs = np.unique(img_ids, return_index=True)
    img_ids = np.arange(len(img_ids), dtype=float)[:, np.newaxis]
    
    lat_lowerleft = np.array(dataframe["lat_lowerleft"][img_idxs].tolist())[:, np.newaxis]
    lon_lowerleft = np.array(dataframe["long_lowerleft"][img_idxs].tolist())[:, np.newaxis]
    lat_upperright = np.array(dataframe["lat_upperright"][img_idxs].tolist())[:, np.newaxis]
    lon_upperright = np.array(dataframe["long_upperright"][img_idxs].tolist())[:, np.newaxis]
    
    available_imgs = np.concatenate((img_ids, lat_lowerleft, lon_lowerleft, lat_upperright, lon_upperright), axis=1)

    minlat = np.min(lat_lowerleft)
    minlon = np.min(lon_lowerleft)
    maxlat = np.max(lat_upperright)
    maxlon = np.max(lon_upperright)

    nx = np.ceil((maxlat - minlat) * 2500.0) / gridsize_km
    ny = np.ceil((maxlon - minlon) * 2500.0) / gridsize_km

    if os.path.isfile("./data/intelligent_allocation/synth_y_gold.npy"):
        #create some synthetic "gold"
        x_gold = np.tile(np.arange(nx)[:, np.newaxis], (1, ny)).reshape((nx*ny, 1))
        y_gold = np.tile(np.arange(ny)[np.newaxis, :], (nx, 1)).reshape((nx*ny, 1))
        
        lat_gold = x_gold * (maxlat - minlat) / nx + minlat  
        lon_gold = y_gold * (maxlon - minlon) / ny + minlon
        
        img_ids_gold = []
        for i in range(len(lat_gold)):
            img_ids_i = img_ids[ (lat_lowerleft <= lat_gold[i]) & (lat_upperright >= lat_gold[i]) & (lon_lowerleft <= lon_gold[i]) & (lon_upperright >= lon_gold[i])]
            img_ids_gold.append(img_ids_i)
    
        ddx = x_gold - x_gold.T
        ddy = y_gold - y_gold.T
    
        output_scale = 1.0 / logit(0.75)**2
        
        # Use a fixed length-scale learned from other experiments on the Haiyan data
        ls = 80.0 # 32 / 132.0 * nx gives roughly this, i.e. take it from the prn experiments and convert to the local grid size
        
        K = matern_3_2(ddx, ddy, ls)
        f_mean = np.zeros(len(x_gold))
        
        A = cholesky(K, lower=True, check_finite=False)
        z = norm.rvs(loc=0, scale=1)
        f_gold = f_mean + A.dot(z)
        #f_gold = mvn.rvs(mean=f_mean, cov=K / output_scale) # this doesn't work at large scale
        rho_gold = sigmoid(f_gold)
        t_gold = bernoulli.rvs(rho_gold)
        
        np.save("./data/intelligent_allocation/synth_t_gold.npy", t_gold)
        np.save("./data/intelligent_allocation/synth_rho_gold.npy", rho_gold)
        np.save("./data/intelligent_allocation/synth_f_gold.npy", f_gold)
        np.save("./data/intelligent_allocation/synth_x_gold.npy", x_gold)
        np.save("./data/intelligent_allocation/synth_y_gold.npy", y_gold)
    else:
        t_gold = np.load("./data/intelligent_allocation/synth_t_gold.npy")
        rho_gold = np.load("./data/intelligent_allocation/synth_rho_gold.npy")
        f_gold = np.load("./data/intelligent_allocation/synth_f_gold.npy")
        x_gold = np.load("./data/intelligent_allocation/synth_x_gold.npy")
        y_gold = np.load("./data/intelligent_allocation/synth_y_gold.npy")
        
    # 2. Generate 30 simulated workers of all types (random draws from Beta distributions).
    alpha0 = np.ones(2) 
    alpha0[0] = 3 # prior to draw workers from
    K = 30 #number of workers
    
    pi = np.zeros((2, 2, K))
    pi[0, 0, :] = beta.rvs(alpha0[0], alpha0[1], size=K)
    pi[0, 1, :] = 1 - pi[0, 0, :]
    pi[1, 1, :] = beta.rvs(alpha0[0], alpha0[1], size=K)
    pi[1, 0, :] = 1 - pi[1, 1, :]
    
    nu0 = np.array([100, 100], dtype=float)
    z0 = nu0[0] / np.sum(nu0)
    shape_s0 = 0.5
    rate_s0 = 10.0 * 0.5
    
    # 3. Start with a random assignment. Then proceed to iteratively...
    C = []
    nlabels = 0
    workerIDs = np.arange(K)[:, np.newaxis]
    
    selection = np.random.randint(0, len(x_gold), size=K)
    x_ass = x_gold[selection][:, np.newaxis]
    y_ass = y_gold[selection][:, np.newaxis]
    t_ass = t_gold[selection][:, np.newaxis]
    
    workers = workerIDs
    images = []
    for i in range(len(selection)):
        images_list = img_ids_gold[selection[i]] #find some images that cover x_ass and y_ass
        while not len(images_list):
            newid = np.random.randint(0, len(x_gold))
            x_ass[i] = x_gold[newid]
            y_ass[i] = y_gold[newid]
            t_ass[i] = t_gold[newid]
            images_list = img_ids_gold[newid]
            
        images.append(images_list[np.random.randint(0, len(images_list))])
    images = np.array(images)[:, np.newaxis]
    
    gridxmin, gridymin = convert_lat_lon_to_grid(lat_lowerleft, lon_lowerleft, minlat, minlon, maxlat, maxlon, nx, ny)
    gridxmax, gridymax = convert_lat_lon_to_grid(lat_upperright, lon_upperright, minlat, minlon, maxlat, maxlon, nx, ny)
    
    nlabels_at_iteration = []
    
    while nlabels < 5000:
        #  a) Draw classifications from the simulated workers.
        newclassifications = bernoulli.rvs(pi[t_ass.flatten(), 1, workers.flatten()])[:, np.newaxis]
        Cnew = np.concatenate((workers.astype(int), x_ass.astype(float), y_ass.astype(float), images, newclassifications.astype(int)), axis=1)
        #available_images = x_ass, y_ass, x_ass+width, y_ass+width
        if not np.any(C):
            C = Cnew
        else:
            C = np.concatenate((C, Cnew), axis=0)
            
        # record the number of labels in the current iteration
        nlabels_at_iteration.append(C.shape[0])
            
        #  b) Get new assignments
        available_workers = np.arange(K)
        
        #  c) Time (b) and record the duration and quantity of data at each iteration.
        start = time.clock()
        workers, images = allocate_tasks(C, available_workers, available_imgs)
        end = time.clock()
        print "Allocation took %f seconds process time" % (end - start)
     
        # Given the image assignments, draw random click locations
        x_ass = []
        y_ass = []
        t_ass = []
        for img in images:
            img = int(img)
            x = np.random.randint(np.floor(gridxmin[img]), np.floor(gridxmax[img]) + 1)
            y = np.random.randint(np.floor(gridymin[img]), np.floor(gridymax[img]) + 1)
            x_ass.append(x)
            y_ass.append(y)
            t_ass.append(t_gold[(x_gold==x) & (y_gold==y)])
            
        t_ass = np.array(t_ass)[:, np.newaxis]
        x_ass = np.array(x_ass)[:, np.newaxis]
        y_ass = np.array(y_ass)[:, np.newaxis]   
        images = np.array(images)[:, np.newaxis]
        workers = np.array(workers)[:, np.newaxis]
            
        nlabels = C.shape[0]
        print "nlabels = %i" % nlabels
    
    # 4. Evaluate accuracy by using prediction tests with HeatmapBCC to see how accuracy changes with number of labels.
    from prediction_tests import Tester
    outputdir = "./data/intelligent_allocation/" # %i_%i" % (c, d)
    methods = ['HeatmapBCC']
    tester = Tester(outputdir, methods, C.shape[0], z0, alpha0, nu0, shape_s0, rate_s0, 
                    ls, optimise=False, verbose=False)
    tester.run_tests(C, nx, ny, x_gold, y_gold, t_gold, rho_gold, nlabels_at_iteration[0], 
                     nlabels_at_iteration[1] - nlabels_at_iteration[0]) 
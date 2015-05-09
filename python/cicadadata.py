'''
Created on 9 May 2015

@author: edwin
'''
import numpy as np
import logging

class CicadaDataHandler(object):
    
    K = 1
    nx = 1000
    ny = 1000
    # values will be set automatically from the data
    minlat = 0
    maxlat = 0
    minlon = 0
    maxlon = 0 
    
    datadir = './data'
    
    C = []
    
    #Dictionary defining how we map original categories 
    #to class IDs, using original categories as keys:
    categorymap = {1:1, 3:1,5:{'a':1},6:{'a':1,'b':1}} 
    
    def __init__(self, nx, ny, datadir):
        self.nx = nx
        self.ny = ny
        self.datadir = datadir
        
    def translate_points_to_local(self, latdata, londata):
        logging.debug('Translating original coords to local values.')
            
        latdata = np.float64(latdata)
        londata = np.float64(londata)
        
        normlatdata = (latdata-self.minlat)/(self.maxlat-self.minlat)
        normlondata = (londata-self.minlon)/(self.maxlon-self.minlon)    
            
        latdata = np.array(np.round(normlatdata*self.nx), dtype=np.int)
        londata = np.array(np.round(normlondata*self.ny), dtype=np.int)
            
        return latdata,londata         
    
    def load_data(self):
        dataFile = self.datadir+'/cicada-edwin.csv'
        self.K = 1
        #load the data
    #     reportIDdata = np.genfromtxt(dataFile, np.str, delimieter=',', skip_header=True, usecols=[])
    #     datetimedata = np.genfromtxt(dataFile, np.datetime64, delimiter=',', skip_header=True, usecols=[2,3])
        latdata = np.genfromtxt(dataFile, np.float64, delimiter=',', skip_header=True, usecols=[3])
        londata = np.genfromtxt(dataFile, np.float64, delimiter=',', skip_header=True, usecols=[4])
        
        self.minlat = np.min(latdata)
        self.maxlat = np.max(latdata)
        self.minlon = np.min(londata)
        self.maxlon = np.max(londata)
        latdata,londata = self.translate_points_to_local(latdata,londata)
                
        classids = np.genfromtxt(dataFile, np.str, delimiter=',', skip_header=True, usecols=[6])
        deviceids = np.genfromtxt(dataFile, np.str, delimiter=',', skip_header=True, usecols=[5])
        values = np.genfromtxt(dataFile, np.float64, delimiter=',', skip_header=True, usecols=[8])
        targetdata = np.genfromtxt(dataFile, np.int, delimiter=',', skip_header=True, usecols=[9])
        
        C = np.empty((deviceids.shape[0], 5)) # fifth column will hold the agent type as we have groupings of agents  
        targets = np.empty(targetdata.shape[0], 3)       
        instantiatedagents = []
        
        original_target_values = np.sort(np.unique(targetdata))
        
        for i, deviceid in enumerate(deviceids):
            
            reportvalue = values[i]
            
            #agent ID needs to be a fusion between the class it is detecting and the device
            classid = classids[i]
            agentIDstr = "%s_%s" % (classid, deviceid)
            agenttype = np.int(classid) # use the agenttype to determine which prior is suitable. Will require changing
            # the optimizer.            
            if agentIDstr not in instantiatedagents:
                instantiatedagents.append(agentIDstr)
            
            agentID = np.argwhere(instantiatedagents==agentIDstr)[0,0]
            
            repx = latdata[i]
            repy = londata[i]
            C[i, :] = [agentID, repx, repy, reportvalue, agenttype]
            
            if np.isnan(targetdata[i]):
                targetvalue = np.nan
            else:
                targetvalue = np.argwhere(original_target_values==targetdata[i])[0,0]
            
            targets[i, :] = [repx, repy, targetvalue]     
            
        self.K = len(instantiatedagents)        
        self.C = C
        self.targets = targets
        print "Number of reports: " + str(self.C.shape[0])            
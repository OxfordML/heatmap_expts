'''
Created on 7 May 2015

@author: edwin
'''

import numpy as np
import logging

class UshahidiDataHandler(object):
    
    K = 1
    nx = 1000
    ny = 1000
    
    minlat = 18.2#18.0
    maxlat = 18.8#19.4
    minlon = -72.6#-73.1
    maxlon = -72.0#-71.7   
    
    datadir = './data'
    
    C = []
    
    #Dictionary defining how we map original categories 
    #to class IDs, using original categories as keys:
    categorymap = {1:1, 3:1,5:{'a':1},6:{'a':1,'b':1}} 
    
    discrete = True
    
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
            
        if self.discrete:
            latdata = np.array(np.round(normlatdata*self.nx), dtype=np.int)
            londata = np.array(np.round(normlondata*self.ny), dtype=np.int)
        else:
            latdata = np.array(normlatdata*self.nx)
            londata = np.array(normlondata*self.ny)
            
        return latdata,londata           
    
    def load_data(self):
        dataFile = self.datadir+'/exported_ushahidi.csv'
        self.K = 1
        #load the data
    #     reportIDdata = np.genfromtxt(dataFile, np.str, delimieter=',', skip_header=True, usecols=[])
    #     datetimedata = np.genfromtxt(dataFile, np.datetime64, delimiter=',', skip_header=True, usecols=[2,3])
        latdata = np.genfromtxt(dataFile, np.float64, delimiter=',', skip_header=True, usecols=[4])
        londata = np.genfromtxt(dataFile, np.float64, delimiter=',', skip_header=True, usecols=[5])
        reptypedata = np.genfromtxt(dataFile, np.str, delimiter=',', skip_header=True, usecols=[1])
        latdata,londata = self.translate_points_to_local(latdata,londata)
        C = {}            
        instantiatedagents = []

        for i, reptypetext in enumerate(reptypedata):        
            typetoks = reptypetext.split('.')
            typetoks = typetoks[0].split(',')
            maintype = 1
            if "1a." in reptypetext: # Trapped people
                agentID = 0
            elif "1b." in reptypetext: 
                agentID = 1
            elif "1c." in reptypetext:
                agentID = 2
            elif "1" in reptypetext or "1d." in reptypetext:
                agentID = 3
            elif "2a." in reptypetext:
                agentID = 4
            elif "2b." in reptypetext:
                agentID = 5
            elif "2c." in reptypetext:
                agentID = 6
            elif "2d." in reptypetext:
                agentID = 7
            elif "2e." in reptypetext:
                agentID = 8
            elif "2f." in reptypetext:
                agentID = 9
            elif "2g." in reptypetext:
                agentID = 10              
            elif "3a." in reptypetext:
                agentID = 11
            elif "3." in reptypetext or "3b." in reptypetext or "3c." in reptypetext or "3d." in reptypetext or \
                "3e." in reptypetext  or "6b." in reptypetext:
                agentID = 12
            elif "4a." in reptypetext:
                agentID = 13
            elif "4." in reptypetext or "4b." in reptypetext:
                agentID = 14
            elif "5a." in reptypetext:
                agentID = 15
            elif "5." in reptypetext or "5c." in reptypetext or "5d." in reptypetext:
                agentID = 16
            elif "6." in reptypetext or "6a." in reptypetext:
                agentID = 17
            elif "7a." in reptypetext:
                agentID = 18     
            elif "7b." in reptypetext:
                agentID = 19
            elif "7c." in reptypetext:
                agentID = 20
            elif "7." in reptypetext or "7d." in reptypetext or "7e." in reptypetext or "7f." in reptypetext:
                agentID = 21
            elif "8a." in reptypetext:
                agentID = 22
            elif "8b." in reptypetext or "6c." in reptypetext or "6d." in reptypetext:
                agentID = 23
            elif "8c." in reptypetext:
                agentID = 24
            elif "8e." in reptypetext or "8." in reptypetext or  "8d." in reptypetext or "8f." in reptypetext:
                agentID = 25                
            else: #we don't care about these categories in the demo anyway, but can add them easily here
                agentID = 26
                for typestring in typetoks:
                    if len(typestring)>1:
                        sectype = typestring[1] # second character should be a letter is available
                    else:
                        sectype = 0
                    try:
                        maintype = int(typestring[0]) #first character should be a number
                    except ValueError:
                        logging.warning('Not a report category: ' + typestring)
                        continue                   
                    if maintype in self.categorymap:
                        mappedID = self.categorymap[maintype]
                        if type(mappedID) is dict:
                            if sectype in mappedID:
                                maintype = mappedID[sectype]
                        else:
                            maintype = mappedID

            repx = latdata[i]
            repy = londata[i]
            if repx>=self.nx or repx<0 or repy>=self.ny or repy<0:
                continue
            
            try:
                Crow = np.array([agentID, repx, repy, 1]) # all report values are 1 since we only have confirmations of an incident, not confirmations of nothing happening
            except ValueError:
                logging.error('ValueError creating a row of the crowdsourced data matrix.!')        

            if C=={} or maintype not in C:
                C[maintype] = Crow.reshape((1,4))
            else:
                C[maintype] = np.concatenate((C[maintype], Crow.reshape(1,4)), axis=0)
                
            if agentID not in instantiatedagents:
                instantiatedagents.append(agentID)
                
        instantiatedagents = np.array(instantiatedagents)
        for j in C.keys():
            for r in range(C[j].shape[0]):
                C[j][r,0] = np.argwhere(instantiatedagents==C[j][r,0])[0,0]
        self.K = len(instantiatedagents)
                     
        self.C = C
        print "Number of type one reports: " + str(self.C[1].shape[0])

# -*- coding: utf-8 -*-
"""
PreProcessing test script

Travis Thonstad 2016-01-19
"""

import glob
import os
import scipy.io.wavfile
#import sys
#import matplotlib.pyplot as plt
#import numpy as np




def readWAV(rawDataPath,files=None):
    ''' (names,data) = readWAV(rawDataPath). Reads the contents of the specified 
    directory for *.wav files, converts the files to arrays and returns a dictionary 
    containing the array data. 
    
    The keys for the dictionary are returned in the names list (the 
    filenames of the *.wav files).
    
    Input:
        rawDataPath - the directory where the wav files are stored.
        files (optional) - list of files to be loaded
    
    Output:
        names - a list of the dictionary keys to data.
        data - a dictionary containing the Nx2 arrays of the audio time series. 
    
    '''
    names = [];
    data = {};
    
    if not files:
        files = glob.glob(os.path.join(rawDataPath,"*.wav"))
        
        for name in files:
            fileName = os.path.basename(name).split(".")[0]
            names.append(fileName)
            print('Opening ' + fileName + ' ...')
            
            audioIn = scipy.io.wavfile.read(name);
            data[fileName] = audioIn[1];
    else:
        for fileName in files:
            print('Opening ' + fileName + ' ...')
            names.append(fileName)
            
            audioIn = scipy.io.wavfile.read(os.path.join(rawDataPath,fileName + ".wav"));
            data[fileName] = audioIn[1];
        
    return (names,data)
    
    
    
def getKeys(names,parts):
    ''' keys = getKeys(names,parts). Returns the entries in list filtered using the 
     set of name "parts" (Explained in detail below). Compares parts to 
     entries in provided list of names.
     
     Each key in names sould contain a set of identifiers separated by underscores.
     Format:
         AA_BB_CC_DDDD
         
         AA    - Camera type (GP=GoPro)
         BB    - Bent number (B1,B2,B3 = Bent 1,2,3)
         CC    - Location of camera on the bent (NL,NU,SL,SU = North/South Upper/Lower)
         DDDD  - Unique motion identifier (1A,1B,4,5,6,9A,9B,12,13,14A,S1,S2,S3,9C,
                 S4,S5,14B1,14B2,14C,15,16,17,18,19,20A,20B,21A,21B,21C)
                
     Input: 
         names - A list of data identifiers
         parts - A tuple of lists containing the three name part filters ([BB's],[CC's],[DDDD's]).
                 A value of None indicates that no filter will be applied for that name part.
                 i.e. (['B1'],None,['1A','16',...,'20A'])
                 
    Output:
        keys   - the entries in names that correspond to the provided filters
    '''

    keys = [];
    
    for name in names:
        [cType,bent,loc,motion] = name.split("_");
        if not parts[0]: #empty bent identifier
            if not parts[1]: #empty location identifier
                if not parts[2]: # empty motion identfier
                    keys.append(name)
                elif motion in parts[2]: # motions are input
                    keys.append(name) 
            elif loc in parts[1]: # locations are input
                if not parts[2]: # empty motion identfier
                    keys.append(name)
                elif motion in parts[2]: # motions are input
                    keys.append(name)
        elif bent in parts[0]:
            if not parts[1]: #empty location identifier
                if not parts[2]: # empty motion identfier
                    keys.append(name)
                elif motion in parts[2]: # motions are input
                    keys.append(name) 
            elif loc in parts[1]: # locations are input
                if not parts[2]: # empty motion identfier
                    keys.append(name)
                elif motion in parts[2]: # motions are input
                    keys.append(name) 
                    
    return keys
        
def loadSubset(data,names):
    ''' subset = loadSubset(data,names). Returns the subset of the input dictionary, data,
        specified by the key values given in names.
        
    Input: 
         data  - dictionary containing audio data.
         names - the dictionary keys for the subset data dictionary.
                 
    Output:
        subset - a new dictionary that contains only the specified keys.
    '''
    
    subset = {k: data[k] for k in names};
        
    return subset
    

    
    

    





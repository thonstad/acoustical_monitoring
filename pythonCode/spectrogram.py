# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:24:04 2016

Script that plots spectrograms of audio files. 

@author: Travis
"""

import glob
import os
#import sys
import matplotlib.pyplot as plt


# TEST THE FUNCTIONS
import preprocessing as pp


rawDataPath = os.path.join("..","rawData"); # directory where the audio fles reside

files = glob.glob(os.path.join(rawDataPath,"*.wav"))
names = [];
        
for name in files:
    fileName = os.path.basename(name).split(".")[0]
    names.append(fileName)


filt = (None,None,['17']) # 
audioFiles = pp.getKeys(names,filt);

(names,cDataset) = pp.readWAV(rawDataPath,audioFiles); # opens files and writes to a dictionary
       
#cDataset = loadSubset(data,audioFiles);
    

plt.figure(figsize=(50,20))

for ii in range(len(cDataset)):
    
    plt.subplot(len(cDataset)/2,2,ii+1)
    (Pxx,freqs,bins,im) = plt.specgram(cDataset[audioFiles[ii]][:,0],NFFT=2048,Fs=48000,noverlap=900,cmap=plt.cm.gist_heat)
    #plt.plot(cDataset[audioFiles[ii]][:,0]-cDataset[audioFiles[ii]][:,1],'b')
    #plt.ylim([3000,3800])
    plt.draw()

#plt.ion()
 
#plt.show()
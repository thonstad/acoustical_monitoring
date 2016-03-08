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
import numpy as np


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

# function which calculates fracture locations based on edges in spectrogram:

# smoothing
# options 'canny','sobel' (canny has extra parameters)
# threshold based on column sum, based on hough transform.

# return locations, plot lines on top of original signal,
# on top of histogram?
# decide how to automatically select threshold


def edges2fractures(ts, Fs=48000, edge_type='Sobel', smoothing=None,loc_type='thr',threshold=10):
    from skimage import filters
    from skimage import feature

    NFFT = 2048       # the length of the windowing segments
    # convert the frequency to integer
    Fs = int(Fs)
    # calculate the spectrogram
    plt.figure(figsize = (10,3))
    Pxx, freq, bins, im = plt.specgram(ts, NFFT=NFFT, Fs=Fs, noverlap=900, mode='psd', cmap='gray', aspect = 'auto')

    # applying smoothing
    if smoothing is not None:
        Pxx = filters.gaussian_filter(Pxx, smoothing)


    # extracting edges
    if edge_type=='Sobel':
        edges = filters.sobel_v(Pxx)
        # convert to binary edges
        # fix arbitrary threshold!!!
        edges = (np.abs(edges)>20)
    elif edge_type=='Canny':
        edges = feature.canny(Pxx)


    #plt.figure(figsize = (10,3))
    #plt.imshow(edges, cmap='gray')
    # determining fracture locations
    colSums = np.sum(edges, axis=0)
    print(np.max(colSums))
    print(np.min(colSums))

    # truncate at some number of standard deviations
    std = np.std(colSums)
    md = np.median(colSums)
    frac_idx, = np.where(colSums > md+3*std)

    # plotting the results
    plt.figure(figsize = (10,3))
    plt.plot(np.arange(len(ts))/Fs,ts)
    # plt.plot(colSums)
    fig = [plt.axvline(bins[_x], linewidth=1, color='g') for _x in frac_idx]

    return bins[frac_idx]

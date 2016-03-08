# -*- coding: utf-8 -*-
"""
This file contains functions to detect fractures based on different features.
"""

import glob
import os
import matplotlib.pyplot as plt
import numpy as np


import preprocessing as pp

# directory where the audio fles reside
rawDataPath = os.path.join("..","rawData")

# extract all filenames
files = glob.glob(os.path.join(rawDataPath,"*.wav"))
names = []
for name in files:
    fileName = os.path.basename(name).split(".")[0]
    names.append(fileName)


# specify a specific file
filt = (None,None,['17']) #
audioFiles = pp.getKeys(names,filt);

# opens files and writes to a dictionary
(names,cDataset) = pp.readWAV(rawDataPath,audioFiles);

# cDataset = loadSubset(data,audioFiles);


def edges2fractures(ts, Fs=48000, edge_type='Sobel', smoothing=None):
    """
        edges2fractures predicts peaks locations based on edges
        in the signal spectrogram

        Inputs
        ------

        Fs: scalar
            the frequency of the signal
        edge_type: string
            the type of the edge detector - 'Sobel' or 'Canny'
        smoothing: scalar or None
            if None, no smoothing is applied, otherwise
            smoothed with Gaussian filter with sigma=smoothing

        Returns
        -------
        array containing the times of fractures
    """

    from skimage import filters
    from skimage import feature

    # TODO Possibly allow to change the window
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
        # TODO fix arbitrary threshold!!!
        edges = (np.abs(edges)>20)
    elif edge_type=='Canny':
        edges = feature.canny(Pxx)
    else:
        raise ValueError('Invalid Edge Detector Type')




    #plt.figure(figsize = (10,3))
    #plt.imshow(edges, cmap='gray')
    # determining fracture locations
    colSums = np.sum(edges, axis=0)


    # truncate at some number of standard deviations (here 3)
    std = np.std(colSums)
    md = np.median(colSums)
    frac_idx, = np.where(colSums > md+3*std)
    if len(frac_idx) == 0:
        return []

    # plotting the results
    plt.figure(figsize = (10,3))
    plt.plot(np.arange(len(ts))/Fs,ts)
    # plt.plot(colSums)
    fig = [plt.axvline(bins[_x], linewidth=1, color='g') for _x in frac_idx]

    return bins[frac_idx]

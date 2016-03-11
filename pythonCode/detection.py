# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:33:52 2016

@author: Travis
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt



def med_filt(chan,kernel = 11,thresh = 3):
    '''    
    Identifies fractures within the signal by thresholding. 
    Uses a median filtering method to  subtract the baseline from the absolute
    value of the signal. 
    
    Inputs:
        chan - (nparray) input signal array
        
        optional
        -----------
        kernel (int) kernel size of the median filter.
        
        thresh (int) the number of standard deviations above the mean to consider
                     the identified peak a fracture.
                     
    Outputs:
        fractures (list) indicies of the identified fractures within the signal
    '''
    channel = abs(chan)
    filtChannel = signal.medfilt(channel, kernel_size = kernel)
    corr_chan = channel-filtChannel
    
    fractures = sliding_max(corr_chan,3000,thresh)
    
    time = 1/48000*np.linspace(0,len(corr_chan),len(corr_chan))
    plt.plot(time,corr_chan)
    plt.plot([time[fractures],time[fractures]],[min(corr_chan),max(corr_chan)],'r')
    
    return time[fractures]


def sliding_max(chan,kernel_size,threshold):
    '''
    Identifies local maximum within the signal by chunking the signal. 
    Throws out maximums that are less than a threshold defined using the mean
    and standard deviation of the signal.
    
    
    Inputs:
        chan - (nparray) input signal array
        
        optional
        -----------
        kernel_size (int) the size of the chunks.
        
        threshold (int) the number of standard deviations above the mean to consider
                     the identified peak a fracture.
                     
    Outputs:
        fractures (list) indicies of the identified fractures within the signal
    
    '''
    
    
    fractures = []
    
    cutoff = np.mean(abs(chan)) + threshold*np.std(abs(chan))
    
    pad = kernel_size-(len(chan) % kernel_size);
    pad_chan = np.hstack((chan,np.zeros((pad,))))
    dim = len(pad_chan)/kernel_size
    
    res_chan = np.reshape(abs(pad_chan),(dim,kernel_size))
    ind = np.argmax(res_chan, axis = 1)
    indind = np.linspace(0,len(pad_chan)-kernel_size,dim)
    
    fract = (np.array((ind + indind),dtype=int)).tolist()

    for frac in fract:
        if abs(pad_chan[frac]) > cutoff:
            fractures.append(frac)
            
    
    
    return fractures


def cwt_ridges(chan,dwnsmp_rat = 48,max_width = 20,thresh = 3):
    '''    
    Identifies frctures within the signal using the continuous wavelet
    transform on the dewnsampled signal (significant downsampling is required 
    in order to perform the cwt on most machines).
    
    Inputs:
        chan - (nparray) input signal array
        
        optional
        -----------
        dwnsmp_rat (int) the fraction of the original sample rate.
                         
        max_width (int) the maximum wavelet width considered.
        
        thresh (int) the number of standard deviations above the mean to consider
                     the identified peak a fracture.
                     
    Outputs:
        fractures (list) indicies of the identified fractures within the signal
    '''
    
    fractures = [] 
    
    # downsamples signal in order to perform the transform
    dec_chan = signal.decimate(chan,dwnsmp_rat)
    Fs = 48000/dwnsmp_rat
    dec_time = 1/Fs*np.linspace(0,len(dec_chan),len(dec_chan))
    
    # wavelet widths
    widths = np.linspace(1,max_width,10)

    
    peakInd = signal.find_peaks_cwt(abs(dec_chan),
                                widths,
                                noise_perc=.1,
                                min_snr=1,
                                min_length = 3)
    
    plt.plot(dec_time,dec_chan)
    
    lmin = min(dec_chan)
    ulim = max(dec_chan)
    
    for peak in peakInd:
        if abs(dec_chan[peak]) > np.mean(abs(dec_chan)) + thresh*np.std(abs(dec_chan)):
            fractures.append(peak)
            plt.plot([dec_time[peak],dec_time[peak]],[lmin,ulim],'r')
            
    
    

    return dec_time[fractures]
    
    


def spectrogram_ridges(chan,gap_thresh = 50,min_length = 150):
    '''
    Identifies fractures in the signal based on the spectrogram.
    Connects local maxima for each frequency bin of the spectrogram matrix, .
    Looks for vertical lines of a given length within the frequency content.
    The goal is to find broadband noises within the signal (fractures).
    
    Inputs:
        chan - (nparray) input signal array
        
        optional
        gap_thresh (int) the maximum number of freq bins that can be skipped,
                         while still considering the ridge line connected.
                         
        min_length (int) the minumum length of ridge lines to be considered a 
                         fracture.
    Outputs:
        fractures (list) indicies of the identified fractures within the signal
    '''
    fractures = []    
    
    Pxx, freqs, bins, im = plt.specgram(chan, NFFT=512, Fs=48000, noverlap=0)
    ridge_lines = identify_ridge_lines(Pxx, 0*np.ones(len(bins)), gap_thresh)
    
    for x in ridge_lines:
        if len(x[1]) > min_length:
            fractures.append(bins[x[1][0]])
            plt.plot(bins[x[1][-10:]],freqs[len(freqs)-x[0][-10:]-1],'b')
    
    
    
    
    return fractures
    


def identify_ridge_lines(matr, max_distances, gap_thresh):
    """
    Identify ridges in the 2D matrix. Expect that the width of
    the wavelet feature increases with increasing row number.

    Parameters
    ----------
    matr: 2-D ndarray
        Matrix in which to identify ridge lines.
    max_distances: 1-D sequence
        At each row, a ridge line is only connected
        if the relative max at row[n] is within
        `max_distances`[n] from the relative max at row[n+1].
    gap_thresh: int
        If a relative maximum is not found within `max_distances`,
        there will be a gap. A ridge line is discontinued if
        there are more than `gap_thresh` points without connecting
        a new relative maximum.

    Returns
    -------
    ridge_lines: tuple
        tuple of 2 1-D sequences. `ridge_lines`[ii][0] are the rows of the ii-th
        ridge-line, `ridge_lines`[ii][1] are the columns. Empty if none found.
        Each ridge-line will be sorted by row (increasing), but the order
        of the ridge lines is not specified

    References
    ----------
    Bioinformatics (2006) 22 (17): 2059-2065.
    doi: 10.1093/bioinformatics/btl355
    http://bioinformatics.oxfordjournals.org/content/22/17/2059.long

    Examples
    --------
    >>> data = np.random.rand(5,5)
    >>> ridge_lines = identify_ridge_lines(data, 1, 1)

    Notes:
    ------
    This function is intended to be used in conjuction with `cwt`
    as part of find_peaks_cwt.
    """

    if(len(max_distances) < matr.shape[0]):
        raise ValueError('Max_distances must have at least as many rows as matr')

    all_max_cols = boolrelextrema(matr, np.greater, axis=1, order=1)
    #Highest row for which there are any relative maxima
    has_relmax = np.where(all_max_cols.any(axis=1))[0]
    if(len(has_relmax) == 0):
        return []
    start_row = has_relmax[-1]
    #Each ridge line is a 3-tuple:
    #rows, cols,Gap number
    ridge_lines = [[[start_row],
                   [col],
                   0] for col in np.where(all_max_cols[start_row])[0]]
    final_lines = []
    rows = np.arange(start_row - 1, -1, -1)
    cols = np.arange(0, matr.shape[1])
    for row in rows:
        this_max_cols = cols[all_max_cols[row]]

        #Increment gap number of each line,
        #set it to zero later if appropriate
        for line in ridge_lines:
            line[2] += 1

        #XXX These should always be all_max_cols[row]
        #But the order might be different. Might be an efficiency gain
        #to make sure the order is the same and avoid this iteration
        prev_ridge_cols = np.array([line[1][-1] for line in ridge_lines])
        #Look through every relative maximum found at current row
        #Attempt to connect them with existing ridge lines.
        for ind, col in enumerate(this_max_cols):
            """
            If there is a previous ridge line within
            the max_distance to connect to, do so.
            Otherwise start a new one.
            """
            line = None
            if(len(prev_ridge_cols) > 0):
                diffs = np.abs(col - prev_ridge_cols)
                closest = np.argmin(diffs)
                if diffs[closest] <= max_distances[row]:
                    line = ridge_lines[closest]
            if(line is not None):
                #Found a point close enough, extend current ridge line
                line[1].append(col)
                line[0].append(row)
                line[2] = 0
            else:
                new_line = [[row],
                            [col],
                            0]
                ridge_lines.append(new_line)

        #Remove the ridge lines with gap_number too high
        #XXX Modifying a list while iterating over it.
        #Should be safe, since we iterate backwards, but
        #still tacky.
        for ind in range(len(ridge_lines) - 1, -1, -1):
            line = ridge_lines[ind]
            if line[2] > gap_thresh:
                final_lines.append(line)
                del ridge_lines[ind]

    out_lines = []
    for line in (final_lines + ridge_lines):
        sortargs = np.array(np.argsort(line[0]))
        rows, cols = np.zeros_like(sortargs), np.zeros_like(sortargs)
        rows[sortargs] = line[0]
        cols[sortargs] = line[1]
        out_lines.append([rows, cols])
    return out_lines
    
def boolrelextrema(data, comparator,
                  axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.

    Relative extrema are calculated by finding locations where
    comparator(data[n],data[n+1:n+order+1]) = True.

    Parameters
    ----------
    data: ndarray
    comparator: function
        function to use to compare two data points.
        Should take 2 numbers as arguments
    axis: int, optional
        axis over which to select from `data`
    order: int, optional
        How many points on each side to require
        a `comparator`(n,n+x) = True.
    mode: string, optional
        How the edges of the vector are treated.
        'wrap' (wrap around) or 'clip' (treat overflow
        as the same as the last (or first) element).
        Default 'clip'. See numpy.take

    Returns
    -------
    extrema: ndarray
        Indices of the extrema, as boolean array
        of same shape as data. True for an extrema,
        False else.

    See also
    --------
    argrelmax,argrelmin

    Examples
    --------
    >>> testdata = np.array([1,2,3,2,1])
    >>> argrelextrema(testdata, np.greater, axis=0)
    array([False, False,  True, False, False], dtype=bool)
    """

    if((int(order) != order) or (order < 1)):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[axis]
    locs = np.arange(0, datalen)

    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis, mode=mode)
    for shift in range(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - shift, axis=axis, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        if(~results.any()):
            return results
    return results

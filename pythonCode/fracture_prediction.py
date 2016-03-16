import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

"""
    This module contains function for predicting fracture times in signals.


"""

# apply a fracture predictor to a whole dataset
def applyFracturePredictor(dataSet, fracturePredictor, **kwargs):
    """
       This function takes a dataset dictionary, and returns a dictionary,
       which contains filenames as keys and a list of the predicted locations.

    Inputs
    ------
    fracturePredictor: function
        fracturePredictor is a function which takes an individual signal
        and returns a list of estimated fracture times

    Returns
    -------
    a dictionary which has the same keys as the dataset dictionary,
    and as values has a list of times (maybe an array)

    Note
    ----
    works only on the first channel!


    """

    # create an empty dictionary
    fractureTimes = {}

    for key in dataSet.keys():
        # create an empty dictionary
        fractureTimes[key] = fracturePredictor(dataSet[key][:,0],**kwargs)

    return (fractureTimes)


# ---------- Fracture Predictors ----------------------------


# Prediction based on edges

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


# function to convert a list of function times to a binary vector of fracture locations

# calculate van Rossum between two signals
def dissimilarity(signal1, signal2, *args,dist_type = 'vanRossum'):
    """
    dissimilarity calculates the dissimilarity of two signals

    Inputs
    ------
    signal1: array
    singal2: array
    dist_type: character
        'vanRossum' uses pymuvr package

    Returns
    -------
    scalar:


"""
    if dist_type == 'vanRossum':
    # if True:
        import pymuvr
        D = pymuvr.square_dissimilarity_matrix([[list(signal1)], [list(signal2)]],*args)
    return(D[0][1])

# function which displays the fractures
def displayFractureEstimates(fractureTimes, groundTruth, signal=None):
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='auto')
    index = 1
    offset = 0
    if signal is not None:
        offset = np.max(np.abs(signal/50000))
        ax.plot(np.arange(len(signal))/48000,signal/50000)

    names = fractureTimes.keys()
    bent_labels = [name.split('_')[1] for name in names]
    print(bent_labels)
    b, labels = np.unique(bent_labels, return_inverse=True)

    # colors = mpl.cm.jet
    colors = mpl.cm.rainbow(np.linspace(0,0.8,3))
    for name,l in zip(names,labels):
        ax.plot(fractureTimes[name], index*np.ones_like(fractureTimes[name])+offset, 'o', color = colors[l], markersize=4)
        index += 1
    ax.plot([groundTruth,groundTruth],[0-offset,len(names)+1+offset],'r')
    ax.set_yticks(np.arange(1+offset,len(names)+1+offset))
    ax.set_yticklabels(names)
    #ax.plot()
    ax.set_xlim(30, 40)
    return(ax)

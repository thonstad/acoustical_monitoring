from scipy import signal
import numpy as np

def sync_corr(signal1, signal2, use_envelope = False):
    """ sync_corr(signal1, signal2,use_envelope = False)
        sync_corr synchronizes two signals (1D arrays) based
        on their cross-correlation

        Input
        -----
        signal1: ndarray of shape (n,)
        signal2: ndarray of shape (m,)
        use_envelope: if use_envelope is True the correlation is calculated on the envelopes of the two signals instead of the raw signals; the envelopes are calculated by applying a low-pass Butterworth filter to the absolute value of the signals


        Output
        ------
        offset: integer indicating the offset of signal2 wrt signal1

    """

    # the convolution requires the first signal to be at least as long as the second one
    # so if it is not, we truncate the second signal
    l = len(signal1)

    if use_envelope:
        # Creating a Butterworth filter
        b, a = signal.butter(4, 7./48000, 'low')
        env1 = signal.filtfilt(b, a, np.abs(signal1))
        env2 = signal.filtfilt(b, a, np.abs(signal2))

    # calculating cross-correlation

    # fftconvolve states that the first array needs to be at least as long as the secodn one
    # 'valid' option does not work if the above condition is not satisfied

    if use_envelope:
        cc = signal.fftconvolve(env1,env2[::-1], mode='full')
    else:
        cc = signal.fftconvolve(np.abs(signal1),np.abs(signal2[::-1]), mode='full')


    # finding the maximum correlation
    offset = cc.argmax() + l  - len(cc)

    return(offset)

def find_offset(subset,index_key,other_keys,use_envelope = False):
    ''' offsets = find_offset(subset,index_key,other_keys,use_envelope = False)
    returns the offsets (in indicies) between a single channel and the
    specified channels. Channels (1D arrays) are syncronized by their cross
    correlation.

    Input:
        subset -      rawData dictionary {N entries} of [M,2] arrays.
        index_key -   the key entry in subset used as a reference signal (str).
        other_keys -  list (L,) of keys in subset to compute relative offset
                      on (str).
        use_envelope- if use_envelope is True the correlation is calculated on
                      the envelopes of the two signals instead of the raw signals;
                      the envelopes are calculated by applying a low-pass Butterworth
                      filter to the absolute value of the signals.

    Output:
        offsets -    the relative offset (in count) of the dictionary arrays
                     to the index array. {L entries}

    '''
    # initializes offsets with the offset of the inex_key channel to itself.
    offsets = {index_key : 0}

    # grabs the index_key array from the dictionary.
    signal1 = subset[index_key][:,0]

    # loops throuh channels in other_keys and saves the offsets to the dictionary.
    for chani in other_keys:
        signali = subset[chani][:,0]
        # sync_corr(s1,s2) computes the relative offet of s2 to s1 using
        # the cross corelation of the signals.
        offseti = sync_corr(signal1, signali,use_envelope)
        offsets[chani] = offseti

    return(offsets)

if __name__ == "__main__":

    # importing common packages
    import numpy as np
    from scipy import signal
    import matplotlib.pyplot as plt
    import os

    # importing local packages
    import preprocessing as pp


    # reading all the audio data into a dictionary
    audioPath = os.path.join(os.getcwd(),'..','..','rawAudio')

    filenames, audioDict = pp.readWAV(audioPath)
    filenames, audioDict = pp.readWAV(audioPath)

    # extracting the filenames of interest
    keys = pp.getKeys(filenames, (['B1'],None,['15']))

    # subsetting the data
    subset = pp.loadSubset(audioDict,keys)


    # Note: for now we design sync_envelope to work with two signals
    # However, there may be several time processing steps which could be applied to all of the signals at the same time: instead of two by two.
    # it is possible that even the cross-correlation can be spead up by doing it in ND

    # add test using a specific sine wave or something
    # add test for shifting the same signal


    # we assume the signals are 1D

    signal1 = subset.values()[0][:,0]
    signal2 = subset.values()[1][:,0]

    # calculate offset
    offset = sync_corr(signal1,signal2)

    if offset < 0:
        plt.plot(np.abs(signal1))
        plt.plot(-np.abs(signal2[abs(offset):]))
    else:
        plt.plot(np.abs(signal1[offset:]))
        plt.plot(-np.abs(signal2))
    plt.show()

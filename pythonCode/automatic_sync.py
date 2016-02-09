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

def syncData(offsets,dataDict):
    """
        offsets = syncData(offsets,dataDict)
        syncData function takes a dictionary of offsets and syncs the dataDict based on it. It assumes the keys of offsets and dataDict are the same

        Input
        -----
        offsets: dict
                keys are names of files
                values are corresponding offsets
        dataDict: dict
                keys are names of files
                values are the signal arrays
                # issue - the signals are two dimensional


        Output
        ------
        syncedDataFrame: pandas data frame
            columns: file names
            rows: time steps (supposedly synced)


        Notes
        -----
        # later we could create an object dataset and have offsets and the dataDict as attributes and the syncing functions as methods.

    """
    # make this import global
    import pandas as pd

    # We assume that there will be always an overlap
    print(offsets)
    max_offset = max(offsets.values())
    min_offset = min(offsets.values())
    abs_offset = abs(max_offset - min_offset)

    # extract the length of each signal

    # create the output data frame
    # see if we can make it more efficiently without a loop
    # for now I will extract only the first channel
    # TODO: decide how to store when keep both channels
    # initially we do not know

    # calculate the max length
    lengths = [(len(dataDict[key])- abs_offset-offsets[key]) for (d,key) in zip(dataDict.values(),dataDict.keys())]
    l = min(lengths)
    print(lengths)

    #syncedDataFrame = pd.DataFrame(columns=dataDict.keys())

    for key in dataDict.keys():

        dataDict[key] = dataDict[key][abs_offset + offsets[key] :(l+offsets[key]),0]

    print(dataDict)

    syncedDataFrame = pd.DataFrame(dataDict)
    return(syncedDataFrame)





if __name__ == "__main__":

    # importing common packages
    import numpy as np
    from scipy import signal
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

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


    # Testing the syncData function

    # creating dummy offsets
    offsets = dict(zip(subset.keys(), list(range(len(subset.keys())))))
    syncedDataFrame = syncData(offsets, subset)
    print(offsets)

    def test_syncData():
        # aaumes two channels
        dataDict = {'0':np.array([np.arange(5,20),np.arange(5,20)]),
                    '1':np.array([np.arange(0,15),np.arange(0,15)]),
                    '2':np.array([np.arange(7,19),np.arange(7,19)])}
        offsets = dict(zip(['0','1','2'],[0,5,-2]))
        res = syncData(offsets,dataDict)
        print(res)
    test_syncData()

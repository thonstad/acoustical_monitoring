from .. import automatic_sync as autosync

import numpy as np
import numpy.testing as npt

def test_base():
    return True



# test sync_corr (sometimes these tests fail - remove the randomness)

def test_sync_corr_forward():
    from numpy import random
    x = np.linspace(-2, 2, 200)

    signal1 = (x**5 + 4*x**4 + 3*x**3 + 2*x**2 + x + 1)*np.exp(-x**2)
    # signal1 = random.randn(10)
    signal2 = signal1[2:]
    offset = autosync.sync_corr(signal1, signal2)
    print(offset)
    npt.assert_equal(offset, 2)


def test_sync_corr_backward():
    from numpy import random
    x = np.linspace(-2, 2, 200)

    signal1 = (x**5 + 4*x**4 + 3*x**3 + 2*x**2 + x + 1)*np.exp(-x**2)
    #signal1 = random.randn(10)
    signal2 = signal1[2:]
    offset = autosync.sync_corr(signal2, signal1)
    print(offset)
    npt.assert_equal(offset, -2)


def test_sync_dataset():
    # create a dummy dataset
    dataDict = {'0':np.array([np.arange(5,20),np.arange(5,20)]),
                    '1':np.array([np.arange(0,15),np.arange(0,15)]),
                    '2':np.array([np.arange(7,19),np.arange(7,19)])}

    dataDict = {'1':np.array([np.arange(0,15),np.arange(0,15)]),
                        '2':np.array([np.arange(7,22),np.arange(7,22)])}
    x = np.linspace(-2, 2, 200)

    y = (x**5 + 4*x**4 + 3*x**3 + 2*x**2 + x + 1)*np.exp(-x**2)

    dataDict = {'0':np.array([y[2:],y[2:]]).T,
                '1':np.array([y,y]).T,
                '2':np.array([y[5:],y[5:]]).T}


    offsets, syncedDict = autosync.sync_dataset(dataDict,'0',list(dataDict.keys()))


    npt.assert_equal(syncedDict['0'][:,0],dataDict['0'][3:,0])
    npt.assert_equal(syncedDict['1'][:,0],dataDict['1'][5:,0])
    npt.assert_equal(syncedDict['2'][:,0],dataDict['2'][:,0])



def test_find_offset():
    x = np.linspace(-2, 2, 200)

    y = (x**5 + 4*x**4 + 3*x**3 + 2*x**2 + x + 1)*np.exp(-x**2)

    dataDict = {'1':np.array([y,y]).T,
                            '2':np.array([y[5:],y[5:]]).T}
    offsets = autosync.find_offset(dataDict,'1',['2'])

    print(offsets.values())
    npt.assert_equal(np.array([offsets['1'],offsets['2']]),np.array([0,5])) 

def test_synced_dataset():
    """
        test_synced_dataset tests test_sync_dataset with an input
        which is already synced. Should return a dataset the same as the input one.
    """
    # Note we are testing only the first channel
    # What is a good way to test all the entries of the dictionary at the same time?

    x = np.linspace(-2, 2, 200)
    y = (x**5 + 4*x**4 + 3*x**3 + 2*x**2 + x + 1)*np.exp(-x**2)

    dataDict = {'0':np.array([y,y]).T,
                '1':np.array([y,y]).T,
                '2':np.array([y,y]).T}


    offsets, syncedDict = autosync.sync_dataset(dataDict,'0',['0','1','2'])


    # syncedDict should be the same as dataDict
    vals1 = np.hstack(tuple(dataDict.values()))
    vals2 = np.hstack(tuple(syncedDict.values()))
	
    npt.assert_equal(vals1,vals2) 
    

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:44:00 2016

@author: Travis
"""

import preprocessing as pp

import glob
import os.path
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

#def localmaxmin(t,v,delta):
#
#    #Initializes Table of Minima and Maxima:
#    MaxTable = []
#    MinTable = []
#    #Initializes Local Min at +Infinity and Local Max at -Infinity:
#    LocMin = np.Inf 
#    LocMax = -np.Inf
#    #Initializes Minimum and Maximum Positions
#    MinPos = 0 
#    MaxPos = 0
#    #Initializes Counter which Allows Toggling Between Looking for Minima and Looking for Maxima:
#    tog = 1
#
#    for i in range(len(v)):
#        #Defines the Current Value of v:
#        cval = v[i]
#        #Sets Current Value = LocMax if it is Greater than Existing Value of LocMax:
#        if cval > LocMax:
#            LocMax = cval
#            MaxPos = t[i]
#        #Sets Current Value = LocMin if it is Less than Existing Value of LocMax:
#        if cval < LocMin: 
#            LocMin = cval 
#            MinPos = t[i]
#      
#        #Defines Algorithm for Determining Local Minima and Maxima:
#        if tog != 0:
#            #Finds Local Maximum if Current Value is Beyond Noise Threshold:
#            if cval < LocMax-delta:
#                #Appends Row to MaxTable:
#                MaxTable.append([MaxPos,LocMax])
#                #Resets LocMin to Current Value and MinPos to Current Time:
#                LocMin = cval
#                MinPos = t[i]
#                #Method Found Local Maximum, so Toggle is Set to Next Find Local Minimum
#                tog = 0;
#        else:
#            #Finds Local Minimum if Current Value is Beyond Noise Threshold:
#            if cval > LocMin+delta:
#                #Appends Row to MinTable:
#                MinTable.append([MinPos,LocMin])
#                #Resets LocMax to Current Value and MaxPos to Current Time:
#                LocMax = cval 
#                MaxPos = t[i]
#                #Method Found Local Maximum, so Toggle is Set to Next Find Local Minimum
#                tog = 1;
#                    
#    return MaxTable, MinTable


rawDataPath = os.path.join("..","rawData"); # directory where the audio fles reside

files = glob.glob(os.path.join(rawDataPath,"*.wav"))
names = [];
        
for name in files:
    fileName = os.path.basename(name).split(".")[0]
    names.append(fileName)


filt = (None,None,['17']) # 

audioFiles = pp.getKeys(names,filt);

(names,cDataset) = pp.readWAV(rawDataPath,audioFiles); # opens files and writes to a dictionary

Nf = 24000; # Nyquist freqency in Hz
Fpass = [3200/Nf,3300/Nf]
Fstop = [3100/Nf,3400/Nf]

N, Wn = scipy.signal.ellipord(Fpass,Fstop , 1, 60, False)
b, a = scipy.signal.ellip(N, 1, 60, Wn, 'bandpass')
w, h = scipy.signal.freqs(b, a, np.logspace(-4, 4, 500))


#plt.figure(figsize=(50,20))
#plt.semilogx(w, 20 * np.log10(abs(h)))
#plt.title('Elliptical bandpass filter isolate warning siren')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Amplitude [dB]')
#plt.grid(which='both', axis='both')
#plt.axis([0, 10000, -80, 3])
#plt.show()

t = np.ones(Nf);



plt.figure(figsize=(50,20))

for ii in range(len(cDataset)):
    
    
    rawSignal = cDataset[audioFiles[ii]][:,0]
    filteredSignal = scipy.signal.filtfilt(b,a,rawSignal,padlen=150)
    Nsamp = len(rawSignal)
    
    N, Wn = scipy.signal.ellipord(200/Nf,250/Nf , 1, 60, False)
    b2, a2 = scipy.signal.ellip(N, 1, 60, Wn, 'lowpass')
    smoothedSignal = scipy.signal.filtfilt(b2,a2,filteredSignal**2,padlen=150)
    
    #y= np.convolve(filteredSignal,t,'valid')
    #Ncon = len(y)
    
    time = np.linspace(0,(1/Nf)*Nsamp,Nsamp)
    
    plt.subplot(len(cDataset),2,2*ii+1)
    plt.plot(time,filteredSignal**2,'b')
    #plt.ylim([3000,3800])
    plt.draw()
    
    plt.subplot(len(cDataset),2,2*ii+2)
    #plt.plot(np.linspace(0,(1/Nf)*Ncon,Ncon),y,'b')
    plt.plot(time,filteredSignal/np.max(filteredSignal),'b')
    plt.plot(time,np.cumsum(smoothedSignal)/np.max(np.cumsum(smoothedSignal)),'r')
    plt.plot(time,np.cumsum(smoothedSignal)/np.max(np.cumsum(smoothedSignal))>0.04,'g')
    plt.xlim([50,80])    
    plt.draw()
    
   # MaxTable, MinTable = localmaxmin(time,rawSignal,10)
    
#    plt.subplot(len(cDataset),3,3*ii+3)
#    #plt.plot(np.linspace(0,(1/Nf)*Ncon,Ncon),y,'b')
#    plt.plot(time,rawSignal,[0.7,0.7,0.7])
#    plt.plot(MaxTable[:,0],MaxTable[:,1],'r')
#    plt.plot(MinTable[:,0],MinTable[:,1],'g')
#    plt.draw()
    
    
    




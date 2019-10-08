# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:16:50 2019

@author: project
"""

from scipy.io import wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
#test = [ [i for i in range(0,10)] for x in range(0,10)]
F_s, data = wavfile.read('OneSmallStep.wav')
frames = []

windowed = [[] for x in range(0,int((len(data)/110)))]
for i in range(0,int((len(data)/110))):
    windowed[i]=data[i*110:i*110+275]
    frames.append(i*110)
    frames.append(i*110+275)
#
#fuck = [10 for i in range(len(frames))]
#plt.plot(windowed[1:5])
#for xc in frames:
#    plt.axvline(x=xc, color = 'k',linestyle = '--')
    
data = [1,58,14,6,61,8,4,61,65,96,46,4,7,5,6,9,1,6,8,14,6,84]

weight = np.repeat(1.0, 8)/8

smash = np.convolve(data,weight, 'valid')


plt.plot(smash)
plt.plot(data)
    
    
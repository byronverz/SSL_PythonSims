# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:55:51 2019

@author: project
"""

from scipy.io import wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy import fft,arange
import numpy as np
import math as math
from scipy.interpolate import interp1d
import librosa
import librosa.display as libdisp
from sklearn.preprocessing import MinMaxScaler

F_s, data = wavfile.read('OneSmallStep.wav')
y, s = librosa.load('OneSmallstep.wav')

def windowing(sequence,frame_len,sampling):
    frames = []
    windows =[[] for x in range(0,int((len(sequence)/sampling)))]
    for i in range(0,int((len(sequence)/sampling))):
        windows[i]=sequence[i*sampling:i*sampling+frame_len]
        frames.append(i*sampling)
        frames.append(i*sampling+frame_len)
    return windows,frames

def highpass_filter(y, sr):
  filter_stop_freq = 100  # Hz
  filter_pass_freq = 200  # Hz
  filter_order = 1001
            

  # High-pass filter
  nyquist_rate = sr / 2.
  desired = (0, 0, 1, 1)
  bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
  filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)

  # Apply high-pass filter
  filtered_audio = signal.filtfilt(filter_coefs, [1], y)
  return filtered_audio

out = highpass_filter(data,F_s)
out_n = out/np.max(out)
windowed,frame = windowing(out,275,110)

test = np.zeros(664125).reshape(2415,275)   
output = [[]for x in range(0,int(len(data)))]
for i in range(0,len(windowed)-1):
   currentWindow = windowed[i]
   fftCurrentWindow = np.float16(np.abs((np.fft.fft(currentWindow))))
#   output[i]=np.abs(fftCurrentWindow)
   test[i] = np.abs(fftCurrentWindow)
scaler = MinMaxScaler(feature_range=(0,1))
normalizedData = scaler.fit_transform(test)
plt.imshow(test[0:100])
#plt.plot(test[5])



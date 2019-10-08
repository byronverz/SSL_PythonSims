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

F_s, data1 = wavfile.read('OneSmallStep.wav')
data = np.append(data1,np.zeros(83))
y, s = librosa.load('OneSmallstep.wav')
data_f = np.fft.fft(data)

################################################################################################################################################################################################
################################################################################################################################################################################################

##################   Start of functions    ############################

################################################################################################################################################################################################
################################################################################################################################################################################################

def frame_energy(windowed_sequence,window_length):
    energies = []
    for j in range(0,len(windowed_sequence)-1):
        energy = 0
        for i in range(0,window_length):
            energy = energy + (1/window_length)*(windowed_sequence[j][i]**2)
        energies.append(energy)
##################   For plotting    ############################
#    t = np.arange(0,110*len(energies),110)
#    t_new = np.arange(110*len(energies)-1)
#    output = np.interp(t_new,t,energies)
##################   For plotting    ############################        
    return energies
    
def windowing(sequence,frame_len,sampling):
    frames = []
    windows = [[np.zeros(275)] for x in range(0,int((len(sequence)/sampling)))]
    
    for i in range(0,int((len(sequence)/sampling))-1):
        windows[i]=sequence[i*sampling:(i*sampling)+frame_len]
        frames.append(i*sampling)
        frames.append(i*sampling+frame_len)
    return windows,frames


def highpass_filter(y, sr):
    filter_stop_freq = 100  # Hz
    filter_pass_freq = 200  # Hz
    filter_order = 1001
    nyquist_rate = sr / 2.
    desired = (0, 0, 1, 1)
    bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
    filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)
    filtered_audio = signal.filtfilt(filter_coefs, [1], y)
    return filtered_audio


def simple_ma(array, window_len):
    weighting = np.repeat(1.,window_len)/window_len
    smash = np.convolve(array,weighting,'same')
    return smash
    

def exponential_ma(array, window_len):
    weighting = np.exp(np.linspace(-1.,0,window_len))
    weighting /= weighting.sum()
    e = np.convolve(array,weighting,'same')[:len(array)]
    e[:window_len] = e[window_len]
    return e
    

def dominantFreq(signal,samplingF):
    length = len(signal) 
    k = arange(length)
    T = length/samplingF
    freq = k/T 
    freq = freq[range(int(length/2))] 
    Y = fft(signal)/length 
    Y = Y[range(int(length/2))]
    maxIndex = np.argmax(Y)
    return freq[maxIndex]

################################################################
##############     Don't need I think      ##############

#def STFFT (audio_sig, frame_len,num_frames):
#    output = np.zeros((num_frames)*frame_len).reshape(num_frames,frame_len)
#    complexOut= np.zeros(256*num_frames,dtype=complex).reshape(num_frames,256)
#    for i in range(0,len(audio_sig)-1):
#        currentWindow = audio_sig[i]
#        temp = np.fft.rfft(currentWindow,n=512)
#        temp2 = np.append(temp,np.zeros(18))
#        output[i]= np.float16(temp2)
#        temp3 = np.fft.fft(currentWindow,n=512)
#        complexOut[i] = temp3[0:256]
#        scaler = MinMaxScaler(feature_range=(0,1))
#        normalizedRealData = scaler.fit_transform(output)
#    return normalizedRealData, complexOut

################################################################


def spectral_centroid(windowedSequence,frame_len,num_frames,Fs):
    centroid = np.zeros((num_frames)).reshape(num_frames,1)
    for i in range(0,len(windowedSequence)-1):
        temp = np.abs(np.fft.rfft(windowedSequence[i],n=512))
        output = np.append(temp,np.zeros(18))
        freqs = np.abs(np.fft.fftfreq(len(output),1/Fs)[:(len(out)//2) + 1])
#        print(len(freqs))
        spectrum = (output/np.sum(output))
        centroid[i] = np.sum(freqs*spectrum)
    return centroid
        
    
def VAD_mask(audio_signal,Fs):
    windowed,frame = windowing(out,275,110)
    ste = frame_energy(windowed,275)
#    print(len(ste)) <-- 2416
    centr = spectral_centroid(windowed,275,len(windowed),Fs)
#    print(len(centr[3]))
    centrDiff = differential(centr)
#    print(len(centrDiff[3]))
    steMask = []
    centrMask = []
    for i in range(0,len(ste)):
        if np.abs(ste[i]) > np.average(ste):
            steMask.append(1)
            
        else:
            steMask.append(0)
    for j in range(0,len(centrDiff)):
        if np.abs(centrDiff[j])>300:
            centrMask.append(1)
        else:
            centrMask.append(0)
    vadMask = np.logical_or(steMask,centrMask)
    for j in range(0,len(vadMask)):
        if vadMask[j]==1:
            vadMask[j-15:j-1] = 1
#    t = np.arange(0,len(audio_signal),1)
#    t_new = np.arange(110*len(energies)-1)
#    output = np.interp(t_new,t,energies)
    return steMask,centrMask,centrDiff,centr,vadMask



def differential(arr):
    diff = [0]
    for i in range(2,len(arr)):
        diff.append(arr[i]-arr[i-2])
    return diff
################################################################
    #Not working yet
    
#def periodogram(complexDFT):
#    power = np.empty_like(complexDFT,dtype=np.float16)
#    scaler = MinMaxScaler(feature_range=(0,1))
#    complexDFT1 = scaler.fit_transform(np.abs(complexDFT))
#    for i in range(0,len(complexDFT)):
#        power[i] = ((complexDFT1[i])**2)/512
#        print(np.max(power[i]))
#    return np.float16(power)
###############################################################

################################################################################################################################################################################################
################################################################################################################################################################################################
    
#############  End of functions  ############## 

################################################################################################################################################################################################
################################################################################################################################################################################################

################################################################################################################################################################################################
################################################################################################################################################################################################

#############  Start of implementation  ##############

################################################################################################################################################################################################
################################################################################################################################################################################################

out = highpass_filter(data,F_s)
out_n = out/np.max(out)
out_f = fft(out_n)
fDom = dominantFreq(out_n,F_s)
energyMask, centroidMask,diff,centroid,mask = VAD_mask(data,F_s)
#shortFFT1 = np.abs(librosa.core.stft(out_n,win_length = int(275)))
windowed,frame = windowing(out,275,110)
#shortFFTR,shortFFTI = STFFT(windowed,len(windowed[0]),len(windowed))
#print(shortFFT)
#centr = librosa.feature.spectral_centroid(data,F_s)
#centrDiff = differential(np.ndarray.flatten(centr))
threshold = np.full(len(centroidMask),300)
#out_sma = exponential_ma(np.ndarray.flatten(centr),110)
#out_ema = exponential_ma(ste2,110*50)
#flatness = np.ndarray.flatten(librosa.feature.spectral_flatness(out))
#stDomFreq = []
#for i in range(0,len(windowed)):
#    tempF = dominantFreq(np.fft.fft(windowed[i]),F_s)
#    stDomFreq.append(tempF)
#tempF = dominantFreq(windowed[2],F_s)
mfcc1 = librosa.feature.mfcc(out_n,F_s)
mfcc2 = librosa.feature.mfcc(y,s)
maskTime = np.arange(0,2416,11025)
#maskNew = np.interp(maskAxis,maskNewAxis,mask)
################################################################################################################################################################################################
################################################################################################################################################################################################
    
#############  End of implementation  ############## 

################################################################################################################################################################################################
################################################################################################################################################################################################


################################################################################################################################################################################################
################################################################################################################################################################################################

#############  Start of plotting  ##############

################################################################################################################################################################################################
################################################################################################################################################################################################

#t = np.linspace(0,24.11265,265842)
#


#plt.plot(ste2,label = 'ste',color = 'green')
#plt.plot(t,out,label='audio',color = 'blue')    
#plt.plot(out_sma,label='sma',color = 'purple')
#plt.plot(out_ema,label='ema',color = 'purple')
#
#plt.plot(flatness,label = 'SFM',color = 'c')
#for xc in frame:
#    plt.axvline(x=xc, color = 'k',linestyle = '--')
#libdisp.specshow(librosa.amplitude_to_db(shortFFT1,ref = np.max),y_axis='log',x_axis='time')
#plt.subplot(311)
#plt.plot(centroid)
#plt.plot(np.abs(diff))
#plt.plot(threshold)
plt.subplot(211)
plt.xticks(maskTime)
#plt.plot(energyMask,color = 'red')
#plt.plot(centroidMask,color='purple')
#plt.plot(mask2,color='k')
plt.plot(mask,color='red')

#plt.plot(out_sma)
#libdisp.specshow(mfcc1,x_axis='time')
plt.subplot(212)
#libdisp.specshow(mfcc2,x_axis='time')
plt.axvspan(41024/11025, 62324/11025, color='red', alpha=0.5)
plt.axvspan(170071/11025, 198769/11025, color='red', alpha=0.5)
plt.axvspan(228184/11025, 258095/11025, color='red', alpha=0.5)
plt.xlabel("Time(s)")
plt.plot(np.arange(len(data))/ 11025,data)

#plt.plot(out_f)
#plt.legend()
#plt.show()

################################################################################################################################################################################################
################################################################################################################################################################################################
    
#############  End of plotting  ############## 

################################################################################################################################################################################################
################################################################################################################################################################################################

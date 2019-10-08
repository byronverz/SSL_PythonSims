# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:18:30 2019

@author: project
"""
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import scipy.stats as st
#from sklearn.preprocessing import MinMaxScaler

#######################     Functions      ####################################
def highpass_filter(y, sr):
    filter_stop_freq = 50  # Hz
    filter_pass_freq = 80  # Hz
    filter_order = 1001
    nyquist_rate = sr / 2.
    desired = (0, 0, 1, 1)
    bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
    filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)
    filtered_audio = signal.filtfilt(filter_coefs, [1], y)
    normalizedAudio = filtered_audio/np.max(filtered_audio)
    return normalizedAudio

def idealDOA (arrayPos,array, sourcePos):
    global c
    
    mics = np.empty_like(array)
    distance = np.zeros(array.shape[0])
    for i in range(0,array.shape[0]):
        for j in range(0,array.shape[1]):
            mics[i][j] = arrayPos[j]+array[i][j]
    for t in range(0,mics.shape[0]):
        distance[t] = np.sqrt((sourcePos[0]-mics[t][0])**2+(sourcePos[1]-mics[t][1])**2)
    times = np.array([p/343 for p in distance ])
    times = np.insert(times,times.shape,times[0])
    t_d = np.abs(np.diff(times))
    theta = [(np.arcsin((c*n)/((array[1][0]-array[2][0])))) for n in t_d]
    return t_d, theta,times

def CCV_DOA(audioArray,F_s = 44100):
    global c
    filtered = np.empty_like(audioArray)
#    convolution = np.empty_like(audioArray)
#    tau = np.empty(audioArray.shape[0])
#    theta = np.empty(audioArray.shape[0])
#    ccv = np.empty(4).reshape(2,2)
    for i in range(0,audioArray.shape[0]):
        filtered[i] = highpass_filter(audioArray[i],F_s)
    convol1 = np.correlate(filtered[0],filtered[1],mode='full')
    convol2 = np.correlate(filtered[1],filtered[2],mode='full')   
    convol3 = np.correlate(filtered[2],filtered[0],mode='full')    
    tau1 = (np.argmax(convol1)-len(convol1)/2)/(F_s)
    tau2 = (np.argmax(convol2)-len(convol2)/2)/(F_s)
    tau3 = (np.argmax(convol3)-len(convol3)/2)/(F_s)
    taus = np.array([tau1,tau2,tau3])
    theta1 = np.arcsin(c*tau1/0.1)
    theta2 = np.arcsin(c*tau2/0.1)
    theta3 = np.arcsin(c*tau3/0.1)
    print(np.rad2deg(theta1),np.rad2deg(theta2),np.rad2deg(theta3))
#    filtered = np.insert(filtered,filtered.shape[0],filtered[0],axis = 0)
#    for j in range(0,audioArray.shape[0]):
#        ccv= np.corrcoef(filtered[j],filtered[j+1])
#        convolution[j] = fftconvolve(filtered[j],filtered[j+1],mode = 'same')
#        tau[j] = np.argmax(convolution[j])
#        theta = np.arcsin((c*tau[j])/(F_s*0.10))
    return theta1,theta2,theta3,taus

#def CCF_DOA(F_s, audioArray):
#    global c
#    ffts = np.empty_like(audioArray)
#    convolution = np.empty_like(audioArray)
#    tau = np.empty(audioArray.shape[0])
#    theta = np.empty(audioArray.shape[0])
##    ccv = np.empty(4).reshape(2,2)
#    for i in range(0,audioArray.shape[0]):
#        ffts[i] = np.fft.fft(highpass_filter(audioArray[i],F_s))
#    PHAT1 = ffts[0]*np.conjugate(ffts[1])/np.abs(ffts[0]*np.conjugate(ffts[1]))
#    PHAT2 = ffts[1]*np.conjugate(ffts[2])/np.abs(ffts[1]*np.conjugate(ffts[2]))
#    PHAT3 = ffts[2]*np.conjugate(ffts[0])/np.abs(ffts[2]*np.conjugate(ffts[0]))
#######################     End Functions      ################################

#######################     Processing     ####################################
    
    ############        Variables       ################
fs = np.zeros(3)  

fs, reference = wavfile.read('sentence001Short.wav')
frontSource = np.zeros(3*len(reference)).reshape(3,len(reference))
frontSource[0] = reference
frontSource[1] = np.insert(reference[:-11], 0, np.random.normal(0,256,11))
frontSource[2] = np.insert(reference[:-11], 0, np.random.normal(0,256,11))

rightSource = np.zeros_like(frontSource)
rightSource[0] = np.insert(reference[:-6],0,np.random.normal(0,256,6))
rightSource[1] = reference
rightSource[2] = np.insert(reference[:-11],0,np.random.normal(0,256,11))

backSource = np.zeros_like(frontSource)
backSource[0] = np.insert(reference[:-11],0,np.random.normal(0,256,11))
backSource[1] = reference
backSource[2] = reference

leftSource = np.zeros_like(frontSource)
leftSource[0] = np.insert(reference[:-6],0,np.random.normal(0,256,6))
leftSource[1] = np.insert(reference[:-11],0,np.random.normal(0,256,11))
leftSource[2] = reference

#FNAudio = highpass_filter(audio,fs) 

#source = [[10,15],
#          [15,10],
#          [5,10],
#          [10,5]] #Position of speaker in field
source = [[5,10]]
mic_pos = [10,10]  #Position of mics in field
#if source[0] == mic_pos[0] :
#    if source[1] == mic_pos[1]:
#        raise Exception('Source and Mic array have same position: {} and {}'.format(source, mic_pos))
#    sourceAngle = np.pi/2
#else:
#    sourceAngle = np.arctan((source[1]-mic_pos[1])/(source[0]-mic_pos[0]))
    
    
c = 343 #m/s

fRange = np.arange(20,4001)


    ###########     End variables    ###############
    
    
positions = [[0.,0.043],[-0.050,-0.043],[0.050,-0.043]]
mic_array = np.array(positions)  #Position of mics relative to mic_pos/origin of array
fieldImage = np.zeros(20*20).reshape(20,20) #Field of inspection (dimensions in meters(x,y))
fieldImage[source[0][0]][source[0][1]] = 255
fieldImage[mic_pos[0]][mic_pos[1]] = 125


#for x in source:
#    diffs ,angles,timeToMic = idealDOA(mic_pos,mic_array,x)
#    angles[0] = angles[0] + np.pi/3
#    angles[2] = angles[2] - np.pi/3
#    for i in range(3):
#        if angles[i]<0:
#            angles[i] = angles[i] + 2*np.pi
#            if angles[i] > np.pi:
#                angles[i] = angles[i] - np.pi
##    print(timeToMic)
#    fieldImage[x[0]][x[1]] = 255
#    rad = [0,1]
#    plt.subplot(121)
#    ax = plt.subplot(polar = True)
#    ax.plot([0,angles[0]],rad,'r--')
#    ax.plot([0,angles[1]],rad,'g--')
#    ax.plot([0,angles[2]],rad,'b--')
##    ax.plot([0,sourceAngle-(np.pi/2)],r,'y*')
#    plt.subplot(122)
#    plt.imshow(fieldImage)
    

    
output0,output1,output2,tau_1 = CCV_DOA(leftSource)
#conv1 = fftconvolve(mic[0],mic[1],mode='same')
#max1 = np.argmax(conv1)/fs[0]
#conv2 = fftconvolve(mic2,mic3,mode='full')
#max2 = np.argmax(conv2)/fs[1]
#conv3 = fftconvolve(mic3,mic1,mode='full')
#max3 = np.argmax(conv3)/fs[2]
#######################  End Processing     ###################################
r = [0,1]
ax = plt.subplot(121,polar = True)
ax.plot([0,output0],r,'r--')
ax.plot([0,output1],r,'g--')
ax.plot([0,output2],r,'b--')
##ax.plot([0,sourceAngle-(np.pi/2)],r,'y*')
plt.subplot(122)
plt.imshow(fieldImage)
#plt.subplot(311)
#plt.plot(out)
#plt.subplot(312)
#plt.plot(output1)
#plt.subplot(313)
#plt.plot(output2)
#plt.subplot(311)
#plt.plot(conv1)
#plt.subplot(312)
#plt.plot(conv2)
#plt.subplot(313)
#plt.plot(conv3)
#print(max1,max2,max3)
#plt.subplot(311)
#plt.plot(mic[0])
#plt.subplot(312)
#plt.plot(mic[1])
#plt.subplot(313)
#plt.plot(mic[2])


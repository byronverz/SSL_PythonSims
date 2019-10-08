import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import scipy.stats as st

# =============================================================================
#                           Function Definitions
# =============================================================================

def highpass_filter(y, sr):
    filter_stop_freq = 50  # Hz
    filter_pass_freq = 80  # Hz
    filter_order = 9
    nyquist_rate = sr / 2.
    desired = (0, 0, 1, 1)
    bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
    filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)
    filtered_audio = signal.filtfilt(filter_coefs, [1], y)
    normalizedAudio = filtered_audio/np.max(filtered_audio)
    return normalizedAudio

def idealTDOA (arrayPos,array, sourcePos):
# =============================================================================
#    Arguments:
#     arraPos: Position of array origin in field
#     array: microphone positions in field
#     sourcePose: Position of the sound source
#    Returns:
#     t_d: Time differences between microphones
#     times: Time of arrival to each microphone
# =============================================================================
    global c
    
    mics = np.empty_like(array)
    distance = np.zeros(array.shape[0])
    for i in range(0,array.shape[0]):
        for j in range(0,array.shape[1]):
            mics[i][j] = arrayPos[j]+array[i][j]
    for t in range(0,mics.shape[0]):
        distance[t] = np.sqrt((sourcePos[0]-mics[t][0])**2+(sourcePos[1]-mics[t][1])**2)
    times = np.array([p/343 for p in distance ])
    t_d = np.abs(np.diff(times))
    return t_d, times,distance


def CCV_DOA (micArray,AudioArray, Fs):
# =============================================================================
#   Arguments:
#       AudioArray: n x m array of audio where n is the number of mics and m 
#                  the length of the audio in number of samples
#       Fs: Sampling frequency of audio
#   Returns:
#       ccv: cross correlation vector array between microphones
#       theta: angle estimation of microphone pairs
# =============================================================================
    global c
    filtered = np.empty_like(AudioArray)
    ccv = np.zeros((AudioArray.shape[0],(AudioArray.shape[1]*2)-1))
    tau = np.empty(AudioArray.shape[0])
    theta = np.empty(AudioArray.shape[0])
    for i in range(0,AudioArray.shape[0]):
        filtered[i] = highpass_filter(AudioArray[i],Fs)
    ccv[0] = np.correlate(filtered[0],filtered[1],mode='full')
    ccv[1] = np.correlate(filtered[1],filtered[2],mode='full')   
    ccv[2] = np.correlate(filtered[2],filtered[3],mode='full')
    ccv[3] = np.correlate(filtered[3],filtered[0],mode='full')
    for x in range(0,len(ccv)):
        tau[x] = (np.argmax(ccv[x])-int(len(ccv[x])/2))/(Fs)
    for k in range(0,len(tau)):
        if tau[k]>0.000271:
            theta[k] = np.arcsin(c*0.000271/0.093)
        elif tau[k]<-0.000271:
            theta[k] = np.arcsin(c*(-0.000271)/0.093)
        else:
            theta[k] = np.arcsin(c*tau[k]/0.093)
    if tau[0]<0:
        theta[1] = (-theta[1] + 2*np.pi)+np.pi
    elif tau[0]>0:
        theta[3] = (-theta[3] + 2*np.pi)+np.pi
    if tau[1]<0:
        theta[2] = (-theta[2] + 2*np.pi)+np.pi
    elif tau[1]>0:
        theta[0] = (-theta[0] + 2*np.pi)+np.pi
    return ccv,theta,tau,filtered
   
def fieldDisplay(field, mics, micPosition,sourcePosition,locateAngle):
# =============================================================================
#    Arguments:
#       field: 2D array representing field image 
#       mics: mic position w.r.t the array origin
#       micPosition: position of mic array origin
#       sourcePosition: location of source in field
#    Returns:
#       field: Image of field with details
# =============================================================================
    mic_arrayDisp = mics.astype(int)
    micsDisp = np.empty_like(mic_arrayDisp)
    for i in range(0,mic_arrayDisp.shape[0]):
        for j in range(0,mic_arrayDisp.shape[1]):
            micsDisp[i][j] = 10*micPosition[j]+1000*mics[i][j]
    for p in micsDisp:
        field[p[0]][p[1]] = 125
        
        
    field[10*sourcePosition[0]][10*sourcePosition[1]] = 255
    field[10*sourcePosition[0]-1][10*sourcePosition[1]] = 255
    field[10*sourcePosition[0]+1][10*sourcePosition[1]] = 255
    field[10*sourcePosition[0]][10*sourcePosition[1]+1] = 255
    field[10*sourcePosition[0]][10*sourcePosition[1]-1] = 255    
    
    
    field[10*micPosition[0]][10*micPosition[1]] = 125
    field[11*micPosition[0]-9][10*micPosition[1]] = 125
    field[9*micPosition[0]+9][10*micPosition[1]] = 125
    field[10*micPosition[0]][11*micPosition[1]-9] = 125
    field[10*micPosition[0]][9*micPosition[1]+9] = 125
    return field

def simulateTDOA(audioFile,Fs,timeOfArrival,numMics):
# =============================================================================
#   Arguments:     
#       audioFile: Sample audio to time shift
#       timeOfArrival: array of the arrival time to mics
#       numMics: Number of mics
#       Fs: sampling frequency
# =============================================================================
    reference = np.zeros((numMics,len(audioFile)))
    delays = np.empty(len(timeOfArrival+1))
    delays = timeOfArrival*Fs
    for x in range(numMics):
        reference[x] = np.insert(audioFile[:-int(round(delays[x]))],0,np.random.normal(0,10,int(round(delays[x]))))
    
    return reference, delays
    
    
# =============================================================================
#                       End Function Definitions 
# =============================================================================

#******************************************************************************

# =============================================================================
#                                   Main
# =============================================================================

c = 343 #speed of sound in m/s
fieldImage = np.zeros((200,200)) # Field of inspection
mic_pos = np.array([10,10])
mic_array = np.array([[-0.0465,0.0465],[0.0465,0.0465],[0.0465,-0.0465],[-0.0465,-0.0465]])
source_pos = np.array([19,18])

timeDiffs, t,dist = idealTDOA(mic_pos,mic_array,source_pos)

fs, audio = wavfile.read('sentence001.wav')
#if (len(audio)%2)==0:
#    print("Last sample deleted")
#    audioCopy = audio[:-1].copy()
#    offsetAudio,d = simulateTDOA(audioCopy, fs, t, mic_array.shape[0])
#else:
#    print("Last sample kept")
offsetAudio,d = simulateTDOA(audio, fs, t, mic_array.shape[0])
   
vector, angle,t_d,filteredAudio = CCV_DOA(mic_array,offsetAudio, fs)
fieldDisp = fieldDisplay(fieldImage,mic_array,mic_pos,source_pos,angle)
plt.subplot(335)
plt.imshow(fieldDisp)
r = [0,1]
midRight = plt.subplot(336,polar = True)
midRight.set_theta_direction(-1)
midRight.plot([0,angle[0]],r,'r--')
midBottom = plt.subplot(338,polar=True)
midBottom.set_theta_zero_location("S")
midBottom.set_theta_direction(-1)
midBottom.plot([0,angle[1]],r,'g--')
midLeft = plt.subplot(334,polar=True)
midLeft.set_theta_direction(-1)
midLeft.set_theta_zero_location("W")
midLeft.plot([0,angle[2]],r,'b--')
midTop = plt.subplot(332,polar = True)
midTop.set_theta_zero_location("N")
midTop.set_theta_direction(-1)
midTop.plot([0,angle[3]],r,'k--')





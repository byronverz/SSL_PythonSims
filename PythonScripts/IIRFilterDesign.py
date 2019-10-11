# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


def TF2SOS (b_mat,a_mat):
# =============================================================================
#    Func: Convert tf form of IIR filter coeffs into second order system form
#           in the same format as what arm_biquad_cascade_df1_init_f32 function
#           in the CMSIS-DSP library requires
#    Input: b,a : b (numerator) and a (denominator) coefficients in tf form
#    Output: (N/2) x 5 array where N is the filter order
# =============================================================================
    zeros, poles, filterGain = signal.tf2zpk(b,a)
    z = rootSort(zeros)
    p = rootSort(poles)
    M = len(z)
    N = len(p)
    normFactor = max(len(b)-1,len(a)-1)
    
    gainPerStage = filterGain **(2/normFactor)
    
    
def rootSort(root):
    sortedArr = np.zeros_like(root)
    frontIndex = 0
    revIndex = -1
    for i in range(len(root)):
        if root[i].imag == 0:
            sortedArr[revIndex] = root[i]
            revIndex -= 1
        else:
            sortedArr[frontIndex] = root[i]
            frontIndex += 1
    return sortedArr
def cmsis_Coefficient_Header (b_mat,a_mat, filename):
# =============================================================================
#     Function: Write a file to include in project for cmsis-dsp based IIR
#               filter for stm32f3 platform
#    -- Inputs:
#           - a_mat: a coefficients matrix in higher order form
#           - b_mat: b coefficients matrix in higher order form
#           - filename: Output filename
# =============================================================================
    sos_mat = signal.tf2sos(b_mat,a_mat)
    nStages, cols = sos_mat.shape 
    file = open(filename,'w')   #Open file in write format
    file.write('#include <stdint.h> \n')
    file.write('#ifndef STAGES \n #define STAGES {} \n'.format(nStages))
    file.write('#endif \n')
    file.write('float32_t IIRCoeffs[{}]= {\n'.format(5*nStages))
#    for x in range(nStages):
#        file.write('%15.12f,%15.12f,%15.12f,%15.12f,%15.12f,%15.12f \n'%\(sos_mat[x,0]))
    
    
    
    
    
    
    
    
Fs = 42000                  ## Sampling frequency in Hz
Fn = Fs/2                   ## Nyquist frequency in Hz
Ws = 42000*np.pi*2          ## Sampling frequency in rad/s
Wn = Ws/2                   ## Nyquist frequency in rad/s
f_cut = 50                  ## Cutoff frequency in Hz
w_cut = f_cut*2*np.pi       ## Cutoff frequency in rad/s
w_cut_norm = w_cut/Wn       ## Normalized cutoff frequency
f_stop = 5                  ## Stopband frequency in Hz
w_stop = f_stop *2*np.pi    ## Stopband frequency in rad/s
w_stop_norm = w_stop/Wn     ## Normalized stopband frequency
rpass = 0.1                 ## Passband ripple
rstop = 60                  ## Stopband attenuation

N, Wn = signal.cheb2ord(w_cut_norm,w_stop_norm,rpass,rstop)
b, a = signal.cheby2(N,rstop, Wn, 'high', analog=False ,output='ba')


w, h = signal.freqz(b,a,2**12)
w *= Fs/(2*np.pi)

hline = [-3 for i in range(0,len(h))]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(w, 20*np.log10(abs(h)))
ax.axvline(x=30,color='red',linestyle='--')
ax.axvline(x=6,color='red',linestyle='--')
ax.plot(w, hline, color = 'red', linestyle = '--')
ax.set_xscale('log')
ax.set_title("Cheby2 IIR highpass filter")
ax.set_xlabel("Frequency [rad/s]")
ax.set_ylabel("Amplitude [dB]")
ax.axis((0.1,50000,-60,20))
ax.grid(which='both',axis='both')
plt.show()
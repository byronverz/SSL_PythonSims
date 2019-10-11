# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:58:08 2019

@author: byron
"""

import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

def tf2sos(b,a):
     """
     Cascade of second-order sections (SOS) conversion.
     Convert IIR transfer function coefficients, (b,a), to a
     matrix of second-order section coefficients, sos_mat. The
     gain coefficients per section are also available.
     SOS_mat,G_array = tf2sos(b,a)
    
     b = [b0, b1, ..., bM-1], the numerator filter coefficients
     a = [a0, a1, ..., aN-1], the denominator filter ceofficients
    
     SOS_mat = [[b00, b01, b02, 1, a01, a02],
     [b10, b11, b12, 1, a11, a12],
     ...]
     G_stage = gain per full biquad; square root for 1st-order stage
    
     where K is ceil(max(M,N)/2).
    
     Mark Wickert March 2015
     """
     Kactual = max(len(b)-1,len(a)-1)
     Kceil = 2*int(np.ceil(Kactual/2))
     z_unsorted,p_unsorted,k = signal.tf2zpk(b,a)
     z = shuffle_real_roots(z_unsorted)
     p = shuffle_real_roots(p_unsorted)
     M = len(z)
     N = len(p)
     SOS_mat = np.zeros((Kceil//2,6))
     # For now distribute gain equally across all sections
     G_stage = k**(2/Kactual)
     for n in range(Kceil//2):
         if 2*n + 1 < M and 2*n + 1 < N:
             SOS_mat[n,0:3] = np.array([1,-(z[2*n]+z[2*n+1]).real,(z[2*n]*z[2*n+1]).real])
             SOS_mat[n,3:] = np.array([1,-(p[2*n]+p[2*n+1]).real,(p[2*n]*p[2*n+1]).real])
             N = 5
             SOS_mat[n,0:3] = SOS_mat[n,0:3]*G_stage
         else:
             SOS_mat[n,0:3] = np.array([1,-(z[2*n]+0).real, 0])
             SOS_mat[n,3:] = np.array([1,-(p[2*n]+0).real, 0])
             SOS_mat[n,0:3] = SOS_mat[n,0:3]*np.sqrt(G_stage)
     return SOS_mat, G_stage
 
def shuffle_real_roots(z):
     """
     Move real roots to the end of a root array so
     complex conjugate root pairs can form proper
     biquad sections.
    
     Need to add root magnitude re-ordering largest to
     smallest or smallest to largest.
    
     Mark Wickert April 2015
     """
     z_sort = np.zeros_like(z)
     front_fill = 0
     end_fill = -1
     for k in range(len(z)):
         if z[k].imag == 0:
             z_sort[end_fill] = z[k]
             end_fill -= 1
         else:
             z_sort[front_fill] = z[k]
             front_fill += 1
     return z_sort 
 
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
sos_mat = signal.tf2sos(b,a)
z,p ,gain1 = signal.tf2zpk(b,a)
sos_matrix ,gain = tf2sos(b,a)
print('SOS matrix 1: \n')
print(sos_mat)
print(gain1)
print('\n')
print('SOS matrix 2: \n')
print(sos_matrix)
print(gain)







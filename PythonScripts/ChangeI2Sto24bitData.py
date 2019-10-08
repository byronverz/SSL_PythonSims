import numpy as np
import matplotlib.pyplot as plt

fs = 16e3

with open("track.i2s", "rb") as fil:
	arr = fil.read()
	
# Change to binary
arr2 = ['{0:08b}'.format(i) for i in arr]

# Break it in 4x8 bits
arr = [''.join(arr2[i:i+4]) for i in np.arange(0, len(arr2), 4)]

# Make all ints
makeSmall = 2.0**32
arr2 = [int(i, 2)/makeSmall for i in arr]

arrRight = []
arrLeft = []

#Split Channels
for i in np.arange(0,len(arr2),2):
	arrLeft.append(arr2[i])
	arrRight.append(arr2[i+1])
	
	
t = np.arange(len(arrLeft))/fs
sp = np.fft.fft(arrRight)
sp2 = np.fft.fft(arrLeft)
freq = np.fft.fftfreq(t.shape[-1])
plt.plot(freq, np.abs(sp), freq, np.abs(sp2))

plt.show()
	
	
plt.subplot(211)
plt.title("Left Channel")
plt.plot(np.arange(len(arrLeft))/fs, arrLeft)
plt.subplot(212)
plt.title("Right Channel")
plt.plot(np.arange(len(arrRight))/fs, arrRight)
plt.show()


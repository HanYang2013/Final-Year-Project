# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 21:31:24 2016

@author: hanyang
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import fftpack
import scipy.signal as signal
from pywt import wavedec
from sklearn.decomposition import FastICA

#For Interictal data
"""
train110='E:/EEG/train_1/1_1_0.mat'  
data1=sio.loadmat(train110) 
oac_struct=data1['dataStruct']
oac_struct.shape
val=oac_struct[0,0]
EEG=val['data']"""

train111='E:/EEG/train_1/1_1_1.mat'  
data2=sio.loadmat(train111) 
oac_struct=data2['dataStruct']
oac_struct.shape
val1=oac_struct[0,0]
EEG1=val1['data']
lin = map(sum,EEG1)

x=EEG1[:,0] 
for i in range(1,16):
    x = EEG1[:,i]+x
    
coeffs = wavedec(x, 'db7', level=8)
cA8, cD8,cD7,cD6,cD5, cD4 ,cD3 ,cD2 ,cD1 = coeffs


meanD8=np.mean(abs(cD8))
meanD7=np.mean(abs(cD7))
meanD6=np.mean(abs(cD6))
meanD5=np.mean(abs(cD5))
print(meanD8)
print(meanD7)
print(meanD6)
print(meanD5)

ED8=sum([c*c for c in cD8])/len(cD8)
ED7=sum([c*c for c in cD7])/len(cD7)
ED6=sum([c*c for c in cD6])/len(cD6)
ED5=sum([c*c for c in cD5])/len(cD5)

print(ED8)
print(ED7)
print(ED6)
print(ED5)

Std8=np.std(cD8)
Std7=np.std(cD7)
Std6=np.std(cD8)
Std5=np.std(cD8)

print(Std8)
print(Std7)
print(Std6)
print(Std5)

ratio1=meanD8/meanD7
ratio2=meanD7/meanD6
ratio3=meanD6/meanD5



plt.figure(figsize=(60,13))
plt.subplot(511) 
plt.plot(EEG1[:,0],color="black",linewidth=0.3)   
plt.xlabel("time sec")
plt.ylabel("original")


plt.subplot(512)
plt.plot(cD8,color="black",linewidth=0.3)
plt.xlabel("time sec")
plt.ylabel("D8")

plt.subplot(513)
plt.plot(cD7,color="black",linewidth=0.3)
plt.xlabel("time sec")
plt.ylabel("D7")

plt.subplot(514)
plt.plot(cD6,color="black",linewidth=0.3)
plt.xlabel("time sec")
plt.ylabel("D6")

plt.subplot(515)
plt.plot(cD5,color="black",linewidth=0.3)
plt.xlabel("time sec")
plt.ylabel("D5")

plt.autoscale(enable=True,tight=True)
plt.show()

Source=np.array([meanD8,meanD7,meanD6,meanD5,ED8,ED7,ED6,ED5,Std8,Std7,Std6,Std5,ratio1,ratio2,ratio3])
ica = FastICA(n_components=5)
S_ = ica.fit_transform(Source)



"""
sampling_rate = 400
fft_size = 240000
t = np.arange(0,600, 1.0/sampling_rate)
x = EEG[:,0]
xs = x[:fft_size]
xf = np.fft.rfft(xs)/fft_size
freqs = np.linspace(0, sampling_rate/2, fft_size/2+1)

plt.figure(figsize=(30,4))
plt.subplot(211)
plt.plot(t[:fft_size], xs,color="black",linewidth=0.3)
plt.subplot(212)
plt.plot(freqs, xf,color="black",linewidth=0.3)
plt.xlabel(u"频率(Hz)")
plt.subplots_adjust(hspace=0.4)
plt.show()


plt.figure(figsize=(30,51))
nyq=sampling_rate*0.5
b,a=signal.iirdesign(0.27, 0.412, 0.1, 60.0, ftype='cheby1')
for i in range(0,16):
    x = EEG[:,i]
    xs = x[:fft_size]
    sf = signal.filtfilt(b, a, xs)
    xf1 = np.fft.rfft(sf)/fft_size
    ax1=plt.subplot(16,1,i+1)
    plt.plot(freqs[0:40000], xf1[0:40000],color="black",linewidth=0.3)
    plt.sca(ax1) 
    plt.xlabel("frequency")
    plt.ylabel("EEG Amplitude")
    plt.autoscale(enable=True,tight=True)

plt.show()


#For preictal data
train111='E:/EEG/train_1/1_1_1.mat'  
data2=sio.loadmat(train111) 
oac_struct=data2['dataStruct']
oac_struct.shape
val1=oac_struct[0,0]
EEG1=val1['data']
plt.figure(figsize=(30,51))
b,a=signal.iirdesign(0.27, 0.412, 0.1, 60.0, ftype='cheby1')
for i in range(0,16):
    x = EEG1[:,i]
    xs = x[:fft_size]
    sf = signal.filtfilt(b, a, xs)
    xf1 = np.fft.rfft(sf)/fft_size
    ax1=plt.subplot(16,1,i+1)
    plt.plot(freqs[0:40000], xf1[0:40000],color="black",linewidth=0.3)
    plt.sca(ax1) 
    plt.xlabel("frequency")
    plt.ylabel("EEG Amplitude")
    plt.autoscale(enable=True,tight=True)

plt.show()
"""
"""
t=np.arange(0,600,0.0025)
plt.figure(figsize=(30,51))
plt.scatter(t,EEG[:,0],s=0.1)
for i in range(0,16):
    ax1=plt.subplot(16,1,i+1)
    plt.plot(t,EEG[:,i],color="black",linewidth=0.3)
    plt.sca(ax1) 
    plt.xlabel("Time")
    plt.ylabel("EEG Amplitude")
    plt.autoscale(enable=True,tight=True)
plt.show()

#For preictal data
train111='E:/EEG/train_1/1_1_1.mat'  
data2=sio.loadmat(train111) 
oac_struct=data2['dataStruct']
oac_struct.shape
val1=oac_struct[0,0]
EEG1=val1['data']
print (EEG1.shape)

plt.figure(figsize=(30,51))
#plt.scatter(t,EEG[:,0],s=0.1)
for i in range(0,16):
    ax2=plt.subplot(16,1,i+1)
    plt.plot(t,EEG1[:,i],color="black",linewidth=0.3)
    plt.sca(ax2) 
    plt.xlabel("Time")
    plt.ylabel("EEG Amplitude")
    plt.autoscale(enable=True,tight=True)

plt.show()
"""

    
"""N,wn=signal.buttord(wp,ws,3,16)
b,a=signal.butter(N,wn,btype='low')
sf=signal.lfilter(b,a,xs)"""









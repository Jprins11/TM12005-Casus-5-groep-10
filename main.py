import scipy.io
from scipy.signal import butter, filtfilt, lfilter, find_peaks
from scipy.ndimage import uniform_filter1d
import numpy as np
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt


# Pan Tompkins 
def bandpass_filter(ecg, fs, lowcut=5, highcut=15, order=2):
    # Design bandpass Butterworth filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter
    filtered_ecg = filtfilt(b, a, ecg)
    
    return filtered_ecg

def derivate_filter(ecg, time): 
    a = 8 * time[1]
    b = np.array([-1, -2, 0, 2, 1])
    return lfilter(b,[a],ecg)

def square(ecg):
    return np.square(ecg)

def moving_mean(ecg,window_size):
    return uniform_filter1d(ecg, size=window_size, mode='nearest')

def find_tops(ecg,fs):
    peaks, _ = find_peaks(ecg, height=3,distance=0.3*fs)
    return peaks

def calculate_heartrate(peaks,time):
    return 60/np.mean(np.diff(time[peaks]))

def RR_interval(peaks,time):
    RR_interval = np.diff(time[peaks])
    diff_RRinterval = np.diff(RR_interval)

    return diff_RRinterval


if __name__ == '__main__':
    mat = scipy.io.loadmat("006_Roodhart_PVCs_1.mat")
    fs = 200
    ecg = mat["ecg"]["sig"][0,0][:,1]/1000
    ecg_time_adjust = ecg[0:20000]
    time = np.array(range(0,20000))/fs
    window_size = 20

    bandpass_ecg = bandpass_filter(ecg_time_adjust, fs)
    derivative_ecg = derivate_filter(bandpass_ecg,time)
    square_ecg = square(derivative_ecg)
    movmean_ecg = moving_mean(square_ecg, window_size)
    peaks_ecg = find_tops(movmean_ecg, fs)
    diff_RRinterval = RR_interval(peaks_ecg, time)
    print(round(calculate_heartrate(peaks_ecg,time),2))
    plt.plot(time,movmean_ecg)
    plt.plot(time[peaks_ecg],movmean_ecg[peaks_ecg], "ro", label='Peaks')
    plt.legend()
    # plt.show()
    # plt.figure()
    plt.plot(diff_RRinterval)
    plt.show()

from scipy.signal import butter, filtfilt, lfilter, find_peaks
from scipy.ndimage import uniform_filter1d
import numpy as np
import matplotlib.pyplot as plt

class PanTompkins:
    def __init__(self, ecg, fs):
        self.ecg = ecg
        self.fs = fs

    def __bandpass_filter(self, lowcut=5, highcut=15, order=2):
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, self.ecg)

    def __derivative_filter(self, signal):
        b = np.array([-1, -2, 0, 2, 1])
        return lfilter(b, [8 / self.fs], signal)

    def __square(self, signal):
        return np.square(signal)

    def __moving_mean(self, signal, window_size=150):
        return uniform_filter1d(signal, size=window_size, mode='nearest')

    def __find_peaks(self, signal):
        peaks, _ = find_peaks(signal, height=np.max(signal) * 0.3, distance=0.3 * self.fs)
        peaks = np.array(peaks) - 1 #TODO we moet hier iets doen met DELAY
        peaks = peaks[self.ecg[peaks] >=-0.01]
        return peaks

    def __calculate_heartrate(self, peaks):
        if len(peaks) < 2:
            return 0
        rr_intervals = np.diff(peaks) / self.fs  # Convert indices to time (seconds)
        return 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0

    def __rr_interval(self, peaks):
        if len(peaks) < 2:
            return np.array([])
        return np.diff(peaks) / self.fs

    def __calculate_regularities(self, rr_intervals):
        if len(rr_intervals) == 0:
            return {'mean_rr': 0, 'std_rr': 0, 'cv_rr': 0}
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        cv_rr = std_rr / mean_rr if mean_rr != 0 else 0  # Coefficient of Variation

        return {
            'mean_rr': mean_rr,
            'std_rr': std_rr,
            'cv_rr': cv_rr
        }

    def run(self):
        filtered = self.__bandpass_filter()
        derivative = self.__derivative_filter(filtered)
        squared = self.__square(derivative)
        smoothed = self.__moving_mean(squared, window_size=int(0.15 * self.fs))
        peaks = self.__find_peaks(smoothed)
        heart_rate = self.__calculate_heartrate(peaks)
        rr_intervals = self.__rr_interval(peaks)
        regularities = self.__calculate_regularities(rr_intervals)

        return {
            'filtered_ecg': filtered,
            'derivative': derivative,
            'squared': squared,
            'smoothed': smoothed,
            'R_peaks': peaks,
            'heart_rate': heart_rate,
            'rr_intervals': rr_intervals,
            'regularities': regularities
        }


def find_Q_S_peaks(ecg, fs, r_peaks):
    q_peaks = []
    s_peaks = []

    for r in r_peaks:
        # om zeker te zijn dat we niet voor t=0 aan het zoeken zijn
        q_search_start = max(0, r - int(0.1 * fs))
        q_point = np.argmin(ecg[q_search_start:r]) + q_search_start
        q_peaks.append(q_point)

        #om zeker te zijn dat we niet na de lengte van het ECG aan het kijken zijn
        s_search_end = min(len(ecg) - 1, r + int(0.1 * fs))
        s_point = r + np.argmin(ecg[r:s_search_end])
        s_peaks.append(s_point)

    return np.array(q_peaks), np.array(s_peaks)

def calculate_QRS_time(q_peaks, s_peaks, fs):
    if len(q_peaks) != len(s_peaks):
        raise ValueError("Mismatch: Q-peaks and S-peaks must have the same length.")

    qrs_durations = (np.array(s_peaks) - np.array(q_peaks)) / fs * 1000
    return qrs_durations

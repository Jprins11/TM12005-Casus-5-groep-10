from scipy.signal import butter, filtfilt, lfilter, find_peaks
from scipy.ndimage import uniform_filter1d
import numpy as np

class PanTompkins:
    def __init__(self, ecg, fs):
        self.ecg = ecg
        self.fs = fs

    def __bandpass_filter(self, lowcut=5, highcut=15, order=2):
        nyquist = 0.5 * self.fs
        low, high = lowcut / nyquist, highcut / nyquist
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
        peaks = np.array(peaks) - 2  # Adjust for delay
        return peaks

    def __calculate_heartrate(self, peaks):
        if len(peaks) < 2:
            return 0
        rr_intervals = np.diff(peaks) / self.fs
        return 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0

    def __calculate_regularities(self, rr_intervals):
        if len(rr_intervals) == 0:
            return {'mean_rr': 0, 'std_rr': 0, 'cv_rr': 0}
        mean_rr, std_rr = np.mean(rr_intervals), np.std(rr_intervals)
        cv_rr = std_rr / mean_rr if mean_rr != 0 else 0
        return {'mean_rr': mean_rr, 'std_rr': std_rr, 'cv_rr': cv_rr}

    def run(self):
        filtered = self.__bandpass_filter()
        derivative = self.__derivative_filter(filtered)
        squared = self.__square(derivative)
        smoothed = self.__moving_mean(squared, window_size=int(0.15 * self.fs))
        peaks = self.__find_peaks(smoothed)
        return {
            'filtered_ecg': filtered,
            'derivative': derivative,
            'squared': squared,
            'smoothed': smoothed,
            'R_peaks': peaks,
            'heart_rate': self.__calculate_heartrate(peaks),
            'rr_intervals': np.diff(peaks) / self.fs if len(peaks) > 1 else np.array([]),
            'regularities': self.__calculate_regularities(np.diff(peaks) / self.fs if len(peaks) > 1 else np.array([]))
        }
    
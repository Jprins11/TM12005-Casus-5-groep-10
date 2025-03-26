from scipy.signal import butter, filtfilt, lfilter, find_peaks
from scipy.ndimage import uniform_filter1d
import numpy as np
from scipy.interpolate import interp1d

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
        return peaks[self.ecg[peaks] >= -0.01]

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

# Q and S Peak Detection
def find_Q_S_peaks(ecg, fs, r_peaks):
    q_peaks, s_peaks = [], []
    for r in r_peaks:
        q_peaks.append(max(0, r - int(0.1 * fs)) + np.argmin(ecg[max(0, r - int(0.1 * fs)):r]))
        s_peaks.append(r + np.argmin(ecg[r:min(len(ecg) - 1, r + int(0.1 * fs))]))
    return np.array(q_peaks), np.array(s_peaks)

# Calculate QRS Duration
def calculate_QRS_time(q_peaks, s_peaks, fs):
    if len(q_peaks) != len(s_peaks):
        raise ValueError("Mismatch: Q-peaks and S-peaks must have the same length.")
    return (np.array(s_peaks) - np.array(q_peaks)) / fs * 1000

# High-pass Filter
def high_pas_filter(ecg, fs, fc, order=2):
    nyquist = 0.5 * fs
    b, a = butter(order, fc / nyquist, btype='high')
    return filtfilt(b, a, ecg)

# Running FFT Analysis
def running_fft(ecg, fs, seglen):
    num_seg = len(ecg) // seglen
    intensities, t = [], []
    for i in range(num_seg):
        seg = ecg[i * seglen:(i + 1) * seglen]
        intensities.append(np.sum(np.abs(np.fft.fft(seg))))
        t.append(i * seglen / fs)
    return t, intensities

# Remove QRS Complexes
def remove_Q_R(ecg, q_peaks, s_peaks):
    mask = np.ones(len(ecg), dtype=bool)
    for q, s in zip(q_peaks, s_peaks):
        mask[q:s+1] = False
    return ecg[mask]

def moving_mean(signal, window_size=150):
    return uniform_filter1d(signal, size=window_size, mode='nearest')


def detect_minima_above_threshold(min_indices, min_values, signal_length, n, m):
    # Step 1: Interpolate to get a smooth lower envelope
    interp_func = interp1d(min_indices, min_values, kind="cubic", fill_value="extrapolate")
    lower_envelope = interp_func(np.arange(signal_length))

    # Step 2: Find where the lower envelope is above the threshold `n`
    above_threshold = lower_envelope > n

    # Step 3: Check for `m` consecutive True values
    count = 0
    for val in above_threshold:
        if val:
            count += 1
            if count >= m:
                result = True  # Found m consecutive values above the threshold
                break
        else:
            count = 0  # Reset counter if a value is below the threshold
    else:
        result = False  # No valid sequence found

    return result
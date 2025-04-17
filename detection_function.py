import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

def replace_pvc_R_peak(ecg,peaks):
    replaced_peaks = []
    non_pvc_peaks = peaks[ecg[peaks] > 0]
    for index, peak in enumerate(peaks):
        if ecg[peak] < 0:
            replaced_peaks.append(int(np.median(np.array([peaks[index-1], peaks[index+1]]))))
        else:
            replaced_peaks.append(int(peak))
    return replaced_peaks,non_pvc_peaks


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
def running_fft(ecg, fs, window_size=128):
    """
    Performs a running FFT over a sliding window of fixed size (e.g., 100 samples),
    computing the total intensity (sum of absolute FFT values) for each point
    after the initial window.

    Parameters:
        ecg (array): The ECG signal.
        fs (float): Sampling frequency in Hz.
        window_size (int): Size of the sliding FFT window.

    Returns:
        t (list): Time points corresponding to each FFT calculation.
        intensities (list): Total intensity from FFT at each time point.
    """
    ecg = np.array(ecg, dtype=float)
    intensities, t = [], []

    for i in range(window_size, len(ecg)):
        window = ecg[i - window_size:i]
        fft_result = np.fft.fft(window)
        intensity = np.sum(np.abs(fft_result))
        intensities.append(intensity)
        t.append(i / fs)

    return t, intensities

def calculate_heartrate(fs, peaks):
        if len(peaks) < 2:
            return 0
        rr_intervals = np.diff(peaks) / fs
        return 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0

# Remove QRS Complexes
def remove_Q_R(ecg, q_peaks, s_peaks):
    mask = np.ones(len(ecg), dtype=bool)
    for q, s in zip(q_peaks, s_peaks):
        mask[q:s+1] = False
    return ecg[mask]

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


def replace_outliers(ecg_signal):
    """
    Replaces ECG values that are outside the IQR bounds (Q1 - 1.5*IQR or Q3 + 1.5*IQR)
    with the mean of the previous 100 values.

    Parameters:
        ecg_signal (list or np.ndarray): The ECG signal values.

    Returns:
        np.ndarray: The corrected ECG signal.
    """
    ecg_signal = np.array(ecg_signal, dtype=float)

    # Calculate IQR bounds
    q1 = np.percentile(ecg_signal, 25)
    q3 = np.percentile(ecg_signal, 75)
    iqr = q3 - q1
    lower_bound = q1 - 10 * iqr
    upper_bound = q3 + 10 * iqr

    corrected_signal = ecg_signal.copy()
    for i in range(len(ecg_signal)):
        if ecg_signal[i] < lower_bound or ecg_signal[i] > upper_bound:
            start_idx = max(0, i - 100)
            past_values = corrected_signal[start_idx:i]
            if len(past_values) > 0:
                corrected_signal[i] = np.mean(past_values)
            else:
                corrected_signal[i] = (q1 + q3) / 2  # fallback to median
    
    return corrected_signal

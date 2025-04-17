import numpy as np
import scipy.signal
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class AF_detection:
    def __init__(self, ecg_signal, SAMPLE_F,time):
        self.signal = np.array(ecg_signal, dtype=float)
        self.fs = SAMPLE_F
        self.original_signal = self.signal.copy()
        self.time = time

    def high_pass_filter(self, fc, order=2):
        nyquist = 0.5 * self.fs
        b, a = butter(order, fc / nyquist, btype='high')
        self.signal = filtfilt(b, a, self.signal)
        return self.signal

    def running_fft(self, window_size=128):
        intensities, t = [], []

        for i in range(window_size, len(self.signal)):
            window = self.signal[i - window_size:i]
            fft_result = np.fft.fft(window)
            intensity = np.sum(np.abs(fft_result))
            intensities.append(intensity)
            t.append(i / self.fs)

        return t, np.array(intensities)

    def replace_outliers(self):
        q1 = np.percentile(self.signal, 25)
        q3 = np.percentile(self.signal, 75)
        iqr = q3 - q1
        lower_bound = q1 - 10 * iqr
        upper_bound = q3 + 10 * iqr

        corrected_signal = self.signal.copy()
        for i in range(len(self.signal)):
            if self.signal[i] < lower_bound or self.signal[i] > upper_bound:
                start_idx = max(0, i - 100)
                past_values = corrected_signal[start_idx:i]
                if len(past_values) > 0:
                    corrected_signal[i] = np.mean(past_values)
                else:
                    corrected_signal[i] = (q1 + q3) / 2
        self.signal = corrected_signal
        return self.signal

    def detect_minima_above_threshold(self, min_indices, min_values, n, m):
        signal_length = len(self.signal)
        interp_func = interp1d(min_indices, min_values, kind="cubic", fill_value="extrapolate")
        lower_envelope = interp_func(np.arange(signal_length))

        above_threshold = lower_envelope > n
        count = 0
        for val in above_threshold:
            if val:
                count += 1
                if count >= m:
                    return True
            else:
                count = 0
        return False
    
    def plot_fft_intensity(self, title="Running FFT Intensity Over Time"):

        plt.figure(figsize=(10, 4))
        plt.plot(self.t, self.fft_intensity, color='purple', linewidth=1.5)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("FFT Intensity (sum of abs values)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_merged_segments(self,begin_an):
        # === Plot Lower Envelope Analysis ===
        plt.figure(figsize=(12, 6))
        plt.scatter(self.min_indices, self.min_values, color="red", label="Local Minima")
        plt.plot(self.lower_envelope, color="black", linestyle="--", linewidth=2, label="Lower Envelope")
        plt.axhline(y=115, color="green", linestyle=":", label="Cutoff Threshold")

        # Mark merged segment start and end points
        for start, end in self.segments:
            plt.scatter([start, end], [self.lower_envelope[start], self.lower_envelope[end]], 
                        color="blue", s=100, edgecolors="black", label="Segment Start/End")
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()

        # Convert FFT indices to time values in the ECG signal
        fft_time = np.linspace(self.t[0], self.t[-1], len(self.lower_envelope))

        # === Plot ECG Signal with Corrected High-Energy Markers ===
        plt.figure(figsize=(12, 5))
        plt.plot(self.time, self.original_signal, label="ECG", color="black", alpha=0.7)

        start_plotted = end_plotted = False

        for start, end in self.segments:
            start_time = fft_time[start] + begin_an/self.fs
            end_time = fft_time[end] + begin_an/self.fs

            # start_amp = ecg_signal[np.argmin(np.abs(time - start_time))]
            start_amp = np.zeros(start_time.size)
            # end_amp = ecg_signal[np.argmin(np.abs(time - end_time))]
            end_amp = np.zeros(start_time.size)

            # Add labels only once
            if not start_plotted:
                plt.scatter(start_time, start_amp, color="green", s=100, edgecolors="black", label="AF Start")
                start_plotted = True
            else:
                plt.scatter(start_time, start_amp, color="green", s=100, edgecolors="black")

            if not end_plotted:
                plt.scatter(end_time, end_amp, color="red", s=100, edgecolors="black", label="AF End")
                end_plotted = True
            else:
                plt.scatter(end_time, end_amp, color="red", s=100, edgecolors="black")

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()
        plt.show()

    def find_high_energy_segments(self, lower_envelope, threshold=115, min_length=600, merge_dist=10000):
        above_threshold = np.where(lower_envelope > threshold)[0]
        segments = []

        if len(above_threshold) > 0:
            start_idx = above_threshold[0]
            for i in range(1, len(above_threshold)):
                if above_threshold[i] != above_threshold[i - 1] + 1:
                    end_idx = above_threshold[i - 1]
                    if (end_idx - start_idx + 1) >= min_length:
                        segments.append((start_idx, end_idx))
                    start_idx = above_threshold[i]

            end_idx = above_threshold[-1]
            if (end_idx - start_idx + 1) >= min_length:
                segments.append((start_idx, end_idx))

        merged_segments = []
        if segments:
            merged_start, merged_end = segments[0]
            for i in range(1, len(segments)):
                start, end = segments[i]
                if start - merged_end <= merge_dist:
                    merged_end = end
                else:
                    merged_segments.append((merged_start, merged_end))
                    merged_start, merged_end = start, end
            merged_segments.append((merged_start, merged_end))

        return np.array(merged_segments)

    def run_analysis(self, window_size=400, hp_cutoff=4, find_segments_func=None):
        # Step 1: Remove outliers
        self.replace_outliers()

        # Step 2: Save original
        self.original_signal = self.signal.copy()

        # Step 3: High-pass filter
        self.high_pass_filter(fc=hp_cutoff)

        # Step 4: Running FFT
        self.t, self.fft_intensity = self.running_fft(window_size=window_size)

        # Step 5: Local minima of FFT
        self.min_indices = scipy.signal.argrelextrema(self.fft_intensity, np.less, order=5)[0]
        self.min_values = self.fft_intensity[self.min_indices]

        # Step 6: Interpolate lower envelope
        interp_func = interp1d(self.min_indices, self.min_values, kind="cubic", fill_value="extrapolate")
        self.lower_envelope = interp_func(np.arange(len(self.fft_intensity)))

        # Step 7: Use external function to find segments
        self.segments = self.find_high_energy_segments(self.lower_envelope)

        return {
            "fft_time": self.t,
            "fft_intensity": self.fft_intensity,
            "min_indices": self.min_indices,
            "min_values": self.min_values,
            "lower_envelope": self.lower_envelope,
            "segments": self.segments
        }

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

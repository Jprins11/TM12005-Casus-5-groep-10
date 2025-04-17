import numpy as np
import matplotlib.pyplot as plt

def analyze_fft(ecg_signal: np.ndarray, time: np.ndarray, sample_freq: int, min_freq: float = 4.0, plot: bool = True) -> float:
    """
    Perform FFT analysis on an ECG signal and calculate area under the curve above a specified frequency.
    
    Parameters:
        ecg_signal (np.ndarray): The ECG signal.
        time (np.ndarray): Corresponding time vector.
        sample_freq (int): Sampling frequency in Hz.
        min_freq (float): Minimum frequency threshold for integration.
        plot (bool): Whether to plot the FFT and signal.
    """

    # Perform FFT
    fft_result = np.fft.fft(ecg_signal)
    fft_freq = np.fft.fftfreq(len(ecg_signal), d=1/sample_freq)

    # Keep only positive frequencies
    half_n = len(ecg_signal) // 2
    fft_result = np.abs(fft_result[:half_n])
    fft_freq = fft_freq[:half_n]

    # Filter frequencies above threshold
    mask = fft_freq > min_freq
    fft_freq_filtered = fft_freq[mask]
    fft_result_filtered = fft_result[mask]

    # Calculate area
    area_above_threshold = np.trapz(fft_result_filtered, fft_freq_filtered)

    if plot:
        # FFT Plot
        plt.figure(figsize=(10, 5))
        plt.plot(fft_freq, fft_result, label='Full Spectrum')
        plt.fill_between(fft_freq_filtered, fft_result_filtered, alpha=0.4, label=f'>{min_freq} Hz Area')
        plt.title("Fourier Spectrum of ECG Signal")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Time domain plot
        plt.figure(figsize=(10, 4))
        plt.plot(time, ecg_signal, label="ECG Signal")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print(f"Area under the curve for frequencies > {min_freq} Hz: {area_above_threshold:.2f}")


def plot_smoothed_ecg(smoothed_signal: np.ndarray):
    """
    Plot the smoothed ECG signal.

    Parameters:
        smoothed_signal (np.ndarray): The smoothed ECG signal from Pan-Tompkins.
        sample_freq (int, optional): Sampling frequency to convert x-axis to seconds. If None, index is used.
    """
    x_axis = np.arange(len(smoothed_signal))
    x_label = 'Sample Index'

    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, smoothed_signal, label='Smoothed ECG', color='blue')
    plt.title('Smoothed ECG Signal')
    plt.xlabel(x_label)
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ecg_peaks(time, ecg_signal, q_peaks, s_peaks, r_peaks_all, r_peaks_clean, start_index=0, sample_freq=200):
    """
    Plot ECG signal with Q, S, and R peaks.

    Parameters:
        time (np.ndarray): Time vector corresponding to ECG signal.
        ecg_signal (np.ndarray): Raw ECG signal.
        q_peaks (np.ndarray): Indices of Q-peaks (after PVC replacement).
        s_peaks (np.ndarray): Indices of S-peaks (after PVC replacement).
        r_peaks_all (np.ndarray): R-peak indices including PVCs.
        r_peaks_clean (np.ndarray): R-peak indices after PVC replacement.
        start_index (int): Offset applied to peak indices (default: 0).
        sample_freq (int): Sampling frequency to convert indices to time (default: 200).
    """

    # Q and S Peaks Plot
    plt.figure()
    plt.plot(time, ecg_signal, label='ECG-signal')
    plt.scatter((q_peaks + start_index) / sample_freq, ecg_signal[q_peaks], color='red', label='Q-peaks')
    plt.scatter((s_peaks + start_index) / sample_freq, ecg_signal[s_peaks], color='green', label='S-peaks')
    plt.xlabel("Time [s]")
    plt.ylabel("ECG Intensity")
    plt.title("ECG with Q and S Peaks")
    plt.legend()
    plt.grid(True)

    # R Peaks Plot (PVC vs Cleaned)
    plt.figure()
    plt.plot(time, ecg_signal, label='ECG-signal')
    plt.scatter((r_peaks_all + start_index) / sample_freq, ecg_signal[r_peaks_all - 1], color='red', label='Detected PVCs (R-peaks)')
    plt.scatter((r_peaks_clean + start_index) / sample_freq, ecg_signal[r_peaks_clean - 1], color='green', label='Clean R-peaks')
    plt.xlabel("Time [s]")
    plt.ylabel("ECG Intensity")
    plt.title("ECG with R-Peak Detection")
    plt.legend()
    plt.grid(True)
    plt.show()
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pan_tompkins import PanTompkins

from detection_function import (
    find_Q_S_peaks,
    calculate_QRS_time,
    high_pas_filter,
    remove_Q_R,
    running_fft,
    replace_pvc_R_peak,
    calculate_heartrate,
    replace_outliers
)


# Load ECG Data
def load_ecg_data(filepath, sample_freq,begin,end):
    """
    Load ECG data from a .mat file and return the processed signal.
    
    Parameters:
        filepath (str): Path to the .mat file.
        sample_freq (int): Sampling frequency of the ECG signal.
    
    Returns:
        np.ndarray: Processed ECG signal.
        np.ndarray: Corresponding time vector.
    """
    mat = scipy.io.loadmat(filepath)
    ecg_signal = mat["ecg"]["sig"][0, 0][:, 1] / 1000  # Convert to mV
    return ecg_signal[begin:end], np.arange(begin, end) / sample_freq

# Find Continuous Segments Above Threshold
def find_high_energy_segments(lower_envelope, threshold=115, min_length=600, merge_dist=10000):
    """
    Identify continuous high-energy segments in the lower envelope of the FFT.

    Parameters: 
        lower_envelope (np.ndarray): Computed lower envelope of the signal.
        threshold (float): Cutoff value to consider a segment as high-energy.
        min_length (int): Minimum length of a segment to be valid.
        merge_dist (int): Maximum gap between segments to merge them.

    Returns:
        list of tuples: Start and end indices of the merged high-energy segments.
    """
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

        # Add the last segment
        end_idx = above_threshold[-1]
        if (end_idx - start_idx + 1) >= min_length:
            segments.append((start_idx, end_idx))

    # Merge segments that are close to each other
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


# Main execution
if __name__ == "__main__":
    SAMPLE_F = 200
    FILE_PATH = "006_Roodhart_PVCs_1.mat"
    start_analysis = 0
    end_analysis = 120000
    # Load ECG Data
    ecg_signal, time = load_ecg_data(FILE_PATH, SAMPLE_F,start_analysis,end_analysis)

        # Perform FFT
    fft_result = np.fft.fft(ecg_signal)
    fft_freq = np.fft.fftfreq(len(ecg_signal), d=1/SAMPLE_F)

    # Only keep the positive half of the spectrum
    half_n = len(ecg_signal) // 2
    fft_result = np.abs(fft_result[:half_n])
    fft_freq = fft_freq[:half_n]

    # Filter to only frequencies above 4 Hz
    mask = fft_freq > 4
    fft_freq_filtered = fft_freq[mask]
    fft_result_filtered = fft_result[mask]

    # Filter to only frequencies above 4 Hz
    mask = fft_freq > 4
    fft_freq_filtered = fft_freq[mask]
    fft_result_filtered = fft_result[mask]

    # Integrate using the trapezoidal rule
    area_above_4Hz = np.trapz(fft_result_filtered, fft_freq_filtered)

    # Plot for visualization
    plt.figure(figsize=(10, 5))
    plt.plot(fft_freq, fft_result, label='Full Spectrum')
    plt.fill_between(fft_freq_filtered, fft_result_filtered, alpha=0.4, label='> 4 Hz Area')
    plt.title("Fourier Spectrum of ECG Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Area under the curve for frequencies > 4 Hz: {area_above_4Hz:.2f}")

    plt.figure()
    plt.plot(time,ecg_signal,label="ecg_signal")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel('Intensity')
    plt.show()

    # Run Pan-Tompkins Algorithm
    pan_tompkins = PanTompkins(ecg_signal, SAMPLE_F)

    results = pan_tompkins.run()
    peaks_replaced, peaks_no_pvc = replace_pvc_R_peak(ecg_signal,results['R_peaks'])
    print(np.mean(np.diff(peaks_replaced)))


    # Find Q and S Peaks
    q_peaks_pvc, s_peaks_pvc = find_Q_S_peaks(ecg_signal, SAMPLE_F,results['R_peaks'])
    q_peaks, s_peaks = find_Q_S_peaks(ecg_signal, SAMPLE_F, peaks_no_pvc)

    print(f'Analysis of {FILE_PATH} from index {start_analysis} to {end_analysis}: ')
    print("\nStatistic of ECG without removal of the PVC")
    print(f"The heartrate is {np.round(results['heart_rate'],2)} BPM")
    print(f'The mean RR-interval is {np.round(np.mean(results['rr_intervals']),2)*1000} ms')
    print(f"The standard deviation of the RR-interval is {np.round(results['regularities']['std_rr'],3)}")
    print(f"The coefficient of variation of the RR-interval is {np.round(results['regularities']['cv_rr'],3)}")
    print(f"The mean QRS-time is {np.mean(s_peaks_pvc-q_peaks_pvc)/200*1000} ms")

    print("\nStatistic of ECG with removal of the PVC (negative peaks) but replaced with median ")
    print(f"The heartrate is {round(calculate_heartrate(SAMPLE_F,peaks_replaced),2)} BPM")
    print(f'The mean RR-interval is {round(np.mean(np.diff(np.array(peaks_replaced)/200)),2)*1000} ms')
    print(f"The standard deviation of the RR-interval is {np.round(results['regularities']['std_rr'],3)}")
    print(f"The coefficient of variation of the RR-interval is {np.round(results['regularities']['cv_rr'],3)}")
    print(f"The mean QRS-time is {round(np.mean(s_peaks-q_peaks)/200*1000,2)} ms")

    plt.figure()
    plt.ylabel("ECG_intensity")
    plt.xlabel("time [s]")
    plt.scatter((q_peaks+start_analysis)/200,(ecg_signal[q_peaks]),color='red',label='q-toppen')
    plt.scatter((s_peaks+start_analysis)/200,(ecg_signal[s_peaks]),color='green',label='s-toppen')
    plt.plot(time,ecg_signal,label='ECG-signal')
    plt.legend()

    plt.figure()
    plt.ylabel("ECG_intensity")
    plt.xlabel("time [s]")
    plt.scatter((results['R_peaks']+start_analysis)/200,ecg_signal[results["R_peaks"]-1],color='red',label='gedetecteerde PVC')
    plt.scatter((peaks_no_pvc+start_analysis)/200,ecg_signal[peaks_no_pvc-1],color='green',label='R-toppen')
    plt.plot(time,ecg_signal,label='ECG-signal')
    plt.legend()
    plt.show()
    ecg_signal = replace_outliers(ecg_signal)
    ecg_signal_original = ecg_signal[:]
    
    
    # High-pass Filter
    ecg_signal = high_pas_filter(ecg_signal, SAMPLE_F, 4)

    # Compute Running FFT
    t, fft_intensity = running_fft(ecg_signal, SAMPLE_F, 400)
    fft_intensity = np.array(fft_intensity)


    # fft_intensity = (fft_intensity - np.mean(fft_intensity))/np.std(fft_intensity)

    plt.figure()
    plt.plot(t,fft_intensity,label='sliding fft power of ECG')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequentie inhoud')
    plt.legend()
    plt.show()

    
    # Find Local Minima of FFT
    min_indices = scipy.signal.argrelextrema(fft_intensity, np.less, order=5)[0]
    min_values = fft_intensity[min_indices]

    # Interpolate Lower Envelope
    interp_func = interp1d(min_indices, min_values, kind="cubic", fill_value="extrapolate")
    lower_envelope = interp_func(np.arange(len(fft_intensity)))

    # Identify High Energy Segments
    merged_segments = find_high_energy_segments(lower_envelope)

    # === Plot Lower Envelope Analysis ===
    plt.figure(figsize=(12, 6))
    plt.scatter(min_indices, min_values, color="red", label="Local Minima")
    plt.plot(lower_envelope, color="black", linestyle="--", linewidth=2, label="Lower Envelope")
    plt.axhline(y=115, color="green", linestyle=":", label="Cutoff Threshold")

    # Mark merged segment start and end points
    for start, end in merged_segments:
        plt.scatter([start, end], [lower_envelope[start], lower_envelope[end]], 
                    color="blue", s=100, edgecolors="black", label="Segment Start/End")
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    # Convert FFT indices to time values in the ECG signal
    fft_time = np.linspace(time[0], time[-1], len(lower_envelope))

    # === Plot ECG Signal with Corrected High-Energy Markers ===
    plt.figure(figsize=(12, 5))
    plt.plot(time, ecg_signal_original, label="ECG", color="black", alpha=0.7)

    start_plotted = end_plotted = False

    for start, end in merged_segments:
        start_time = fft_time[start]
        end_time = fft_time[end]

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

    


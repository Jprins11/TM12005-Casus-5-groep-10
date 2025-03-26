import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from pan_tompkins import PanTompkins, find_Q_S_peaks, calculate_QRS_time, high_pas_filter, remove_Q_R, running_fft, detect_minima_above_threshold
import pywt  # Correct package for wavelet transforms
import pywt.data
import numpy as np
import scipy.signal
from scipy.interpolate import interp1d


# Load ECG Data
def load_ecg_data(filepath, sample_freq):
    mat = scipy.io.loadmat(filepath)
    ecg_signal = mat["ecg"]["sig"][0, 0][:, 1] / 1000  # Convert to mV
    return ecg_signal[:], np.arange(0, len(ecg_signal)) / sample_freq

# Main function
if __name__ == '__main__':
    SAMPLE_F = 200
    ecg_signal, time = load_ecg_data("006_Roodhart_PVCs_1.mat", SAMPLE_F)

    # Run Pan-Tompkins algorithm
    pan_tompkins = PanTompkins(ecg_signal, SAMPLE_F)
    results = pan_tompkins.run()

    # Find Q and S peaks
    q_peaks, s_peaks = find_Q_S_peaks(ecg_signal, SAMPLE_F, results['R_peaks'])
    removed_ecg = remove_Q_R(ecg_signal, q_peaks, s_peaks)

    # Calculate QRS time
    QRS_time = calculate_QRS_time(q_peaks, s_peaks, SAMPLE_F)
    print(f"QRS Duration: {QRS_time} s")
    ecg_signal = high_pas_filter(ecg_signal,200,4)
    # Calculate Running FFT
    t, fft_intensity = running_fft(ecg_signal,SAMPLE_F,16)
    fft_intensity = np.array(fft_intensity)
    min_indices = scipy.signal.argrelextrema(fft_intensity, np.less, order=5)[0]
    min_values = fft_intensity[min_indices]

    # Step 3: Interpolate between minima to get a smooth lower envelope
    interp_func = interp1d(min_indices, min_values, kind="cubic", fill_value="extrapolate")
    lower_envelope = interp_func(np.arange(len(fft_intensity)))

    result = np.where(lower_envelope > 0.2)[0]

    # Step 3: Identify continuous segments where the lower envelope is above `n`
    m = 10
    segments = []
    if len(result) > 0:
        start_idx = result[0]

        for i in range(1, len(result)):
            if result[i] != result[i - 1] + 1:
                # If not consecutive, close the previous segment
                end_idx = result[i - 1]
                if (end_idx - start_idx + 1) >= m:  # Ensure segment length is at least `m`
                    segments.append((start_idx, end_idx))
                start_idx = result[i]

        # Add the last segment if it's long enough
        end_idx = result[-1]
        if (end_idx - start_idx + 1) >= m:
            segments.append((start_idx, end_idx))
    
    N=100
    # Step 4: Merge segments that are within `N` samples of each other
    merged_segments = []
    if len(segments) > 0:
        merged_start, merged_end = segments[0]

        for i in range(1, len(segments)):
            start, end = segments[i]
            if start - merged_end <= N:
                # Merge this segment into the previous one
                merged_end = end
            else:
                # Save the previous merged segment and start a new one
                merged_segments.append((merged_start, merged_end))
                merged_start, merged_end = start, end

        # Add the final merged segment
        merged_segments.append((merged_start, merged_end))



    # Step 4: Plot the signal and lower envelope
    plt.figure(figsize=(12, 6))
    # plt.plot(fft_intensity, label="Original Signal", alpha=0.6)
    plt.scatter(min_indices, min_values, color="red", label="Local Minima")
    plt.plot(lower_envelope, color="black", linestyle="--", linewidth=2, label="Lower Envelope")
    plt.plot([0,len(lower_envelope)],[0.2,0.2],color='green',label='cut off')
    
    # Mark merged segment start and end points
    for start, end in merged_segments:
        plt.scatter([start, end], [lower_envelope[start], lower_envelope[end]], 
                    color="blue", s=100, edgecolors="black", label="Segment Start/End")

    plt.legend()
    plt.title("Signal with Lower Envelope")
    plt.show()

    # plt.figure()
    # plt.plot(min_indices,min_values)
    # plt.show()
    
    '''
    # Print Heart Rate and RR Interval Statistics
    print(f"Heart Rate: {round(results['heart_rate'], 2)} BPM")
    print(f"Mean RR Interval: {results['regularities']['mean_rr']:.4f} s")
    print(f"RR Interval Std Dev: {results['regularities']['std_rr']:.4f} s")
    print(f"Coefficient of Variation (CV): {results['regularities']['cv_rr']:.4f}")

    
    # Plot ECG with QRS Detection
    plt.figure(figsize=(12, 5))
    plt.plot(time, ecg_signal, label="ECG", color="black", alpha=0.7)
    plt.scatter(time[results['R_peaks']], ecg_signal[results['R_peaks']], color="red",
                 marker="o", label="R-peaks")
    plt.scatter(time[q_peaks], ecg_signal[q_peaks], color="blue", marker="x", label="Q-peaks")
    plt.scatter(time[s_peaks], ecg_signal[s_peaks], color="green", marker="x", label="S-peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.title("ECG Signal with Detected Q, R, S Peaks")
    plt.legend()
    plt.grid()

    # Plot ECG after QRS Removal
    plt.figure(figsize=(12, 4))
    plt.plot(time[:len(removed_ecg)], removed_ecg, color="purple")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.title("ECG Signal After QRS Removal")
    plt.grid()

    plt.show()
    '''
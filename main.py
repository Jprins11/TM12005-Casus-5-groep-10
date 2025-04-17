import scipy.io
import numpy as np
import example_plots
from pan_tompkins import PanTompkins
from detection_function import (
    AF_detection,
    find_Q_S_peaks,
    replace_pvc_R_peak
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


# Main execution
if __name__ == "__main__":
    SAMPLE_F = 200
    FILE_PATH = "006_Roodhart_PVCs_1.mat"
    start_analysis = 50000
    end_analysis = 80000
    # Load ECG Data
    ecg_signal, time = load_ecg_data(FILE_PATH, SAMPLE_F,start_analysis,end_analysis)

    # Run fft on peace of ECG to get example plot of AREA under curve (default > 4 Hz)
    example_plots.analyze_fft(ecg_signal,time,SAMPLE_F)

    # Run Pan-Tompkins Algorithm
    pan_tompkins = PanTompkins(ecg_signal, SAMPLE_F)
    results = pan_tompkins.run()

    peaks_replaced, peaks_no_pvc = replace_pvc_R_peak(ecg_signal,results['R_peaks'])
    print(np.mean(np.diff(peaks_replaced)))

    # Get a plot of the smoothed Pan Tompkins with all R-tops
    smoothed = results['smoothed']
    example_plots.plot_smoothed_ecg(smoothed)

    # Find Q and S Peaks
    q_peaks_pvc, s_peaks_pvc = find_Q_S_peaks(ecg_signal, SAMPLE_F,results['R_peaks'])
    q_peaks, s_peaks = find_Q_S_peaks(ecg_signal, SAMPLE_F, peaks_no_pvc)

    # Give information about the analysis
    print(f'Analysis of {FILE_PATH} from index {start_analysis} to {end_analysis}: ')
    print("\nStatistic of ECG without removal of the PVC")
    print(f"The heartrate is {np.round(results['heart_rate'],2)} BPM")
    print(f'The mean RR-interval is {np.round(np.mean(results['rr_intervals']),2)*1000} ms')
    print(f"The standard deviation of the RR-interval is {np.round(results['regularities']['std_rr'],3)}")
    print(f"The coefficient of variation of the RR-interval is {np.round(results['regularities']['cv_rr'],3)}")
    print(f"The mean QRS-time is {np.mean(s_peaks_pvc-q_peaks_pvc)/200*1000} ms")

    print("\nStatistic of ECG with removal of the PVC (negative peaks) but replaced with median ")
    print(f"The mean QRS-time is {round(np.mean(s_peaks-q_peaks)/200*1000,2)} ms")

    # Plot the peaks of the ECG with and without PVC.
    example_plots.plot_ecg_peaks(time,ecg_signal,q_peaks,s_peaks,results['R_peaks'],peaks_no_pvc,start_analysis, SAMPLE_F)

    algorithm = AF_detection(ecg_signal,SAMPLE_F,time)

    results_algoritme = algorithm.run_analysis()
    algorithm.plot_fft_intensity()
    algorithm.plot_merged_segments(start_analysis)


    


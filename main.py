import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from pan_tompkins import PanTompkins, find_Q_S_peaks, calculate_QRS_time

if __name__ == '__main__':
    mat = scipy.io.loadmat("006_Roodhart_PVCs_1.mat")
    SAMPLE_F = 200
    ecg = mat["ecg"]["sig"][0,0][:,1] / 1000
    ecg_time_adjust = ecg[:20000]
    time = np.array(range(0, 20000)) / SAMPLE_F

    pan_tompkins = PanTompkins(ecg_time_adjust, SAMPLE_F)
    results = pan_tompkins.run()


    q_peak, s_peak = find_Q_S_peaks(ecg_time_adjust,SAMPLE_F,results['R_peaks'])

    QRS_time = calculate_QRS_time(q_peak,s_peak,SAMPLE_F)
    print(QRS_time)

    print(f"Heart Rate: {round(results['heart_rate'], 2)} BPM")
    print(f"Mean RR Interval: {results['regularities']['mean_rr']:.4f} s")
    print(f"RR Interval Std Dev: {results['regularities']['std_rr']:.4f} s")
    print(f"Coefficient of Variation (CV): {results['regularities']['cv_rr']:.4f}")

    # --- Plot ECG signal with QRS detection ---
    plt.figure(figsize=(12, 4))
    plt.plot(time, ecg_time_adjust, label="ECG", color="black", alpha=0.7)

    # Plot R-peaks (red circles)
    plt.scatter(time[results['R_peaks']], ecg_time_adjust[results['R_peaks']],
                color="red", marker="o", label="R-peaks")

    # Plot Q-peaks (blue crosses)
    plt.scatter(time[q_peak], ecg_time_adjust[q_peak], 
                color="blue", marker="x", label="Q-peaks")

    # Plot S-peaks (green crosses)
    plt.scatter(time[s_peak], ecg_time_adjust[s_peak], 
                color="green", marker="x", label="S-peaks")

    # plt.legend()
    # plt.title("ECG Signal with Q, R, S Peaks")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude (mV)")
    # plt.show()

    # plt.figure()
    # plt.plot(time, ecg_time_adjust)
    # #plt.plot(time[results['R_peaks']], results['smoothed'][results['R_peaks']], "ro", label='Peaks')
    # plt.legend()
    # plt.figure()
    # plt.plot(results['rr_intervals'])
    # plt.title("RR Interval Differences")
    plt.show()

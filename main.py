import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from pan_tompkins import PanTompkins

if __name__ == '__main__':
    mat = scipy.io.loadmat("006_Roodhart_PVCs_1.mat")
    SAMPLE_F = 200
    ecg = mat["ecg"]["sig"][0,0][:,1] / 1000
    ecg_time_adjust = ecg[:20000]
    time = np.array(range(0, 20000)) / SAMPLE_F

    pan_tompkins = PanTompkins(ecg_time_adjust, SAMPLE_F)
    results = pan_tompkins.run()

    print(f"Heart Rate: {round(results['heart_rate'], 2)} BPM")
    print(f"Mean RR Interval: {results['regularities']['mean_rr']:.4f} s")
    print(f"RR Interval Std Dev: {results['regularities']['std_rr']:.4f} s")
    print(f"Coefficient of Variation (CV): {results['regularities']['cv_rr']:.4f}")

    plt.plot(time, results['smoothed'])
    plt.plot(time[results['peaks']], results['smoothed'][results['peaks']], "ro", label='Peaks')
    plt.legend()
    plt.figure()
    plt.plot(results['rr_intervals'])
    plt.title("RR Interval Differences")
    plt.show()

# ECG Signal Processing and Analysis

This repository provides an end-to-end solution for ECG signal analysis using various signal processing techniques. The goal of this project is to analyze ECG signals for detecting peaks, heart rate, and removing abnormal peaks like PVCs (Premature Ventricular Contractions). The system uses several algorithms for signal enhancement and analysis, such as the Pan-Tompkins algorithm, FFT analysis, and the detection of AF (Atrial Fibrillation).

## Files

### 1. `main.py`
The main script that orchestrates the workflow, including the loading of ECG data, execution of signal processing techniques, and the visualization of results.

**Key Functions:**
- `load_ecg_data(filepath, sample_freq, begin, end)`: Loads ECG signal data from a `.mat` file.
- `example_plots.analyze_fft()`: Generates FFT analysis and visualizations.
- `PanTompkins`: Runs the Pan-Tompkins algorithm for peak detection and smoothing.
- `replace_pvc_R_peak()`: Replaces PVC-induced R-peaks with median values of adjacent peaks.
- `AF_detection`: Detects and analyzes Atrial Fibrillation segments in the ECG signal.

### 2. `example_plots.py`
Contains helper functions for plotting various types of visualizations of the ECG signal and its analysis.

**Key Functions:**
- `analyze_fft()`: Performs FFT analysis and plots both the frequency spectrum and the time-domain ECG signal.
- `plot_smoothed_ecg()`: Plots the smoothed ECG signal after applying the Pan-Tompkins algorithm.
- `plot_ecg_peaks()`: Plots Q, S, and R peaks on the ECG signal.

### 3. `pan_tompkins.py`
Contains the implementation of the Pan-Tompkins algorithm for detecting R-peaks, along with preprocessing steps such as bandpass filtering and derivative-based peak detection.

**Key Methods:**
- `__bandpass_filter()`: Applies a bandpass filter to the ECG signal.
- `__derivative_filter()`: Computes the derivative of the ECG signal.
- `__find_peaks()`: Detects R-peaks using the filtered ECG signal.
- `run()`: Runs the full Pan-Tompkins algorithm, returning results including heart rate and RR intervals.

### 4. `detection_function.py`
Contains the `AF_detection` class which is used for detecting Atrial Fibrillation (AF) based on FFT analysis and segment merging.

**Key Methods:**
- `high_pass_filter()`: Applies a high-pass filter to remove low-frequency noise.
- `running_fft()`: Calculates the FFT intensity over a sliding window.
- `replace_outliers()`: Detects and replaces outliers in the ECG signal.
- `find_high_energy_segments()`: Identifies segments of high energy, indicating potential AF.

## How to Use

1. **Prepare Your ECG Data:**
   - Ensure that you have a `.mat` file containing ECG signal data. The data should include the raw ECG signal (typically in microvolts or millivolts).

2. **Run the Main Script:**
   - Set the file path and the parameters for the analysis in `main.py`.
   - The script will load the ECG data, apply preprocessing steps, run algorithms like Pan-Tompkins, and produce various plots and statistics.

3. **Visualize Results:**
   - The code will generate plots showing the ECG signal with detected peaks, FFT analysis, and any anomalies such as PVCs or AF.

4. **Customize:**
   - You can adjust parameters such as the sample frequency (`SAMPLE_F`), the range of the analysis (`start_analysis`, `end_analysis`), and the frequency threshold for FFT analysis.

## Dependencies

- `numpy`: Numerical operations
- `scipy`: Signal processing
- `matplotlib`: Plotting and visualization
- `scipy.io`: For loading `.mat` files
- `scipy.signal`: For signal processing functions

You can install the dependencies using `pip`:
```bash
pip install numpy scipy matplotlib

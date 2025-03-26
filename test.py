import numpy as np
import matplotlib.pyplot as plt
import pywt  # Correct package for wavelet transforms
import pywt.data

# Create a synthetic non-stationary signal (a mixture of sinusoids with varying frequency)
def generate_signal(t):
    # Frequency components change over time
    signal = np.sin(2 * np.pi * (5 + 3 * t) * t)  # A sinusoid with a frequency that increases over time
    return signal

# Create time vector
t = np.linspace(0, 1, 1000)  # 1 second sampled at 1000 Hz

# Generate the signal
signal = generate_signal(t)

# Apply the Continuous Wavelet Transform (CWT)
# Use the complex Morlet wavelet ('cmor') for time-frequency analysis
scales = np.arange(1, 128)  # Range of scales to consider
cwt_matrix, frequencies = pywt.cwt(signal, scales, 'cmor')

# Plot the original signal
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Plot the CWT result
plt.subplot(2, 1, 2)
plt.imshow(np.abs(cwt_matrix), extent=[t[0], t[-1], scales[-1], scales[0]], aspect='auto',
           cmap='jet')
plt.title('Wavelet Transform (Time-Frequency Representation)')
plt.xlabel('Time [s]')
plt.ylabel('Scale')
plt.colorbar(label='Magnitude')

plt.tight_layout()
plt.show()

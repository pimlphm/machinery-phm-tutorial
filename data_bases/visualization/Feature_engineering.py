from scipy.fft import fft, fftfreq
import numpy as np

def Feature_engineering(signals, fs, rpm, band_width=50):
    """
    Extracts signal features from both time and frequency domains,
    including energy ratios over fixed-width frequency bands.

    Extracted features:
      - RMS (Root Mean Square)
      - Crest Factor (peak-to-RMS ratio)
      - Kurtosis (statistical measure of distribution shape)
      - 1X Harmonic Amplitude (fundamental frequency)
      - 2X Harmonic Amplitude (second harmonic)
      - Energy ratios in each frequency band (relative to total energy)

    Parameters:
        signals: np.ndarray of shape (N, L)
                 N = number of signals, L = number of time samples per signal
        fs:      Sampling frequency in Hz
        rpm:     Rotational speed in revolutions per minute (RPM)
        band_width: Width of each frequency band in Hz

    Returns:
        X: np.ndarray of shape (N, 5 + num_bands)
           Each row contains the feature vector of one signal
    """

    # Compute the fundamental shaft frequency in Hz
    shaft_freq = rpm / 60  # Convert RPM to Hz

    # Length of each signal (number of time-domain samples)
    L = signals.shape[1]

    # Frequency bins corresponding to the FFT results
    freqs = fftfreq(L, 1/fs)  # Returns values from 0 to Nyquist and negative frequencies

    # Nyquist frequency (half the sampling rate)
    max_freq = fs / 2

    # Define the edges of each frequency band: [0, band_width, 2*band_width, ..., max_freq]
    band_edges = np.arange(0, max_freq + band_width, band_width)

    # Create a list of boolean masks for each frequency band
    # Each mask selects FFT bins within the corresponding band
    band_masks = [
        (freqs >= low) & (freqs < high)
        for low, high in zip(band_edges[:-1], band_edges[1:])
    ]

    # Initialize the list that will hold feature vectors for all signals
    feature_list = []

    # Process each signal individually
    for x in signals:

        # === Time-Domain Features ===

        # Root Mean Square (energy measure)
        rms = np.sqrt(np.mean(x**2))

        # Crest Factor = Peak value divided by RMS
        crest = np.max(np.abs(x)) / rms if rms > 0 else 0

        # Standard deviation (used in kurtosis computation)
        std = np.std(x)

        # Kurtosis = measure of "tailedness" of the signal's amplitude distribution
        kurt = np.mean(((x - np.mean(x)) / std)**4) if std > 0 else 3

        # === Frequency-Domain Features ===

        # Compute the FFT (magnitude only) of the signal
        fft_vals = np.abs(fft(x))

        # Total energy in the frequency domain (used for normalization)
        total_energy = np.sum(fft_vals**2)

        # Identify frequency bins near 1X (fundamental) harmonic: ±2 Hz around shaft frequency
        h1_mask = (freqs > shaft_freq - 2) & (freqs < shaft_freq + 2)

        # Identify frequency bins near 2X harmonic (second harmonic): ±2 Hz around 2*shaft frequency
        h2_mask = (freqs > 2*shaft_freq - 2) & (freqs < 2*shaft_freq + 2)

        # Extract maximum amplitude within each harmonic band (if any values are found)
        h1 = np.max(fft_vals[h1_mask]) if np.any(h1_mask) else 0
        h2 = np.max(fft_vals[h2_mask]) if np.any(h2_mask) else 0

        # === Frequency Band Energy Ratios ===

        # For each frequency band, compute the proportion of total FFT energy
        band_ratios = [
            np.sum(fft_vals[mask]**2) / (total_energy + 1e-12)  # Add small constant to avoid division by zero
            for mask in band_masks
        ]

        # === Combine All Features ===

        # Combine time-domain, harmonic, and band energy ratio features into one vector
        features = [rms, crest, kurt, h1, h2] + band_ratios

        # Append the feature vector to the full list
        feature_list.append(features)

    # Convert the list of feature vectors into a 2D NumPy array (N samples × D features)
    return np.array(feature_list)

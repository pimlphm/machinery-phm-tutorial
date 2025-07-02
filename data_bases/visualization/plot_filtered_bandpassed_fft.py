from scipy.signal import butter, filtfilt
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Create a Butterworth band-pass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Create bandpass filter coefficients using Butterworth design.
    - lowcut: Minimum frequency to keep (Hz)
    - highcut: Maximum frequency to keep (Hz)
    - fs: Sampling frequency (Hz)
    """
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

# Apply band-pass filter to a signal
def apply_bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Filter a signal to keep only frequency components between lowcut and highcut.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, signal)

# Main function to plot filtered FFT spectrum with an interactive slider
def plot_filtered_bandpassed_fft(normal_data, misalign_data, unbalance_data,
                                 looseness_data, fs, duration=0.2, indices=None,
                                 lowcut=5, highcut=1000, fmax_init=500, fmax_slider=2000):
    """
    Plot FFT spectrum after applying band-pass filtering to vibration signals.

    Parameters:
        normal_data, misalign_data, unbalance_data, looseness_data : np.ndarray
            Arrays of vibration signals for each condition. Shape: (num_samples, signal_length)

        fs : int
            Sampling rate in Hz (e.g., 25000 means 25000 samples per second)

        duration : float
            Duration of signal (in seconds) to use for FFT. (e.g., 0.2s = 5000 samples at 25kHz)

        indices : dict
            Dictionary of which sample index to use from each dataset.
            Example: {'Normal': 0, 'Misalignment': 2, 'Unbalance': 1, 'Looseness': 4}

        lowcut : float
            Minimum frequency to keep in the filter (Hz). Removes DC and very low frequencies.

        highcut : float
            Maximum frequency to keep in the filter (Hz). Removes high-frequency noise.

        fmax_init : float
            Initial frequency range shown in the plot (x-axis max).

        fmax_slider : float
            Maximum frequency you can view by using the interactive slider.
    """
    if indices is None:
        indices = {'Normal': 0, 'Misalignment': 0, 'Unbalance': 0, 'Looseness': 0}

    display_samples = int(duration * fs)  # Number of points to use from each signal
    fault_types = ['Normal', 'Misalignment', 'Unbalance', 'Looseness']
    signals = [normal_data, misalign_data, unbalance_data, looseness_data]
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']  # Color for each fault type

    # Create a 2x2 plot layout
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        f"{ftype} (Index {indices[ftype]})" for ftype in fault_types
    ])

    for i, (label, data, color) in enumerate(zip(fault_types, signals, colors)):
        # Select the signal by index and crop it to the desired duration
        raw_signal = data[indices[label]][:display_samples]

        # Apply band-pass filter to remove noise outside target frequency band
        filtered_signal = apply_bandpass_filter(raw_signal, lowcut, highcut, fs)

        # Compute FFT of the filtered signal
        freqs = np.fft.rfftfreq(display_samples, d=1/fs)
        spectrum = np.abs(np.fft.rfft(filtered_signal))

        # Keep only valid frequency range (between lowcut and slider limit)
        valid = (freqs > lowcut) & (freqs <= fmax_slider)
        freqs = freqs[valid]
        spectrum = spectrum[valid]

        # Determine subplot position (row and column)
        row = i // 2 + 1
        col = i % 2 + 1

        # Plot each spectrum
        fig.add_trace(
            go.Scatter(x=freqs, y=spectrum, mode='lines', name=label,
                       line=dict(color=color)),
            row=row, col=col
        )

        # Set axis labels and initial x-axis range
        fig.update_xaxes(title_text='Frequency (Hz)', row=row, col=col, range=[lowcut, fmax_init])
        fig.update_yaxes(title_text='Amplitude', row=row, col=col)

    # Set global plot layout
    fig.update_layout(
        height=700, width=1000,
        title='Band-Pass Filtered FFT Spectrum (Interactive)',
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    # Add an interactive slider to zoom frequency axis
    fig.update_layout(
        sliders=[{
            'active': 0,
            'currentvalue': {"prefix": "Max Frequency: "},
            'pad': {"t": 50},
            'steps': [
                {
                    'label': f'{f_max}Hz',
                    'method': 'relayout',
                    'args': [
                        {
                            'xaxis.range': [lowcut, f_max],
                            'xaxis2.range': [lowcut, f_max],
                            'xaxis3.range': [lowcut, f_max],
                            'xaxis4.range': [lowcut, f_max],
                        }
                    ]
                }
                for f_max in range(fmax_init, fmax_slider + 1, 200)
            ]
        }]
    )

    # Show the interactive plot
    fig.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_vibration_signals(normal_data, misalign_data, unbalance_data,
                           looseness_data, fs, duration=0.2, figsize=(14,8),
                           indices=None):
    """
    Plot vibration signals in the time domain from four fault conditions.
    Allows custom selection of sample index per condition.

    Parameters:
        normal_data, misalign_data, unbalance_data, looseness_data: np.ndarray
            Arrays of shape [n_samples, n_timesteps] for each condition.
        fs: float
            Sampling rate in Hz.
        duration: float
            Duration (in seconds) to display from each signal. Default is 0.2s.
        figsize: tuple
            Size of the output figure (width, height).
        indices: dict or None
            Optional dictionary specifying sample index for each condition.
            e.g., {'Normal': 0, 'Misalignment': 2, 'Unbalance': 0, 'Looseness': 5}
            If None, defaults to index 0 for all.
    """
    display_samples = int(duration * fs)
    time_axis = np.linspace(0, display_samples / fs, display_samples)  # in seconds

    # Default indices if none provided
    if indices is None:
        indices = {
            'Normal': 0,
            'Misalignment': 0,
            'Unbalance': 0,
            'Looseness': 0
        }

    # Extract signals using custom indices
    signals = {
        'Normal':       normal_data[indices['Normal'], :display_samples],
        'Misalignment': misalign_data[indices['Misalignment'], :display_samples],
        'Unbalance':    unbalance_data[indices['Unbalance'], :display_samples],
        'Looseness':    looseness_data[indices['Looseness'], :display_samples],
    }

    # Define color palette
    colors = {
        'Normal':       '#1f77b4',
        'Misalignment': '#d62728',
        'Unbalance':    '#2ca02c',
        'Looseness':    '#ff7f0e',
    }

    # Plot configuration
    plt.style.use('seaborn-v0_8-white')
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Vibration Signals\n(Time Domain, Duration = {duration:.2f}s)',
                 fontsize=20, fontweight='bold', y=1.02)
    axes = axes.flatten()

    for ax, (label, signal) in zip(axes, signals.items()):
        ax.plot(time_axis * 1000, signal, color=colors[label], linewidth=1.8)
        ax.set_title(f"{label} (Index {indices[label]})", fontsize=16, fontweight='bold')
        ax.set_xlabel('Time (ms)', fontsize=14)
        ax.set_ylabel('Amplitude (g)', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

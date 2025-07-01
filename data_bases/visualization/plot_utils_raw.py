import numpy as np
import matplotlib.pyplot as plt

def plot_vibration_signals(normal_data,
                           misalign_data,
                           unbalance_data,
                           looseness_data,
                           fs,
                           duration=0.2,
                           channel=0,
                           figsize=(14, 8)):
    """
    Plot time-domain vibration signals for different fault types.

    Parameters:
        normal_data (ndarray): Array of shape [n_samples, n_timesteps] for normal condition.
        misalign_data (ndarray): Array for misalignment condition.
        unbalance_data (ndarray): Array for unbalance condition.
        looseness_data (ndarray): Array for looseness condition.
        fs (int or float): Sampling rate in Hz.
        duration (float): Length of signal to display in seconds. Default is 0.2s.
        channel (int): Channel index to visualize (used as row index). Default is 0.
        figsize (tuple): Size of the output figure. Default is (14, 8).
    """

    # Number of samples to display
    display_samples = int(duration * fs)
    time_axis = np.linspace(0, display_samples / fs, display_samples)  # in seconds

    # Extract signals from specified channel (sample index)
    signals = {
        'Normal':       normal_data[channel, :display_samples],
        'Misalignment': misalign_data[channel, :display_samples],
        'Unbalance':    unbalance_data[channel, :display_samples],
        'Looseness':    looseness_data[channel, :display_samples],
    }

    # High-contrast color palette
    colors = {
        'Normal':       '#1f77b4',  # Blue
        'Misalignment': '#d62728',  # Red
        'Unbalance':    '#2ca02c',  # Green
        'Looseness':    '#ff7f0e',  # Orange
    }

    # Plot configuration
    plt.style.use('seaborn-v0_8-white')
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Vibration Signals (Sample {channel}, Time Domain, {duration:.2f}s)',
                 fontsize=20, fontweight='bold', y=1.02)
    axes = axes.flatten()

    for ax, (label, signal) in zip(axes, signals.items()):
        ax.plot(time_axis * 1000, signal, color=colors[label], linewidth=1.8)
        ax.set_title(label, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time (ms)', fontsize=14)
        ax.set_ylabel('Amplitude (g)', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

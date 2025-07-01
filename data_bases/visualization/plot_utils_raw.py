# plot_utils.py

import numpy as np
import matplotlib.pyplot as plt

def plot_vibration_signals(normal_data, misalign_data, unbalance_data,
                           looseness_data, fs, duration=0.2, figsize=(14,8)):
    """
Plot vibration signal (time domain), only display the data of the first duration seconds.

Parameters:
normal_data, misalign_data, unbalance_data, looseness_data:
Arrays of vibration signals in four states, shape [n_samples, n_timepoints]
fs: sampling rate (Hz)
duration: display time length (seconds), default 0.2s
figsize: image size tuple, default (14, 8)
    """
    # calculating the needed points
    display_samples = int(duration * fs)
    time_axis = np.linspace(0, display_samples / fs, display_samples)  # 单位：秒

    # Only the first sample
    signals = {
        'Normal':      normal_data[0, :display_samples],
        'Misalignment':misalign_data[0, :display_samples],
        'Unbalance':   unbalance_data[0, :display_samples],
        'Looseness':   looseness_data[0, :display_samples],
    }

    colors = {
        'Normal':       '#1f77b4',
        'Misalignment': '#d62728',
        'Unbalance':    '#2ca02c',
        'Looseness':    '#ff7f0e',
    }

    plt.style.use('seaborn-v0_8-white')
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Vibration Signals\n(First Sample, Time Domain, {duration}s)',
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

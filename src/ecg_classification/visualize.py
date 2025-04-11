"""visualize

Module for visualizing data
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

def plot_ecg(signal: npt.NDArray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    time = np.arange(len(signal))

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(15, 3))
    ax.plot(time, signal)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Amplitude")
    return ax

def plot_multiple_ecg(signals: list[npt.NDArray]) -> plt.Axes:
    fig, ax = plt.subplots(1, 1, figsize=(15, 3))
    for s in signals:
        time = np.arange(len(s))
        ax.plot(time, s)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Amplitude")
    return ax
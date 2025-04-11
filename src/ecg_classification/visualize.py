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

def plot_multiple_ecg(signals: list[npt.NDArray], labels: list[str], layout: str = "vstack") -> plt.Axes:
    _layout = None
    if layout == "overlay":
        _layout = (1, 1)
    elif layout == "vstack":
        _layout = (len(signals), 1)
    elif layout == "hstack":
        _layout = (1, len(signals))

    fig, ax = plt.subplots(*_layout, figsize=(15, 3))
    for i, (s, lbl) in enumerate(zip(signals, labels)):
        time = np.arange(len(s))
        if layout == "overlay":
            ax.plot(time, s, label=lbl)
        else:
            ax[i].plot(time, s, label=lbl)
            ax[i].legend()
    return fig, ax

    # if layout == "overlay":
    #     fig, ax = plt.subplots(1, 1, figsize=(15, 3))
    #     for s, lbl in zip(signals, labels):
    #         ax.plot(time, s, label=lbl)
    #     ax.set_xlabel("Time (sec)")
    #     ax.set_ylabel("Amplitude")
    #     return ax
    # elif layout == "vstack":
    #     for i, s, lbl in enumerate(zip(signals, labels)):
    #         ax = plt.subplot(2, 1, i + 1)
    #         plt.plot(s)
    #         plt.title("Original")
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    # elif layout == "hstack":
    #     for i, s, lbl in enumerate(zip(signals, labels)):
    #         ax = plt.subplot(2, 1, i + 1)
    #         plt.plot(s)
    #         plt.title("Original")
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    # return ax
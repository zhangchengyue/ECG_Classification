"""preprocessing

Helper functions for preprocessing
"""

import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve1d
from scipy.signal import detrend
from sklearn.preprocessing import normalize

def preprocess_ecg_signals(X: npt.NDArray, conv_window_size: int = 3) -> npt.NDArray:
    """Apply basic preprocessing to a matrix of ecg signals.
    Each row of `X` is an independent ecg signal."""
    og_shape = X.shape
    # 1. Moving average filter (+ zero padding to maintain original length)
    X = convolve1d(X, np.ones(conv_window_size), mode="constant", axis=1) / conv_window_size
    # 2. De-trend, so classifier doesn't pay too much attention to baseline wander noise
    X = detrend(X, type="linear")
    # 3. Normalize, so classifer doesn't pay too much attention to the specific amplitude values
    X = normalize(X, axis=1, norm="max")
    assert X.shape == og_shape, f"Shape should not change (input: {og_shape}, output: {X.shape})"
    return X


class SyntheticNoise:
    """Generates synthetic noise"""

    def __init__(self, random_state: int):
        self.rng = np.random.default_rng(seed=random_state)

    def gaussian_noise(self, x: npt.NDArray, stddev: float) -> npt.NDArray:
        noisy_array = x + np.random.normal(loc=0.0, scale=stddev, size=x.shape)
        noisy_array = normalize(noisy_array, norm="max", axis=1)
        return noisy_array

    def powerline_interference(self):
        raise NotImplementedError()

    def muscle_artifacts(self):
        raise NotImplementedError()

    def baseline_wander(self):
        raise NotImplementedError()
"""Fixed-effect helper functions."""

import numpy as np


def scale_binary_FE(x: np.ndarray, variance: float) -> np.ndarray:
    '''
    Mean-centers and rescales a 1D binary/two-level array to have a specified variance.
    Parameters:
        x (1D array): Array containing exactly two unique observed values.
        variance (float): Target variance for the output array. Must be non-negative.
    Returns:
        x_scaled (1D array): Array of the same length as `x`, centered at 0 and with variance `variance`.
    '''
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input `x` must be a 1D array.")
    if variance < 0:
        raise ValueError("Input `variance` must be non-negative.")

    unique_vals = np.unique(x)
    if unique_vals.size != 2:
        raise ValueError("Input `x` must contain exactly two unique observed values.")

    x_centered = x - x.mean()
    x_std = x_centered.std()
    if x_std == 0:
        raise ValueError("Input `x` must have non-zero variance.")

    x_scaled = x_centered * np.sqrt(variance) / x_std
    return x_scaled

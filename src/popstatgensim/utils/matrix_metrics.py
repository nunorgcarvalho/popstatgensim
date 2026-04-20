"""Reusable comparison metrics for observed versus imputed matrices."""

from __future__ import annotations

import numpy as np


def _coerce_matrix_pair(observed: np.ndarray, imputed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    imputed = np.asarray(imputed, dtype=float)
    if observed.ndim != 2 or imputed.ndim != 2:
        raise ValueError("observed and imputed must both be 2D arrays.")
    if observed.shape != imputed.shape:
        raise ValueError("observed and imputed must have the same shape.")
    return observed, imputed


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_ss = float(np.dot(x_centered, x_centered))
    y_ss = float(np.dot(y_centered, y_centered))
    if np.isclose(x_ss, 0.0) or np.isclose(y_ss, 0.0):
        return np.nan
    return float(np.dot(x_centered, y_centered) / np.sqrt(x_ss * y_ss))


def matrix_mean_squared_error(observed: np.ndarray, imputed: np.ndarray) -> float:
    observed, imputed = _coerce_matrix_pair(observed, imputed)
    return float(np.mean((imputed - observed) ** 2))


def matrix_root_mean_squared_error(observed: np.ndarray, imputed: np.ndarray) -> float:
    return float(np.sqrt(matrix_mean_squared_error(observed, imputed)))


def matrix_mean_absolute_error(observed: np.ndarray, imputed: np.ndarray) -> float:
    observed, imputed = _coerce_matrix_pair(observed, imputed)
    return float(np.mean(np.abs(imputed - observed)))


def matrix_pearson_correlation(observed: np.ndarray, imputed: np.ndarray) -> float:
    observed, imputed = _coerce_matrix_pair(observed, imputed)
    return _safe_corr(observed.ravel(), imputed.ravel())


def matrix_axiswise_correlation(observed: np.ndarray, imputed: np.ndarray,
                                axis: int = 1) -> np.ndarray:
    observed, imputed = _coerce_matrix_pair(observed, imputed)
    if axis not in {0, 1}:
        raise ValueError("axis must be 0 (columns) or 1 (rows).")

    if axis == 0:
        observed = observed.T
        imputed = imputed.T

    corrs = np.full(observed.shape[0], np.nan, dtype=float)
    for idx in range(observed.shape[0]):
        corrs[idx] = _safe_corr(observed[idx], imputed[idx])
    return corrs


def summarize_matrix_error(observed: np.ndarray, imputed: np.ndarray) -> dict:
    observed, imputed = _coerce_matrix_pair(observed, imputed)
    row_corr = matrix_axiswise_correlation(observed, imputed, axis=1)
    col_corr = matrix_axiswise_correlation(observed, imputed, axis=0)
    diff = imputed - observed

    def _nan_stat(values: np.ndarray, fn) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.nan
        return float(fn(finite))

    return {
        "mse": matrix_mean_squared_error(observed, imputed),
        "rmse": matrix_root_mean_squared_error(observed, imputed),
        "mae": matrix_mean_absolute_error(observed, imputed),
        "bias": float(np.mean(diff)),
        "corr": matrix_pearson_correlation(observed, imputed),
        "mean_row_corr": _nan_stat(row_corr, np.mean),
        "median_row_corr": _nan_stat(row_corr, np.median),
        "mean_col_corr": _nan_stat(col_corr, np.mean),
        "median_col_corr": _nan_stat(col_corr, np.median),
        "n_defined_row_corr": int(np.isfinite(row_corr).sum()),
        "n_defined_col_corr": int(np.isfinite(col_corr).sum()),
    }

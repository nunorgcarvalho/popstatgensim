"""Statistical helper functions used across popstatgensim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.stats import norm


@dataclass
class LinearRegressionResult:
    """Container for ordinary least-squares regression outputs."""

    coef: np.ndarray
    coef_se: np.ndarray
    coef_vcov: np.ndarray
    n_samples: int
    n_predictors: int
    n_parameters: int
    dof_resid: int
    y_mean: float
    y_var: float
    residual_var: float
    rss: float


def corr(x, y):
    '''
    Computes Pearson correlation coefficient between two vectors x and y.
    Parameters:
        x (1D array): First vector.
        y (1D array): Second vector.
    Returns:
        r (float): Pearson correlation coefficient between x and y.
    '''
    x_norm = (x - x.mean()) / np.sqrt(np.var(x))
    y_norm = (y - y.mean()) / np.sqrt(np.var(y))
    r = (x_norm * y_norm).mean()
    return r


def standardize_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Mean-center and variance-standardize a vector."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f'`{name}` must be a 1D array.')
    sd = float(np.std(values))
    if not np.isfinite(sd) or np.isclose(sd, 0.0):
        raise ValueError(f'Cannot standardize `{name}` because it has zero variance.')
    return (values - float(np.mean(values))) / sd


def fit_linear_regression(
    y: np.ndarray,
    X: np.ndarray,
    add_intercept: bool = True,
) -> LinearRegressionResult:
    """Fit an ordinary least-squares linear regression."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    if y.ndim != 1:
        raise ValueError('`y` must be a 1D array.')
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError('`X` must be a 1D or 2D array.')
    if X.shape[0] != y.shape[0]:
        raise ValueError('`X` and `y` must have the same number of rows.')
    if not np.isfinite(y).all():
        raise ValueError('`y` contains non-finite values.')
    if not np.isfinite(X).all():
        raise ValueError('`X` contains non-finite values.')

    n_samples = int(y.shape[0])
    n_predictors = int(X.shape[1])
    if add_intercept:
        design = np.column_stack((np.ones((n_samples, 1), dtype=float), X))
    else:
        design = X

    n_parameters = int(design.shape[1])
    dof_resid = n_samples - n_parameters
    if dof_resid <= 0:
        raise ValueError('Not enough samples to fit the linear regression.')

    xtx_inv = np.linalg.pinv(design.T @ design)
    coef = xtx_inv @ (design.T @ y)
    resid = y - (design @ coef)
    rss = float(resid @ resid)
    residual_var = rss / dof_resid
    coef_vcov = residual_var * xtx_inv
    coef_se = np.sqrt(np.clip(np.diag(coef_vcov), 0.0, None))

    return LinearRegressionResult(
        coef=np.asarray(coef, dtype=float),
        coef_se=np.asarray(coef_se, dtype=float),
        coef_vcov=np.asarray(coef_vcov, dtype=float),
        n_samples=n_samples,
        n_predictors=n_predictors,
        n_parameters=n_parameters,
        dof_resid=int(dof_resid),
        y_mean=float(np.mean(y)),
        y_var=float(np.var(y)),
        residual_var=float(residual_var),
        rss=float(rss),
    )


def report_CI(point_and_se: Tuple[float, float], CI: float = 0.95) -> str:
    '''
    Returns a string with the point estimate and confidence interval.
    Parameters:
        point_and_se (list): Tuple containing the point estimate and standard error (point, se).
        CI (float): Confidence level. Default is 0.95.
    Returns:
        report (str): String with the point estimate and confidence interval.
    '''
    (point, se) = point_and_se
    z = norm.ppf((1 + CI) / 2)  # z-score for the given confidence level
    lower = point - z * se
    upper = point + z * se
    return f'{point:.3f} [{lower:.3f}, {upper:.3f}]'


__all__ = [
    'LinearRegressionResult',
    'corr',
    'fit_linear_regression',
    'report_CI',
    'standardize_vector',
]

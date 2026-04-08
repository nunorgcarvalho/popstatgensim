"""Statistical helper functions used across popstatgensim."""

import numpy as np
from scipy.stats import norm
from typing import Tuple


def corr(x,y):
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

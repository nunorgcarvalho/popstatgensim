"""Polygenic score weight simulation helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np


def simulate_pgs_standardized_weights(
    beta: np.ndarray,
    h2: float,
    R2: Optional[float] = None,
    N: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    '''
    Simulates estimated standardized polygenic score weights.

    Parameters
    ----------
    beta : np.ndarray
        1D array of true standardized variant effects.
    h2 : float
        Additive heritability used in the R2-based noise-variance formula.
    R2 : float, optional
        Expected prediction R^2 of the polygenic score. Mutually exclusive with N.
    N : int, optional
        GWAS sample size used in the large-sample approximation V_eps = 1 / N.
        Mutually exclusive with R2.
    rng : np.random.Generator, optional
        Random number generator used to draw estimation noise.
    seed : int, optional
        Seed used to initialize a new random generator when rng is not provided.

    Returns
    -------
    np.ndarray
        Estimated standardized weights with the same shape as beta.
    '''
    beta = np.asarray(beta, dtype=float)
    if beta.ndim != 1:
        raise ValueError('beta must be a 1D array.')
    if beta.size == 0:
        raise ValueError('beta must be non-empty.')
    if h2 < 0:
        raise ValueError('h2 must be non-negative.')

    provided = int(R2 is not None) + int(N is not None)
    if provided != 1:
        raise ValueError('Provide exactly one of R2 or N.')
    if rng is not None and seed is not None:
        raise ValueError('Provide only one of rng or seed.')

    M = beta.shape[0]
    if R2 is not None:
        if R2 <= 0:
            raise ValueError('R2 must be positive.')
        V_eps = (h2 / M) * ((h2 / R2) - 1.0)
        if V_eps < -1e-12:
            raise ValueError('R2 must be less than or equal to h2 so that V_eps is non-negative.')
        V_eps = max(V_eps, 0.0)
    else:
        if N <= 0:
            raise ValueError('N must be a positive integer.')
        V_eps = 1.0 / float(N)

    rng = np.random.default_rng(seed) if rng is None else rng
    epsilon = rng.normal(loc=0.0, scale=np.sqrt(V_eps), size=M)
    return beta + epsilon

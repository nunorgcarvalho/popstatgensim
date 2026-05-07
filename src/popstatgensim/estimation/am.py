"""Assortative-mating estimation utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..genome.pca import compute_PCA


def _normalize_chrom_idx(chrom_idx: Sequence[int] | np.ndarray, M: int) -> np.ndarray:
    """Validate and normalize chromosome start indices."""
    chrom_idx = np.asarray(chrom_idx, dtype=int)
    if chrom_idx.ndim != 1:
        raise ValueError('`chrom_idx` must be a 1D array-like of chromosome start indices.')
    if chrom_idx.size == 0:
        chrom_idx = np.array([0], dtype=int)
    if 0 not in chrom_idx:
        chrom_idx = np.concatenate([np.array([0], dtype=int), chrom_idx])

    chrom_idx = np.unique(chrom_idx)
    if chrom_idx[0] != 0:
        raise ValueError('`chrom_idx` must include chromosome start index 0.')
    if np.any(chrom_idx < 0) or np.any(chrom_idx >= M):
        raise ValueError(f'`chrom_idx` entries must lie between 0 and {M - 1}.')
    return chrom_idx


def _standardize_vector(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f'`{name}` must be a 1D array.')
    sd = float(values.std())
    if not np.isfinite(sd) or np.isclose(sd, 0.0):
        raise ValueError(f'`{name}` has zero variance and cannot be standardized.')
    return (values - values.mean()) / sd


def _fit_linear_regression(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fits OLS and returns coefficient estimates with standard errors."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if y.ndim != 1:
        raise ValueError('`y` must be a 1D array.')
    if X.ndim != 2:
        raise ValueError('`X` must be a 2D array.')
    if X.shape[0] != y.shape[0]:
        raise ValueError('`X` and `y` must have the same number of rows.')

    n_samples, n_params = X.shape
    dof = n_samples - n_params
    if dof <= 0:
        raise ValueError('Not enough samples to fit the even-odd regression.')

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - (X @ beta)
    sigma2 = float(resid @ resid) / dof
    vcov = sigma2 * xtx_inv
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))
    return beta, se


def _compute_predictor_pcs(G: np.ndarray, n_pcs: int, standardized_geno: bool) -> np.ndarray:
    """Computes predictor-side chromosome PCs for EO-AM adjustment."""
    if n_pcs <= 0:
        return np.empty((G.shape[0], 0), dtype=float)
    if standardized_geno:
        pca = compute_PCA(X=G, n_components=n_pcs)
    else:
        p = np.asarray(G.mean(axis=0) / 2.0, dtype=float)
        pca = compute_PCA(G=G, p=p, P=2, n_components=n_pcs)
    return np.asarray(pca.scores[:, :n_pcs], dtype=float)


def run_EO_AM(
    X: np.ndarray,
    pgs_weights: np.ndarray,
    chrom_idx: Sequence[int] | np.ndarray,
    adjust_PCs: int = 0,
    even_against_odd: bool = True,
    standardized_geno: bool = False,
) -> dict:
    r"""
    Estimate assortative mating from even- versus odd-chromosome polygenic scores.

    This implements the even-odd chromosome regression approach described by
    Yengo et al. using standardized chromosome-specific PGS values, optionally
    adjusting for principal components computed from the predictor-side
    chromosome set only.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('`X` must be a 2D array.')
    if not np.isfinite(X).all():
        raise ValueError('`X` contains non-finite values.')

    pgs_weights = np.asarray(pgs_weights, dtype=float)
    if pgs_weights.ndim != 1:
        raise ValueError('`pgs_weights` must be a 1D array.')
    if not np.isfinite(pgs_weights).all():
        raise ValueError('`pgs_weights` contains non-finite values.')
    if X.shape[1] != pgs_weights.shape[0]:
        raise ValueError(
            f'`pgs_weights` must have length {X.shape[1]} to match the number of variants in `X`.'
        )

    adjust_PCs = int(adjust_PCs)
    if adjust_PCs < 0:
        raise ValueError('`adjust_PCs` must be non-negative.')

    M = X.shape[1]
    chrom_idx = _normalize_chrom_idx(chrom_idx=chrom_idx, M=M)
    chrom_end = np.concatenate([chrom_idx[1:], np.array([M], dtype=int)])

    odd_mask = np.zeros(M, dtype=bool)
    even_mask = np.zeros(M, dtype=bool)
    for chrom_number, (start, stop) in enumerate(zip(chrom_idx, chrom_end), start=1):
        if chrom_number % 2 == 1:
            odd_mask[start:stop] = True
        else:
            even_mask[start:stop] = True

    if not odd_mask.any() or not even_mask.any():
        raise ValueError(
            'At least two chromosomes are required to compute even-odd assortative mating.'
        )

    pgs_odd = X[:, odd_mask] @ pgs_weights[odd_mask]
    pgs_even = X[:, even_mask] @ pgs_weights[even_mask]
    pgs_odd = _standardize_vector(pgs_odd, name='odd PGS')
    pgs_even = _standardize_vector(pgs_even, name='even PGS')

    if even_against_odd:
        y = pgs_even
        predictor = pgs_odd
        pc_source = X[:, odd_mask]
        predictor_label = 'odd'
        outcome_label = 'even'
    else:
        y = pgs_odd
        predictor = pgs_even
        pc_source = X[:, even_mask]
        predictor_label = 'even'
        outcome_label = 'odd'

    covariates = _compute_predictor_pcs(
        G=np.asarray(pc_source, dtype=float),
        n_pcs=adjust_PCs,
        standardized_geno=standardized_geno,
    )

    intercept = np.ones((X.shape[0], 1), dtype=float)
    design = np.column_stack([intercept, predictor[:, None], covariates])
    beta, se = _fit_linear_regression(y=y, X=design)

    covariate_names = [f'PC{i}' for i in range(1, covariates.shape[1] + 1)]
    return {
        'theta_est': float(beta[1]),
        'theta_se': float(se[1]),
        'intercept_est': float(beta[0]),
        'intercept_se': float(se[0]),
        'covar_est': np.asarray(beta[2:], dtype=float),
        'covar_se': np.asarray(se[2:], dtype=float),
        'covariate_names': covariate_names,
        'n_covariates': int(covariates.shape[1]),
        'even_against_odd': bool(even_against_odd),
        'predictor_side': predictor_label,
        'outcome_side': outcome_label,
    }


__all__ = ["run_EO_AM"]

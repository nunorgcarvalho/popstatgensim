"""Assortative-mating estimation utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np


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


def run_EO_AM(
    X: np.ndarray,
    pgs_weights: np.ndarray,
    chrom_idx: Sequence[int] | np.ndarray,
) -> float:
    r"""
    Estimate assortative mating from even- versus odd-chromosome polygenic scores.

    This implements the even-odd chromosome correlation approach described by
    Yengo et al.:
    Yengo L, Robinson MR, Keller MC, et al. *Imprint of assortative mating on
    the human genome*. Nature Human Behaviour. 2018;2(12):948-954.
    doi:10.1038/s41562-018-0476-3

    Parameters
    ----------
    X : ndarray, shape (N, M)
        Standardized genotype matrix.
    pgs_weights : ndarray, shape (M,)
        Variant weights used to form the polygenic scores.
    chrom_idx : array-like of int
        Variant indices marking the start of each chromosome. Index 0 is always
        treated as a chromosome start.

    Returns
    -------
    float
        Pearson correlation between the odd-chromosome and even-chromosome PGS
        values across the same individuals.
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
    return float(np.corrcoef(pgs_odd, pgs_even)[0, 1])


__all__ = ["run_EO_AM"]

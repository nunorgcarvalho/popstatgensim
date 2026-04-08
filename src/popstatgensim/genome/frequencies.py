"""Allele-frequency simulation and summary helpers."""

import numpy as np

def compute_freqs(G: np.ndarray, P: int) -> np.ndarray:
    '''
    Computes allele frequencies in genotypes.
    Parameters:
        G (2D array): N*M matrix of genotypes. First dimension is individuals, second dimension is variants. Each element is an integer ranging from 0 to P (the ploidy).
        P (int): Ploidy of genotype matrix.
    Returns:
        p (1D array): Array of length M containing allele frequencies.
    '''
    p = G.mean(axis=0) / P
    return p

def get_FST(p1: np.ndarray, p2: np.ndarray, method: str = 'wright',
            N1: int = None, N2: int = None) -> float:
    '''
    Computes FST between two populations from vectors of allele frequencies.
    Parameters:
        p1 (1D array): Array of allele frequencies for population 1.
        p2 (1D array): Array of allele frequencies for population 2. Must have the same length as `p1`.
        method (str): Method used to estimate FST. Options are:
            'wright': Uses the Wright estimator: (p1 - p2)^2 / (p_bar * (1 - p_bar)), where p_bar is the average allele frequency across the two populations. This is a biased estimator that does not adjust for finite sample sizes.
            'hudson': Uses the Hudson estimator, which adjusts for finite sample sizes. Requires `N1` and `N2`.
        N1 (int): Sample size for population 1. Required if `method='hudson'`.
        N2 (int): Sample size for population 2. Required if `method='hudson'`.
    Returns:
        FST (float): FST estimate across the provided variants.
    '''
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    if p1.ndim != 1 or p2.ndim != 1:
        raise ValueError('`p1` and `p2` must both be 1-dimensional arrays.')
    if p1.shape != p2.shape:
        raise ValueError('`p1` and `p2` must have the same length.')

    valid_mask = np.isfinite(p1) & np.isfinite(p2)
    if not np.any(valid_mask):
        raise ValueError('`p1` and `p2` must contain at least one finite allele frequency.')

    p1 = p1[valid_mask]
    p2 = p2[valid_mask]

    if np.any((p1 < 0) | (p1 > 1)) or np.any((p2 < 0) | (p2 > 1)):
        raise ValueError('Allele frequencies must lie between 0 and 1.')

    method = method.lower()

    if method == 'wright':
        p_bar = (p1 + p2) / 2
        num = (p1 - p2) ** 2
        denom = p_bar * (1 - p_bar)
    elif method == 'hudson':
        if N1 is None or N2 is None:
            raise ValueError('`N1` and `N2` must be provided when `method="hudson"`.')
        if N1 <= 1 or N2 <= 1:
            raise ValueError('`N1` and `N2` must both be greater than 1 for the Hudson estimator.')

        num = (p1 - p2) ** 2 - p1 * (1 - p1) / (N1 - 1) - p2 * (1 - p2) / (N2 - 1)
        denom = p1 + p2 - 2 * p1 * p2
    else:
        raise ValueError("`method` must be either 'exact' or 'hudson'.")

    informative_mask = denom > 0
    if not np.any(informative_mask):
        raise ValueError('No informative variants remain after filtering monomorphic sites.')

    FST = np.mean(num[informative_mask]) / np.mean(denom[informative_mask])
    return FST

def get_fixation_t(ps: np.ndarray = None) -> np.ndarray:
    '''
    For each variant, finds *first* generation (returned as an index) for which the allele frequency is 0 (loss) or 1 (fixation). A value of -1 means the variant never got fixed
    Parameters:
        ps (2D array): T*M matrix (where T is number of generations) containing allele frequencies over time.
    Returns:
        t_fix (1D array): Array of length M with the first generation (as an index) for which the respective allele was lost or fixed. If the allele was not fixed by the most recent simulation, a -1 is returned.
    '''
    # gets mask of whether frequency is 0 or 1
    ps_mask = np.any((ps == 0, ps == 1), axis=0)
    # finds first instance of True for each variant
    t_fix = np.where(ps_mask.any(axis=0), ps_mask.argmax(axis=0), -1)        
    return t_fix

def summarize_ps(ps: np.ndarray = None, quantiles: tuple = (0.25, 0.5, 0.75)) -> tuple[np.ndarray, np.ndarray]:
    '''
    Returns the mean as well as the specified quantiles of variants across each generation.

    Parameters:
        ps (2D array): T*M matrix (where T is number of generations) containing allele frequencies over time.
        quantiles (tuple): List of quantiles (e.g. 0.99) of allele frequencies across variants at each generation to plot. `summarize` must be set to True. Default is median, lower quartile, and upper quartile.
    Returns:
        tuple ((ps_mean, ps_quantile)):
        Where:
        - ps_mean (1D array): Array of length T (where T is the total number of generations) of mean allele frequency at each generation.
        - ps_quantile (2D array): K*T matrix (where K is the number of quantiles specified) of allele frequency for each quantile at each generation.
    '''
    # computes mean allele frequency over time
    ps_mean = ps.mean(axis=1)
    # computes quantiles over time
    ps_quantile = np.quantile(ps, quantiles, axis=1)
    return (ps_mean, ps_quantile)

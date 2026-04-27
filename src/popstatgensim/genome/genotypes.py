"""Genotype-matrix construction and basic transforms."""

import numpy as np


def make_G(H: np.ndarray) -> np.ndarray:
    '''
    Collapses haplotypes into genotypes.
    Parameters:
        H (3D array): N*M*P array of alleles.
    Returns:
        G (2D array): N*M array of genotypes. First dimension is individuals, second dimension is variants. Each element is an integer ranging from 0 to P (the ploidy).
    '''
    G = H.sum(axis=2).astype(np.uint8)
    return G

def center_G(G: np.ndarray, p: np.ndarray, P: int) -> np.ndarray:
    '''
    Centers genotype matrix so that the mean of each column is 0 (or approximately).
    Parameters:
        G (2D array): N*M matrix of genotypes. First dimension is individuals, second dimension is variants. Each element is an integer ranging from 0 to P (the ploidy).
        p (1D array): Array of length M containing allele frequencies from which to center genotypes.
        P (int): Ploidy of genotype matrix.
    Returns:
        G_centered (2D array): Centered genotype matrix.
    '''
    G_centered = G - P*p[None,:]
    return G_centered

def standardize_G(G: np.ndarray, p: np.ndarray, P: int,
                      impute: bool = True, std_method: str = 'observed',
                      target_var: float = 1.0):
    '''
    Standardizes genotype matrix so that each column has mean 0 and variance `target_var` (or approximately).
    Parameters:
        G (2D array): N*M matrix of genotypes. First dimension is individuals, second dimension is variants. Each element is an integer ranging from 0 to P (the ploidy).
        p (1D array): Array of length M containing allele frequencies from which to center genotypes. Also used to scale genotypes.
        P (int): Ploidy of genotype matrix.
        impute (bool): If genotype matrix is a masked array, missing values are filled with the mean genotype value. Default is True.
        std_method (str): Method for calculating genotype standard deviations. If 'observed' (default), then the actual mathematical standard deviation is used. If 'binomial', then the expected standard deviation based on binomial sampling of the allele frequency is used.
        target_var (float): Desired variance of each standardized column. Default is 1.0.
    Returns:
        X (2D array): N*M standardized genotype matrix.
    '''
    # centers genotype matrix
    G = center_G(G, p, P)
    # imputes missing values if specified
    if np.ma.isMaskedArray(G) and impute:
        G[G.mask] = 0
        G = G.data
    # determines standard deviation of genotypes
    if std_method == 'binomial':
        var_G = P * p * (1 - p) 
    elif std_method == 'observed': # Ensures Var[G] = 1
        var_G = G.var(axis=0)
    # replaces monomorph variances with 1 so no divide by 0 error
    var_G[var_G == 0] = 1
    # standardizes genotype matrix
    X = G / np.sqrt(var_G)[None,:]
    if target_var != 1:
        X = X * np.sqrt(target_var)
    return X

def compute_GRM(X: np.ndarray, w: np.ndarray = None) -> np.ndarray:
    '''
    Computes the genetic relationship matrix (GRM) from a standardized genotype matrix (X).
    Parameters:
        X (2D array): N*M standardized genotype matrix.
        w (1D array): Optional array of length M containing non-negative weights
            for each variant. If provided, the GRM is the weighted mean covariance
            across variants.
    Returns:
        GRM (2D array): An N*N genetic relationship matrix. Each element is the mean covariance of standardized genotypes across all variants for the two individuals.
    '''
    M = X.shape[1]
    if w is None:
        # computes GRM
        GRM = (X @ X.T) / M
        return GRM

    w = np.asarray(w, dtype=float)
    if w.ndim != 1:
        raise ValueError("`w` must be a 1D array.")
    if w.shape[0] != M:
        raise ValueError("`w` must have length equal to the number of columns in `X`.")
    if not np.all(np.isfinite(w)):
        raise ValueError("`w` must contain only finite values.")
    if np.any(w < 0):
        raise ValueError("`w` must contain non-negative weights.")

    weight_sum = w.sum()
    if weight_sum <= 0:
        raise ValueError("`w` must contain at least one positive weight.")

    GRM = ((X * w[None, :]) @ X.T) / weight_sum
    return GRM

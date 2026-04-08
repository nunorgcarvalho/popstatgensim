"""Principal-components analysis for genotype matrices."""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy import linalg

from .genotypes import standardize_G


@dataclass
class PCAResult:
    '''
    Container for PCA outputs on genotype data.
    '''
    scores: np.ndarray
    eigenvalues: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    n_samples: int
    n_features: int
    metadata: dict = field(default_factory=dict)

def _validate_pca_axes(pcs: Tuple[int, int]) -> Tuple[int, int]:
    '''
    Validates the pair of principal components requested for plotting.
    '''
    if len(pcs) != 2:
        raise ValueError('pcs must contain exactly two principal-component indices.')
    pc_x, pc_y = (int(pcs[0]), int(pcs[1]))
    if pc_x < 1 or pc_y < 1:
        raise ValueError('Principal-component indices must be 1-based positive integers.')
    if pc_x == pc_y:
        raise ValueError('pcs must contain two distinct principal components.')
    return (pc_x, pc_y)

def _orient_pca_scores(scores: np.ndarray) -> np.ndarray:
    '''
    Flips component signs for more stable plotting orientation across repeated runs.
    '''
    scores = np.asarray(scores, dtype=float)
    for j in range(scores.shape[1]):
        i_max = int(np.argmax(np.abs(scores[:, j])))
        if scores[i_max, j] < 0:
            scores[:, j] *= -1
    return scores

def _format_pc_axis_label(pca: PCAResult, pc: int) -> str:
    '''
    Formats an axis label including the variance explained by a principal component.
    '''
    explained = 100.0 * float(pca.explained_variance_ratio[pc - 1])
    return f'PC{pc} ({explained:.2f}%)'

def compute_PCA(X: np.ndarray = None, G: np.ndarray = None,
                p: np.ndarray = None, P: int = 2,
                n_components: int = 2,
                impute: bool = True,
                std_method: str = 'observed') -> PCAResult:
    '''
    Computes a PCA on standardized genotypes and returns a compact result object.
    Parameters:
        X (2D array): Optional standardized genotype matrix. If provided, must already be
            centered across individuals.
        G (2D array): Optional genotype matrix. Used together with `p` and `P` to build `X`
            if `X` is not provided.
        p (1D array): Allele frequencies used to standardize `G`.
        P (int): Ploidy of genotype matrix.
        n_components (int): Number of leading PCs to compute.
        impute (bool): Passed to `standardize_G()` when constructing `X` from `G`.
        std_method (str): Passed to `standardize_G()` when constructing `X` from `G`.
    Returns:
        PCAResult: Object containing PC scores and variance-explained summaries.
    '''
    if X is None:
        if G is None or p is None:
            raise ValueError('Must provide either X or both G and p to compute PCA.')
        X = standardize_G(G, np.asarray(p, dtype=float), P,
                          impute=impute, std_method=std_method)
    elif G is not None or p is not None:
        raise ValueError('Provide either X or G/p inputs, but not both.')

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('PCA input matrix must be a 2D array.')

    N, M = X.shape
    min_dim = min(N, M)
    if n_components is None:
        n_components = min_dim
    n_components = int(n_components)
    if n_components < 1:
        raise ValueError('n_components must be at least 1.')
    if n_components > min_dim:
        raise ValueError(
            f'n_components={n_components} exceeds the maximum possible rank {min_dim}.'
        )

    frob_norm_sq = float(np.sum(X * X))
    if np.isclose(frob_norm_sq, 0.0):
        raise ValueError(
            'PCA cannot be computed because the standardized genotype matrix has zero total variance.'
        )

    if N <= M:
        gram = X @ X.T
        start = N - n_components
        eigenvalues, eigenvectors = linalg.eigh(
            gram,
            subset_by_index=[start, N - 1],
        )
        eigenvalues = np.clip(eigenvalues[::-1], 0.0, None)
        eigenvectors = eigenvectors[:, ::-1]
        singular_values = np.sqrt(eigenvalues)
        scores = eigenvectors * singular_values[None, :]
    else:
        cov = X.T @ X
        start = M - n_components
        eigenvalues, eigenvectors = linalg.eigh(
            cov,
            subset_by_index=[start, M - 1],
        )
        eigenvalues = np.clip(eigenvalues[::-1], 0.0, None)
        eigenvectors = eigenvectors[:, ::-1]
        singular_values = np.sqrt(eigenvalues)
        scores = X @ eigenvectors

    scores = _orient_pca_scores(scores)
    explained_variance_ratio = eigenvalues / frob_norm_sq

    return PCAResult(
        scores=scores,
        eigenvalues=eigenvalues,
        singular_values=singular_values,
        explained_variance_ratio=explained_variance_ratio,
        n_samples=N,
        n_features=M,
    )

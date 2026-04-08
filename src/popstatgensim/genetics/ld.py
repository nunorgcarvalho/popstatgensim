"""Linkage-disequilibrium helper functions."""

import numpy as np
from scipy import sparse


def make_neighbor_matrix(positions: np.ndarray, LDwindow: float = None) -> sparse.coo_matrix:
    '''
    Makes boolean sparse matrix of variants within the specified LD window distance.
    Parameters:
        positions (array): Array of length M containing physical positions of variants. Positions must already be in ascending order.
        LDwindow (float): Maximum distance between variants to be considered neighbors. In the same units as that provided in `positions`. If not provided, defaults to infinite maximum distance.
    Returns:
        neighbor_matrix (sparse 2D matrix (COO)): An M*M scipy sparse matrix with boolean values, where a 1 at (i,j) indicates that variant i and j are within `LDwindow` of each other, and 0 if not. Returned in COO sparse format.
    '''
    if LDwindow is None:
        LDwindow = positions[-1] - positions[0]
    # Initialize lists to store row and column indices of non-zero entries
    rows = []
    cols = []
    # Iterate through each position
    for i in range(len(positions)):
        # Find indices of positions within LDwindow of positions[i]
        start_idx = np.searchsorted(positions, positions[i] - LDwindow, side='left')
        end_idx = np.searchsorted(positions, positions[i] + LDwindow, side='right')
        # Add the indices to the rows and cols lists
        for j in range(start_idx, end_idx):
            rows.append(i)
            cols.append(j)
    # Create a sparse matrix in COO format
    data = np.ones(len(rows), dtype=bool)  # All non-zero entries are True
    neighbor_matrix = sparse.coo_matrix((data, (rows, cols)),
                                        shape=(len(positions), len(positions)))
    return neighbor_matrix

def compute_corr_matrix(X: np.ndarray, neighbor_matrix: sparse.coo_matrix):
    '''
    Computes the correlation between neighboring pairs of variants. The square of this matrix is the LD matrix.
    Parameters:
        X (2D array): N*M standardized genotype matrix where for every column, the mean is 0 and the variance is 1.
        neighbor_matrix (sparse 2D matrix (COO)): An M*M scipy sparse matrix with boolean values, where True indicates that the correlation between variants i and j is to be computed.
    Returns:
        corr_matrix (sparse 2D matrix (CSR)): M*M correlation matrix between pairs of variants. Returned in CSR sparse format.
    '''
    N, M = X.shape
    # Lists to store row indices, column indices, and data for the sparse matrix
    rows = []
    cols = []
    data = []
    # Iterate over non-zero entries in neighbors_mat
    for i, j in zip(neighbor_matrix.row, neighbor_matrix.col):
        if i < j:  # Only compute for upper triangle (including diagonal)
            # Compute the dot product for (i, j)
            corr_value = X[:, i].dot(X[:, j]) / N
            # Add (i, j) and (j, i) to the sparse matrix
            rows.append(i)
            cols.append(j)
            data.append(corr_value)
            if i != j:  # Avoid duplicating the diagonal
                rows.append(j)
                cols.append(i)
                data.append(corr_value)
    # Add diagonal entries (set to 1)
    for i in range(M):
        rows.append(i)
        cols.append(i)
        data.append(1.0)
    # Create the sparse matrix in COO format
    corr_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(M, M))
    return corr_matrix

def compute_LD_matrix(corr_matrix):
        '''
        Computes LD matrix by just taking the square of a correlation matrix.
        Parameters:
            corr_matrix (sparse 2D matrix (CSR)): M*M correlation matrix between pairs of variants in CSR format. Can be computed with `get_corr_matrix()`.
        Returns:
            LD_matrix (sparse 2D matrix (CSR)): M*M LD matrix (square of correlation) between pairs of variants in CSR format.
        '''
        LD_matrix = corr_matrix.multiply(corr_matrix)  # Element-wise square
        return LD_matrix

"""Compact family-relation helper functions."""

import numpy as np


def initialize_relations(N: int, N1: int = None, parent_source: str = 'past'):
    '''
    Initializes a dictionary of relations for a population of N individuals.
    Parameters:
        N (int): Number of individuals in the population.
        N1 (int): Number of individuals in the parental generation. If not specified, defaults to N.
        parent_source (str): Where compact parent indices point. Use 'past' when they
            index the previous generation object and 'current' when they index rows in
            the current population object itself (e.g. combined populations).
    '''
    if parent_source not in {'past', 'current'}:
        raise ValueError("parent_source must be either 'past' or 'current'.")
    if N1 is None:
        N1 = N
    relations = {
        'parents': np.full((N, 2), -1, dtype=np.int32),
        'parent_N': int(N1),
        'parent_source': parent_source,
        'full_sibs': np.full(N, -1, dtype=np.int32),
        'spouses': np.full(N, -1, dtype=np.int32),
    }
    return relations

def apply_relation_to_values(relations: dict, relation: str, values: np.ndarray, N: int,
                             reduce: str = 'mean') -> np.ndarray:
    '''
    Applies one compact relation mapping to a 1D source-value vector.
    Parameters:
        relations (dict): Dictionary returned by `initialize_relations()` or an equivalent
            compact relation mapping.
        relation (str): Relation to apply. Supported values are 'self' and 'parents'.
        values (1D array): Source values indexed by the rows referenced by `relation`.
            For `relation='self'`, this must have length `N`. For `relation='parents'`,
            this must have length `relations['parent_N']`.
        N (int): Number of individuals in the focal population.
        reduce (str): How to combine multiple relatives for one focal individual.
            Supported values are 'mean' and 'sum'. For `relation='self'`, this is ignored.
    Returns:
        transformed_values (1D array): Relation-transformed values for the focal population.
    '''
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError('values must be a 1D array.')

    if relation in {None, 'self'}:
        if values.shape[0] != N:
            raise ValueError(f"Self relation requires a length-{N} value vector.")
        return values.copy()

    if reduce not in {'mean', 'sum'}:
        raise ValueError("reduce must be either 'mean' or 'sum'.")

    if relation == 'parents':
        parent_ids = np.asarray(relations['parents'], dtype=np.int32)
        if parent_ids.ndim != 2 or parent_ids.shape != (N, 2):
            raise ValueError("Compact 'parents' relation must have shape (N, 2).")

        parent_N = int(relations.get('parent_N', N))
        if values.shape[0] != parent_N:
            raise ValueError(
                f"Parent relation requires a length-{parent_N} source vector, got length {values.shape[0]}."
            )

        transformed = np.zeros(N, dtype=float)
        valid_mask = parent_ids >= 0
        if not np.any(valid_mask):
            return transformed

        parent_values = np.zeros((N, 2), dtype=float)
        parent_values[valid_mask] = values[parent_ids[valid_mask]]
        transformed = parent_values.sum(axis=1)

        if reduce == 'sum':
            return transformed

        counts = valid_mask.sum(axis=1)
        nonzero = counts > 0
        transformed[nonzero] /= counts[nonzero]
        return transformed

    raise ValueError(f"Unsupported relation '{relation}'.")

def get_relation_matrix(relations: dict, relation: str, N: int,
                        dtype: np.dtype = np.uint8) -> np.ndarray:
    '''
    Converts compact relation identifiers into a dense matrix.
    Parameters:
        relations (dict): Dictionary returned by `initialize_relations()` or an equivalent
            compact relation mapping.
        relation (str): Relation to convert. Supported values are 'parents',
            'spouses', and 'full_sibs'.
        N (int): Number of individuals in the focal population.
        dtype (numpy dtype): Output dtype for the matrix.
    Returns:
        relation_matrix (2D array): Dense matrix representation of the requested relation.
    '''
    if relation == 'parents':
        parent_ids = np.asarray(relations['parents'], dtype=np.int32)
        if parent_ids.ndim != 2 or parent_ids.shape[1] != 2:
            raise ValueError("Compact 'parents' relation must have shape (N, 2).")
        N1 = int(relations.get('parent_N', N))
        parent_matrix = np.zeros((N, N1), dtype=dtype)
        valid_mask = parent_ids >= 0
        if np.any(valid_mask):
            rows = np.broadcast_to(np.arange(N, dtype=np.int32)[:, None], parent_ids.shape)[valid_mask]
            cols = parent_ids[valid_mask]
            parent_matrix[rows, cols] = 1
        return parent_matrix

    if relation == 'spouses':
        spouse_ids = np.asarray(relations['spouses'], dtype=np.int32)
        if spouse_ids.ndim != 1 or spouse_ids.shape[0] != N:
            raise ValueError("Compact 'spouses' relation must have shape (N,).")
        spouse_matrix = np.zeros((N, N), dtype=dtype)
        valid_mask = spouse_ids >= 0
        spouse_matrix[np.arange(N, dtype=np.int32)[valid_mask], spouse_ids[valid_mask]] = 1
        return spouse_matrix

    if relation == 'full_sibs':
        family_ids = np.asarray(relations['full_sibs'], dtype=np.int32)
        if family_ids.ndim != 1 or family_ids.shape[0] != N:
            raise ValueError("Compact 'full_sibs' relation must have shape (N,).")
        valid_mask = family_ids >= 0
        sibling_matrix = (
            valid_mask[:, None]
            & valid_mask[None, :]
            & (family_ids[:, None] == family_ids[None, :])
        ).astype(dtype, copy=False)
        np.fill_diagonal(sibling_matrix, 0)
        return sibling_matrix

    raise ValueError(f"Unsupported relation '{relation}'.")

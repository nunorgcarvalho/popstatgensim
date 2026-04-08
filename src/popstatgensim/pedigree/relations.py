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

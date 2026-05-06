"""Generic sample-alignment helpers for real-data workflows."""

from __future__ import annotations

import warnings
from typing import Sequence, Union

import numpy as np


def _normalize_1d_ids(
    ids: Union[np.ndarray, Sequence[object]],
    *,
    name: str,
    allow_duplicates: bool = False,
) -> np.ndarray:
    ids = np.asarray(ids)
    if ids.ndim == 2:
        if ids.shape[1] < 1:
            raise ValueError(f'`{name}` must contain at least one column of IDs.')
        ids = ids[:, -1]
    elif ids.ndim != 1:
        raise ValueError(f'`{name}` must be a 1D array or a 2D FID/IID array.')

    ids = ids.astype(str, copy=False)
    if ids.size == 0:
        raise ValueError(f'`{name}` must contain at least one ID.')
    if np.any(ids == ''):
        raise ValueError(f'`{name}` contains empty IDs.')
    if not allow_duplicates:
        unique_ids, counts = np.unique(ids, return_counts=True)
        duplicated = unique_ids[counts > 1]
        if duplicated.size > 0:
            preview = ', '.join(duplicated[:5])
            raise ValueError(f'`{name}` contains duplicate IIDs, including {preview}.')
    return ids


def _coerce_keep_iids(keep: Union[Sequence[object], np.ndarray, None]) -> np.ndarray | None:
    if keep is None:
        return None
    return _normalize_1d_ids(keep, name='keep', allow_duplicates=False)


def _build_indexer(ids: np.ndarray) -> dict[str, int]:
    return {iid: idx for idx, iid in enumerate(ids.tolist())}


def _apply_keep_filter(
    ids: np.ndarray,
    *,
    keep: np.ndarray | None,
) -> np.ndarray:
    if keep is None:
        return np.ones(ids.shape[0], dtype=bool)
    keep_set = set(keep.tolist())
    return np.array([iid in keep_set for iid in ids], dtype=bool)


def _ids_match_exactly(id_groups: Sequence[np.ndarray]) -> bool:
    first = id_groups[0]
    return all(group.shape == first.shape and np.array_equal(group, first) for group in id_groups[1:])


def _normalize_matrix_list(
    matrices: Union[np.ndarray, Sequence[np.ndarray], None],
    *,
    name: str,
) -> tuple[list[np.ndarray], bool]:
    if matrices is None:
        return [], False
    if isinstance(matrices, np.ndarray):
        return [np.asarray(matrices)], True
    out = [np.asarray(matrix) for matrix in matrices]
    if len(out) == 0:
        raise ValueError(f'`{name}` must contain at least one matrix when provided.')
    return out, False


def _normalize_matrix_id_list(
    ids: Union[np.ndarray, Sequence[np.ndarray], None],
    *,
    n_matrices: int,
    name: str,
) -> list[np.ndarray]:
    if n_matrices == 0:
        return []
    if ids is None:
        raise ValueError(f'`{name}` must be provided when matrices are supplied.')
    if isinstance(ids, np.ndarray):
        return [_normalize_1d_ids(ids, name=name)] * n_matrices
    ids_list = list(ids)
    if len(ids_list) != n_matrices:
        raise ValueError(f'`{name}` must have length {n_matrices}.')
    return [_normalize_1d_ids(group, name=f'{name}[{i}]') for i, group in enumerate(ids_list)]


def subset_grm_by_ids(
    grm: np.ndarray,
    ids: Union[np.ndarray, Sequence[object]],
    target_ids: Union[np.ndarray, Sequence[object]],
) -> np.ndarray:
    """
    Subsets and reorders a square matrix to match a target IID order.
    """
    grm = np.asarray(grm)
    if grm.ndim != 2 or grm.shape[0] != grm.shape[1]:
        raise ValueError('`grm` must be a square 2D array.')
    ids = _normalize_1d_ids(ids, name='ids')
    target_ids = _normalize_1d_ids(target_ids, name='target_ids')
    if grm.shape[0] != ids.shape[0]:
        raise ValueError('`grm` dimension does not match the number of provided IDs.')
    if ids.shape == target_ids.shape and np.array_equal(ids, target_ids):
        return grm
    indexer = _build_indexer(ids)
    missing = [iid for iid in target_ids.tolist() if iid not in indexer]
    if missing:
        preview = ', '.join(missing[:5])
        raise ValueError(f'`target_ids` contains IDs not found in `ids`, including {preview}.')
    order = np.fromiter((indexer[iid] for iid in target_ids.tolist()), dtype=int)
    return grm[np.ix_(order, order)]


def align_samples(
    *,
    y: np.ndarray | None = None,
    y_ids: Union[np.ndarray, Sequence[object], None] = None,
    X: np.ndarray | None = None,
    X_ids: Union[np.ndarray, Sequence[object], None] = None,
    Rs: Union[np.ndarray, Sequence[np.ndarray], None] = None,
    R_ids: Union[np.ndarray, Sequence[np.ndarray], None] = None,
    keep: Union[Sequence[object], np.ndarray, None] = None,
) -> dict[str, np.ndarray | list[np.ndarray] | None]:
    """
    Align phenotypes, covariates, and relationship matrices to a shared IID set.
    """
    R_list, Rs_was_single = _normalize_matrix_list(Rs, name='Rs')
    if y is None and X is None and len(R_list) == 0:
        raise ValueError('At least one of `y`, `X`, or `Rs` must be provided.')

    keep_ids = _coerce_keep_iids(keep)
    id_groups = []

    y_array = None if y is None else np.asarray(y, dtype=float)
    if y_array is not None:
        if y_array.ndim != 1:
            raise ValueError('`y` must be a 1D array.')
        if y_ids is None:
            raise ValueError('`y_ids` must be provided when `y` is supplied.')
        y_ids_norm = _normalize_1d_ids(y_ids, name='y_ids')
        if y_ids_norm.shape[0] != y_array.shape[0]:
            raise ValueError('`y` and `y_ids` must have the same length.')
        id_groups.append(y_ids_norm)
    else:
        y_ids_norm = None

    X_array = None if X is None else np.asarray(X, dtype=float)
    if X_array is not None:
        if X_array.ndim != 2:
            raise ValueError('`X` must be a 2D array.')
        if X_ids is None:
            raise ValueError('`X_ids` must be provided when `X` is supplied.')
        X_ids_norm = _normalize_1d_ids(X_ids, name='X_ids')
        if X_ids_norm.shape[0] != X_array.shape[0]:
            raise ValueError('`X` row count must match `X_ids`.')
        id_groups.append(X_ids_norm)
    else:
        X_ids_norm = None

    R_ids_list = _normalize_matrix_id_list(R_ids, n_matrices=len(R_list), name='R_ids')
    for i, (matrix, ids_i) in enumerate(zip(R_list, R_ids_list)):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f'`Rs[{i}]` must be a square 2D array.')
        if matrix.shape[0] != ids_i.shape[0]:
            raise ValueError(f'`Rs[{i}]` dimension must match `R_ids[{i}]`.')
        id_groups.append(ids_i)

    reference_ids = id_groups[0]
    same_ids = _ids_match_exactly(id_groups)
    if same_ids:
        keep_mask = _apply_keep_filter(reference_ids, keep=keep_ids)
        aligned_ids = reference_ids[keep_mask]
        if keep_ids is not None:
            missing_keep = np.setdiff1d(keep_ids, aligned_ids, assume_unique=False)
            if missing_keep.size > 0:
                warnings.warn(
                    'Some requested `keep` IDs were not found across all aligned inputs: '
                    + ', '.join(missing_keep[:5].tolist()),
                    stacklevel=2,
                )
        out_Rs = [matrix[np.ix_(keep_mask, keep_mask)] for matrix in R_list]
        return {
            'iid': aligned_ids,
            'y': None if y_array is None else y_array[keep_mask],
            'X': None if X_array is None else X_array[keep_mask],
            'Rs': out_Rs[0] if Rs_was_single else out_Rs,
        }

    common_ids = set(reference_ids.tolist())
    for ids_group in id_groups[1:]:
        common_ids &= set(ids_group.tolist())
    if keep_ids is not None:
        common_ids &= set(keep_ids.tolist())

    aligned_ids = np.array([iid for iid in reference_ids.tolist() if iid in common_ids], dtype=str)
    if aligned_ids.size == 0:
        raise ValueError('No shared samples remain after alignment.')

    if keep_ids is not None:
        missing_keep = np.array([iid for iid in keep_ids.tolist() if iid not in common_ids], dtype=str)
        if missing_keep.size > 0:
            warnings.warn(
                'Some requested `keep` IDs were not found across all data inputs: '
                + ', '.join(missing_keep[:5].tolist()),
                stacklevel=2,
            )

    y_out = None
    if y_array is not None:
        if np.array_equal(y_ids_norm, aligned_ids):
            y_out = y_array
        else:
            y_indexer = _build_indexer(y_ids_norm)
            y_out = y_array[np.fromiter((y_indexer[iid] for iid in aligned_ids.tolist()), dtype=int)]

    X_out = None
    if X_array is not None:
        if np.array_equal(X_ids_norm, aligned_ids):
            X_out = X_array
        else:
            X_indexer = _build_indexer(X_ids_norm)
            X_out = X_array[np.fromiter((X_indexer[iid] for iid in aligned_ids.tolist()), dtype=int)]

    R_out = []
    for matrix, ids_i in zip(R_list, R_ids_list):
        if np.array_equal(ids_i, aligned_ids):
            R_out.append(matrix)
            continue
        R_out.append(subset_grm_by_ids(matrix, ids_i, aligned_ids))

    return {
        'iid': aligned_ids,
        'y': y_out,
        'X': X_out,
        'Rs': R_out[0] if Rs_was_single else R_out,
    }


def prepare_reml_inputs(
    *,
    y: np.ndarray,
    Rs: Union[np.ndarray, Sequence[np.ndarray]],
    y_ids: Union[np.ndarray, Sequence[object], None] = None,
    R_ids: Union[np.ndarray, Sequence[np.ndarray], None] = None,
    X: np.ndarray | None = None,
    X_ids: Union[np.ndarray, Sequence[object], None] = None,
    keep: Union[Sequence[object], np.ndarray, None] = None,
) -> dict[str, np.ndarray | list[np.ndarray] | None]:
    """
    Prepare aligned REML inputs from already-loaded arrays and ID vectors.
    """
    return align_samples(
        y=y,
        y_ids=y_ids,
        X=X,
        X_ids=X_ids,
        Rs=Rs,
        R_ids=R_ids,
        keep=keep,
    )


__all__ = [
    "align_samples",
    "prepare_reml_inputs",
    "subset_grm_by_ids",
]

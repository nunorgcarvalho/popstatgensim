"""Import and export helpers for GCTA-compatible files."""

import math
import warnings
from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np

_MISSING_TOKENS = {"", "na", "nan", "n/a", ".", "-9", "-9.0"}


def _infer_square_size_from_lower_triangle(n_values: int) -> int:
    if n_values <= 0:
        raise ValueError(
            'GCTA GRM binary file length is not a valid lower-triangle size.'
        )

    root = math.isqrt(8 * n_values + 1)
    n = (root - 1) // 2
    if n * (n + 1) // 2 != n_values:
        raise ValueError(
            'GCTA GRM binary file length is not a valid lower-triangle size.'
        )
    return n


def _replace_grm_bin_suffix(path: Path, suffix: str) -> Path:
    path_str = str(path)
    if path_str.endswith('.grm.bin'):
        return Path(f'{path_str[:-len(".grm.bin")]}{suffix}')
    return Path(f'{path_str}{suffix}')


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


def _coerce_keep_columns(
    keep: Union[int, Sequence[int], None],
    n_columns: int,
) -> list[int]:
    if keep is None:
        return list(range(1, n_columns + 1))
    if np.isscalar(keep):
        keep = [keep]
    out = []
    seen = set()
    for col in keep:
        if not isinstance(col, (int, np.integer)):
            raise TypeError('`keep` must contain integer column numbers.')
        col = int(col)
        if col < 1 or col > n_columns:
            raise ValueError(
                f'`keep` entries must be between 1 and {n_columns}, received {col}.'
            )
        if col not in seen:
            out.append(col)
            seen.add(col)
    if len(out) == 0:
        raise ValueError('`keep` must select at least one data column.')
    return out


def _coerce_discretize_columns(
    discretize: Union[int, Sequence[int], None],
    *,
    keep_columns: Sequence[int],
    n_columns: int,
) -> set[int]:
    if discretize is None:
        return set()
    if np.isscalar(discretize):
        discretize = [discretize]
    keep_set = set(keep_columns)
    out = set()
    for col in discretize:
        if not isinstance(col, (int, np.integer)):
            raise TypeError('`discretize` must contain integer column numbers.')
        col = int(col)
        if col < 1 or col > n_columns:
            raise ValueError(
                f'`discretize` entries must be between 1 and {n_columns}, received {col}.'
            )
        if col not in keep_set:
            raise ValueError(
                f'`discretize` column {col} was requested but is not present in `keep`.'
            )
        out.add(col)
    return out


def _check_for_missing_tokens(values: np.ndarray, *, name: str) -> None:
    flat = np.char.strip(values.astype(str, copy=False).ravel())
    missing_mask = np.array([token.casefold() in _MISSING_TOKENS for token in flat])
    if np.any(missing_mask):
        raise ValueError(f'`{name}` contains missing values, which are not currently allowed.')


def _read_gcta_table_raw(
    input_path: Union[str, Path],
    *,
    skip_FID: bool = False,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    table = np.loadtxt(input_path, dtype=str, ndmin=2)
    if table.ndim != 2 or table.shape[0] == 0:
        raise ValueError('Input table must contain at least one row.')

    id_cols = 1 if skip_FID else 2
    if table.shape[1] <= id_cols:
        raise ValueError(
            'Input table does not contain any data columns after the ID column(s).'
        )

    fid = None if skip_FID else table[:, 0]
    iid = table[:, 0] if skip_FID else table[:, 1]
    values = table[:, id_cols:]

    _check_for_missing_tokens(iid[:, None], name='IID column')
    if fid is not None:
        _check_for_missing_tokens(fid[:, None], name='FID column')
    _check_for_missing_tokens(values, name='data columns')
    iid = _normalize_1d_ids(iid, name='IID column', allow_duplicates=False)

    return fid, iid, values


def _dummy_encode_categories(
    values: np.ndarray,
    *,
    column_number: int,
) -> tuple[np.ndarray, list[str]]:
    categories = np.unique(values.astype(str, copy=False))
    if categories.size < 2:
        raise ValueError(
            f'Categorical column {column_number} must contain at least two distinct values.'
        )
    baseline = categories[0]
    expanded = []
    labels = []
    for category in categories[1:]:
        expanded.append((values == category).astype(float))
        labels.append(f'V{column_number}={category}')
    return np.column_stack(expanded), labels


def _parse_selected_gcta_columns(
    raw_values: np.ndarray,
    *,
    keep_columns: Sequence[int],
    discretize_columns: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    matrices = []
    labels: list[str] = []
    for col in keep_columns:
        values = raw_values[:, col - 1]
        if col in discretize_columns:
            encoded, encoded_labels = _dummy_encode_categories(values, column_number=col)
            matrices.append(encoded)
            labels.extend(encoded_labels)
            continue
        try:
            numeric = values.astype(float)
        except ValueError as exc:
            raise ValueError(
                f'Continuous column {col} contains non-numeric values and cannot be parsed.'
            ) from exc
        if not np.isfinite(numeric).all():
            raise ValueError(f'Continuous column {col} contains non-finite values.')
        matrices.append(numeric[:, None])
        labels.append(f'V{col}')

    values = np.column_stack(matrices).astype(float, copy=False)
    return values, np.asarray(labels, dtype=str)


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=0)
    sds = values.std(axis=0)
    if np.any(sds <= 0):
        bad = np.flatnonzero(sds <= 0)[0] + 1
        raise ValueError(
            f'Cannot standardize because output column {bad} has zero variance.'
        )
    return (values - means) / sds


def _coerce_keep_iids(keep: Union[Sequence[object], np.ndarray, None]) -> np.ndarray | None:
    if keep is None:
        return None
    keep_ids = _normalize_1d_ids(keep, name='keep', allow_duplicates=False)
    return keep_ids


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


def export_GRM_GCTA(grm: np.ndarray, M: Union[int, float], output_prefix: Union[str, Path]) -> None:
    '''
    Exports a genomic relationship matrix (GRM) to GCTA's binary GRM format.
    Parameters:
        grm (np.ndarray): An N x N genomic relationship matrix.
        M (int | float): Number of SNPs used to calculate the GRM.
            This value is written to the `.grm.N.bin` file for each lower-triangle element.
        output_prefix (str | Path): Output path prefix. GCTA suffixes
            `.grm.bin`, `.grm.N.bin`, and `.grm.id` are appended automatically.
    Returns:
        None
    '''
    grm = np.asarray(grm)
    if grm.ndim != 2 or grm.shape[0] != grm.shape[1]:
        raise ValueError('`grm` must be a square 2D array.')

    if not np.isfinite(grm).all():
        raise ValueError('`grm` must contain only finite values.')

    if not np.isfinite(M):
        raise ValueError('`M` must be finite.')

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    n_individuals = grm.shape[0]

    lower_triangle = grm[np.tril_indices(n_individuals)].astype(np.float32, copy=False)
    n_pairs = lower_triangle.size
    snp_counts = np.full(n_pairs, float(M), dtype=np.float32)

    Path(f'{prefix}.grm.bin').write_bytes(lower_triangle.tobytes())
    Path(f'{prefix}.grm.N.bin').write_bytes(snp_counts.tobytes())

    ids = np.arange(1, n_individuals + 1, dtype=int)
    with Path(f'{prefix}.grm.id').open('w', encoding='utf-8') as handle:
        for sample_id in ids:
            handle.write(f'{sample_id}\t{sample_id}\n')


def read_GRM_GCTA(
    grm_bin_path: Union[str, Path],
    read_N: bool = False,
    read_id: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    '''
    Reads a GCTA-style binary genomic relationship matrix (GRM).
    Parameters:
        grm_bin_path (str | Path): Path to a GCTA `.grm.bin` file. The suffix is
            expected by convention but is not required.
        read_N (bool): If True, also reads the corresponding `.grm.N.bin` file.
        read_id (bool): If True, also reads the corresponding `.grm.id` file.
    Returns:
        np.ndarray | dict: By default, an N x N GRM array. If `read_N` or
            `read_id` is True, returns a dictionary with key `grm` and any
            requested sidecar arrays under keys `N` and `id`.
    '''
    grm_bin_path = Path(grm_bin_path)
    lower_triangle = np.fromfile(grm_bin_path, dtype=np.float32)
    n_individuals = _infer_square_size_from_lower_triangle(lower_triangle.size)

    triangle_indices = np.tril_indices(n_individuals)
    grm = np.empty((n_individuals, n_individuals), dtype=np.float32)
    grm[triangle_indices] = lower_triangle
    grm[(triangle_indices[1], triangle_indices[0])] = lower_triangle

    if not read_N and not read_id:
        return grm

    output = {'grm': grm}
    if read_N:
        n_path = _replace_grm_bin_suffix(grm_bin_path, '.grm.N.bin')
        n_values = np.fromfile(n_path, dtype=np.float32)
        if n_values.size != lower_triangle.size:
            raise ValueError('`.grm.N.bin` length does not match `.grm.bin` length.')
        output['N'] = n_values

    if read_id:
        id_path = _replace_grm_bin_suffix(grm_bin_path, '.grm.id')
        ids = np.loadtxt(id_path, dtype=str, ndmin=2)
        if ids.shape[0] != n_individuals:
            raise ValueError('`.grm.id` row count does not match GRM dimension.')
        output['id'] = ids

    return output


def read_table_GCTA(
    input_path: Union[str, Path],
    *,
    keep: Union[int, Sequence[int], None] = None,
    standardize: bool = False,
    discretize: Union[int, Sequence[int], None] = None,
    skip_FID: bool = False,
) -> Dict[str, np.ndarray | None]:
    '''
    Reads a GCTA-style phenotype/covariate table.
    Parameters:
        input_path (str | Path): Path to a whitespace-delimited file with no header.
        keep (int | sequence[int] | None): 1-based data-column numbers to keep,
            excluding the FID/IID columns. If None, keep all data columns.
        standardize (bool): If True, standardize all returned output columns.
        discretize (int | sequence[int] | None): 1-based data-column numbers to
            treat as categorical variables and dummy-encode.
        skip_FID (bool): If True, assume the first column is IID and there is no FID.
    Returns:
        dict: Dictionary with keys `values`, `iid`, `fid`, `columns`,
            `keep_columns`, and `discretized_columns`.
    '''
    fid, iid, raw_values = _read_gcta_table_raw(input_path, skip_FID=skip_FID)
    keep_columns = _coerce_keep_columns(keep, raw_values.shape[1])
    discretize_columns = _coerce_discretize_columns(
        discretize,
        keep_columns=keep_columns,
        n_columns=raw_values.shape[1],
    )
    values, columns = _parse_selected_gcta_columns(
        raw_values,
        keep_columns=keep_columns,
        discretize_columns=discretize_columns,
    )
    if standardize:
        values = _standardize_columns(values)
    return {
        'values': values,
        'iid': iid,
        'fid': None if fid is None else fid.astype(str, copy=False),
        'columns': columns,
        'keep_columns': np.asarray(keep_columns, dtype=int),
        'discretized_columns': np.asarray(sorted(discretize_columns), dtype=int),
    }


def subset_grm_by_ids(
    grm: np.ndarray,
    ids: Union[np.ndarray, Sequence[object]],
    target_ids: Union[np.ndarray, Sequence[object]],
) -> np.ndarray:
    '''
    Subsets and reorders a GRM to match a target IID order.
    Parameters:
        grm (np.ndarray): Square relationship matrix.
        ids (array-like): Current sample IDs for the GRM.
        target_ids (array-like): Desired IID order.
    Returns:
        np.ndarray: Subsetted GRM in the order of `target_ids`.
    '''
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
) -> Dict[str, np.ndarray | list[np.ndarray] | None]:
    '''
    Aligns phenotypes, covariates, and relationship matrices to a shared IID set.
    Parameters:
        y (np.ndarray | None): Outcome vector.
        y_ids (array-like | None): IIDs for `y`.
        X (np.ndarray | None): Covariate matrix.
        X_ids (array-like | None): IIDs for `X`.
        Rs (np.ndarray | sequence[np.ndarray] | None): One or more square matrices.
        R_ids (array-like | sequence[array-like] | None): IID arrays for `Rs`.
        keep (array-like | None): Optional IIDs to retain.
    Returns:
        dict: Dictionary containing aligned `iid`, `y`, `X`, and `Rs`.
    '''
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
        keep_set = set(keep_ids.tolist())
        common_ids &= keep_set

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
    already_aligned: bool = False,
) -> Dict[str, np.ndarray | list[np.ndarray] | None]:
    '''
    Prepares aligned REML inputs from already-loaded arrays.
    Parameters:
        y (np.ndarray): Outcome vector.
        Rs (np.ndarray | sequence[np.ndarray]): One or more relationship matrices.
        y_ids (array-like | None): Outcome IIDs.
        R_ids (array-like | sequence[array-like] | None): Relationship-matrix IIDs.
        X (np.ndarray | None): Optional covariate matrix.
        X_ids (array-like | None): Covariate IIDs.
        keep (array-like | None): Optional IID filter applied during alignment.
        already_aligned (bool): If True, skip ID-based alignment and trust the inputs.
    Returns:
        dict: Dictionary with `iid`, `y`, `X`, and `Rs`.
    '''
    R_list, Rs_was_single = _normalize_matrix_list(Rs, name='Rs')
    y_array = np.asarray(y, dtype=float)
    X_array = None if X is None else np.asarray(X, dtype=float)

    if already_aligned:
        if y_array.ndim != 1:
            raise ValueError('`y` must be a 1D array.')
        n = y_array.shape[0]
        if X_array is not None and (X_array.ndim != 2 or X_array.shape[0] != n):
            raise ValueError('`X` must be a 2D array with the same number of rows as `y`.')
        for i, matrix in enumerate(R_list):
            if matrix.ndim != 2 or matrix.shape != (n, n):
                raise ValueError(f'`Rs[{i}]` must have shape ({n}, {n}).')
        if y_ids is not None:
            iid = np.asarray(y_ids)
            if iid.ndim == 2:
                iid = iid[:, -1]
            iid = iid.astype(str, copy=False)
        elif X_ids is not None:
            iid = np.asarray(X_ids)
            if iid.ndim == 2:
                iid = iid[:, -1]
            iid = iid.astype(str, copy=False)
        elif R_ids is not None:
            if isinstance(R_ids, np.ndarray):
                iid = np.asarray(R_ids)
                if iid.ndim == 2:
                    iid = iid[:, -1]
                iid = iid.astype(str, copy=False)
            else:
                iid = np.asarray(list(R_ids)[0])
                if iid.ndim == 2:
                    iid = iid[:, -1]
                iid = iid.astype(str, copy=False)
        else:
            iid = np.arange(1, n + 1, dtype=int).astype(str)
        return {
            'iid': iid,
            'y': y_array,
            'X': X_array,
            'Rs': R_list[0] if Rs_was_single else R_list,
        }

    return align_samples(
        y=y_array,
        y_ids=y_ids,
        X=X_array,
        X_ids=X_ids,
        Rs=R_list[0] if Rs_was_single else R_list,
        R_ids=R_ids,
        keep=keep,
    )


def export_trait(
    trait: np.ndarray,
    output_path: Union[str, Path],
    format: str = 'GCTA',
) -> None:
    '''
    Exports a 1D trait array to a phenotype file.
    Parameters:
        trait (np.ndarray): Array of trait values for each individual.
        output_path (str | Path): Path to the output phenotype file.
        format (str): Export format. Currently only `GCTA` is supported.
    Returns:
        None
    '''
    if format != 'GCTA':
        raise ValueError("Only `format='GCTA'` is currently supported.")

    trait = np.asarray(trait)
    if trait.ndim != 1:
        raise ValueError('`trait` must be a 1D array.')

    if not np.isfinite(trait).all():
        raise ValueError('`trait` must contain only finite values.')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ids = np.arange(1, trait.shape[0] + 1, dtype=int)
    with output_path.open('w', encoding='utf-8') as handle:
        for sample_id, phenotype in zip(ids, trait):
            handle.write(f'{sample_id}\t{sample_id}\t{phenotype}\n')

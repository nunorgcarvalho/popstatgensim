"""Import and export helpers for GCTA-compatible files."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np

from .alignment import _normalize_1d_ids

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


def _missing_row_mask(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError('`values` must be a 2D array when checking for missing rows.')
    stripped = np.char.strip(values.astype(str, copy=False))
    flat_mask = np.array([token.casefold() in _MISSING_TOKENS for token in stripped.ravel()])
    return flat_mask.reshape(stripped.shape).any(axis=1)


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


def export_GRM_GCTA(grm: np.ndarray, M: Union[int, float], output_prefix: Union[str, Path]) -> None:
    """
    Export a genomic relationship matrix (GRM) to GCTA's binary GRM format.
    """
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
    """
    Read a GCTA-style binary genomic relationship matrix (GRM).
    """
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
    """
    Read a GCTA-style phenotype/covariate table.
    """
    fid, iid, raw_values = _read_gcta_table_raw(input_path, skip_FID=skip_FID)
    keep_columns = _coerce_keep_columns(keep, raw_values.shape[1])
    discretize_columns = _coerce_discretize_columns(
        discretize,
        keep_columns=keep_columns,
        n_columns=raw_values.shape[1],
    )

    relevant = [iid[:, None], raw_values[:, np.asarray(keep_columns) - 1]]
    if fid is not None:
        relevant.insert(0, fid[:, None])
    missing_mask = _missing_row_mask(np.column_stack(relevant))
    n_removed = int(missing_mask.sum())
    if n_removed > 0:
        warnings.warn(
            f'Removed {n_removed} sample(s) from `{input_path}` because the selected GCTA fields contained missing data.',
            stacklevel=2,
        )
        keep_mask = ~missing_mask
        iid = iid[keep_mask]
        raw_values = raw_values[keep_mask]
        if fid is not None:
            fid = fid[keep_mask]

    iid = _normalize_1d_ids(iid, name='IID column', allow_duplicates=False)
    if iid.size == 0:
        raise ValueError('No samples remain after removing rows with missing selected values.')

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


def export_trait(
    trait: np.ndarray,
    output_path: Union[str, Path],
    format: str = 'GCTA',
) -> None:
    """
    Export a 1D trait array to a phenotype file.
    """
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

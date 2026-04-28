"""Import and export helpers for GCTA-compatible files."""

import math
from pathlib import Path
from typing import Dict, Union

import numpy as np


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

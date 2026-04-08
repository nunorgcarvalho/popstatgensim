"""Export helpers for GCTA-compatible files."""

from pathlib import Path
from typing import Union

import numpy as np


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

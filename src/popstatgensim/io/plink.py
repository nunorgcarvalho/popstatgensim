"""Export helpers for plink2 pfiles (.pgen/.pvar/.psam).

Writes biallelic hardcall genotypes to plink2's pfile format using ``pgenlib``'s
``PgenWriter`` in-process (no intermediate VCF). Genotypes are the ALT-allele dosage: with
``REF`` encoding allele 0 and ``ALT`` encoding allele 1, an integer genotype matrix ``G``
(counts of the allele-1 dosage, as produced by ``genome.make_G``) is written directly.

These functions are deliberately array-based and carry no dependency on ``Population`` so that
file-format code stays separate from the simulation/computation layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

_PGEN_MISSING = -9  # pgenlib's missing-genotype sentinel for append_biallelic


def _require_pgenlib():
    try:
        import pgenlib  # noqa: WPS433 (lazy, optional heavy dependency)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            'Writing .pgen files requires the `pgenlib` package. Install it into the active '
            'environment (it ships with plink2 / pip `Pgenlib`).'
        ) from exc
    return pgenlib


def _chrom_labels_from_starts(chrom_starts: np.ndarray, M: int) -> np.ndarray:
    '''
    Converts ascending chromosome START indices (as returned by
    ``Population.get_chrom_idx``) into a per-variant 1-based chromosome label array of length M.
    '''
    starts = np.asarray(chrom_starts, dtype=int)
    return np.searchsorted(starts, np.arange(M), side='right').astype(int)


def _plink_sex_codes(sex: Optional[np.ndarray], N: int) -> np.ndarray:
    '''
    Maps the popstatgensim sex convention (0 = female, 1 = male) to plink SEX codes
    (1 = male, 2 = female, 0 = unknown). ``None`` yields all-unknown.
    '''
    if sex is None:
        return np.zeros(N, dtype=int)
    sex = np.asarray(sex)
    if sex.shape[0] != N:
        raise ValueError('`sex` must have length N (one entry per individual).')
    codes = np.where(sex == 1, 1, np.where(sex == 0, 2, 0)).astype(int)
    return codes


def _write_pvar(path: Path, chrom: np.ndarray, pos: np.ndarray, variant_ids: Sequence[str],
                ref: np.ndarray, alt: np.ndarray) -> None:
    with path.open('w', encoding='utf-8') as handle:
        handle.write('#CHROM\tPOS\tID\tREF\tALT\n')
        for c, p, vid, r, a in zip(chrom, pos, variant_ids, ref, alt):
            handle.write(f'{c}\t{int(p)}\t{vid}\t{r}\t{a}\n')


def _write_psam(path: Path, iids: Sequence[str], fids: Sequence[str],
                sex_codes: np.ndarray) -> None:
    with path.open('w', encoding='utf-8') as handle:
        handle.write('#FID\tIID\tSEX\n')
        for fid, iid, s in zip(fids, iids, sex_codes):
            handle.write(f'{fid}\t{iid}\t{int(s)}\n')


def _write_pgen(prefix: Path, G_block: np.ndarray) -> None:
    '''
    Writes ``<prefix>.pgen`` from an N x M_block integer genotype matrix (ALT dosage 0..2).
    '''
    pgenlib = _require_pgenlib()
    G_block = np.asarray(G_block)
    if G_block.ndim != 2:
        raise ValueError('`G_block` must be a 2D (N x M) genotype matrix.')
    N, M_block = G_block.shape
    # pgenlib appends one variant at a time as an int8 vector over samples (ALT dosage)
    geno_vm = np.ascontiguousarray(G_block.T, dtype=np.int8)  # M_block x N
    writer = pgenlib.PgenWriter(str(prefix.with_suffix('.pgen')).encode(),
                                sample_ct=N, variant_ct=M_block)
    try:
        for v in range(M_block):
            writer.append_biallelic(geno_vm[v])
    finally:
        writer.close()


def export_pgen_fileset(
    prefix: Union[str, Path],
    G: np.ndarray,
    *,
    iids: Sequence[str],
    variant_ids: Optional[Sequence[str]] = None,
    chrom: Optional[Union[int, Sequence[int]]] = None,
    pos: Optional[Sequence[int]] = None,
    fids: Optional[Sequence[str]] = None,
    sex: Optional[np.ndarray] = None,
    ref: Union[str, Sequence[str]] = 'A',
    alt: Union[str, Sequence[str]] = 'G',
) -> Path:
    '''
    Writes a single plink2 pfile set (``<prefix>.pgen/.pvar/.psam``) from a genotype matrix.
    Parameters:
        prefix (str | Path): Output path prefix (suffixes are added).
        G (2D array): N x M integer genotype matrix (ALT-allele dosage, 0..2 for diploid).
        iids (sequence of str): Length-N sample IIDs (should be globally unique).
        variant_ids (sequence of str): Length-M variant IDs. Defaults to ``chr{c}:{pos}``.
        chrom (int | sequence): Per-variant chromosome label(s). Scalar broadcasts to all M.
            Defaults to 1.
        pos (sequence of int): Length-M positions. Defaults to ``1..M``.
        fids (sequence of str): Length-N family IDs. Defaults to ``iids`` (FID == IID).
        sex (1D array): Length-N sex in popstatgensim convention (0 = female, 1 = male);
            mapped to plink SEX codes. ``None`` writes unknown sex.
        ref, alt (str | sequence): Reference / alternate allele codes. Scalars broadcast.
    Returns:
        Path: the ``.pgen`` path written.
    '''
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    G = np.asarray(G)
    if G.ndim != 2:
        raise ValueError('`G` must be a 2D (N x M) matrix.')
    N, M = G.shape

    iids = list(map(str, iids))
    if len(iids) != N:
        raise ValueError('`iids` must have length N (rows of G).')
    fids = list(map(str, fids)) if fids is not None else list(iids)
    if len(fids) != N:
        raise ValueError('`fids` must have length N.')

    if chrom is None:
        chrom = np.ones(M, dtype=int)
    elif np.isscalar(chrom):
        chrom = np.full(M, int(chrom), dtype=int)
    else:
        chrom = np.asarray(chrom, dtype=int)
        if chrom.shape[0] != M:
            raise ValueError('`chrom` must have length M.')

    if pos is None:
        pos = np.arange(1, M + 1, dtype=int)
    else:
        pos = np.asarray(pos, dtype=int)
        if pos.shape[0] != M:
            raise ValueError('`pos` must have length M.')

    if variant_ids is None:
        variant_ids = [f'chr{c}:{p}' for c, p in zip(chrom, pos)]
    else:
        variant_ids = list(map(str, variant_ids))
        if len(variant_ids) != M:
            raise ValueError('`variant_ids` must have length M.')

    ref = np.full(M, ref) if np.isscalar(ref) else np.asarray(ref)
    alt = np.full(M, alt) if np.isscalar(alt) else np.asarray(alt)
    if ref.shape[0] != M or alt.shape[0] != M:
        raise ValueError('`ref`/`alt` must be scalars or length-M sequences.')

    sex_codes = _plink_sex_codes(sex, N)

    _write_pgen(prefix, G)
    _write_pvar(prefix.with_suffix('.pvar'), chrom, pos, variant_ids, ref, alt)
    _write_psam(prefix.with_suffix('.psam'), iids, fids, sex_codes)
    return prefix.with_suffix('.pgen')


def export_pgen_by_chrom(
    out_dir: Union[str, Path],
    G: np.ndarray,
    chrom: Sequence[int],
    *,
    iids: Sequence[str],
    variant_ids: Optional[Sequence[str]] = None,
    pos: Optional[Sequence[int]] = None,
    fids: Optional[Sequence[str]] = None,
    sex: Optional[np.ndarray] = None,
    ref: Union[str, Sequence[str]] = 'A',
    alt: Union[str, Sequence[str]] = 'G',
    prefix_fmt: str = 'chr{chrom}',
) -> list[Path]:
    '''
    Writes one plink2 pfile set per chromosome from a full genotype matrix.
    Parameters:
        out_dir (str | Path): Directory to write the per-chromosome pfile sets into.
        G (2D array): N x M integer genotype matrix (ALT dosage).
        chrom (sequence of int): Length-M per-variant chromosome labels (1-based).
        iids, variant_ids, pos, fids, sex, ref, alt: as in ``export_pgen_fileset``; the
            per-chromosome slices of the length-M arrays are taken automatically.
        prefix_fmt (str): Filename-prefix template; ``{chrom}`` is substituted per chromosome.
    Returns:
        list[Path]: the ``.pgen`` paths written, in ascending chromosome order.
    '''
    out_dir = Path(out_dir)
    G = np.asarray(G)
    if G.ndim != 2:
        raise ValueError('`G` must be a 2D (N x M) matrix.')
    N, M = G.shape
    chrom = np.asarray(chrom, dtype=int)
    if chrom.shape[0] != M:
        raise ValueError('`chrom` must have length M (columns of G).')

    if pos is None:
        pos = np.arange(1, M + 1, dtype=int)
    else:
        pos = np.asarray(pos, dtype=int)
    if variant_ids is not None:
        variant_ids = np.asarray(list(map(str, variant_ids)))
    ref_arr = np.full(M, ref) if np.isscalar(ref) else np.asarray(ref)
    alt_arr = np.full(M, alt) if np.isscalar(alt) else np.asarray(alt)

    written = []
    for c in np.unique(chrom):
        mask = chrom == c
        prefix = out_dir / prefix_fmt.format(chrom=c)
        written.append(export_pgen_fileset(
            prefix,
            G[:, mask],
            iids=iids,
            variant_ids=None if variant_ids is None else variant_ids[mask],
            chrom=int(c),
            pos=pos[mask],
            fids=fids,
            sex=sex,
            ref=ref_arr[mask],
            alt=alt_arr[mask],
        ))
    return written

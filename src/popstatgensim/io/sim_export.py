"""Exporters for the simulation → h2est validation bundle.

These write the tabular + JSON artifacts of the export contract
(``docs/architecture/sim_export_contract.md``): ground-truth per-pair IBD, phenotype,
per-variant truth tables, and the truth-manifest / meta JSON. They are deliberately array/dict
based (no ``Population``/``Trait`` dependency) so file-format code stays separate from the
simulation layers; a caller extracts the values from the simulated objects and passes them in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

# Relationship classes distinguished by the h2est truth-IBD build (contract §4).
H2EST_REL_CLASSES = ('FS', 'PO', 'deg2', 'deg3', 'fine_deg')
_DEG1_PO = {'parent', 'offspring'}


def map_rel_class(relationship: str, degree: int, *, max_degree: int = 11) -> Optional[str]:
    '''
    Maps a popstatgensim relationship (``RelObj.relationship``) + integer degree onto the coarse
    class the h2est truth-IBD build distinguishes.
    Parameters:
        relationship (str): broad relationship name, e.g. 'sibling', 'parent', 'offspring',
            'uncle/aunt', '1cousin' (see pedigree.relative_types.REL_TYPES).
        degree (int): integer degree of relatedness from path_to_relationship.
        max_degree (int): pairs with degree above this are dropped (return None).
    Returns:
        str | None: one of H2EST_REL_CLASSES, or None if the pair should be omitted
            (self / unrelated / degree > max_degree).
    '''
    if degree is None or degree < 1 or degree > max_degree:
        return None
    if degree == 1:
        # degree-1 relationships are full-sibling, parent, or offspring
        if relationship == 'sibling':
            return 'FS'
        if relationship in _DEG1_PO:
            return 'PO'
        # any other degree-1 label is still a parent/offspring-like first-degree tie
        return 'PO'
    if degree == 2:
        return 'deg2'
    if degree == 3:
        return 'deg3'
    return 'fine_deg'  # 4 <= degree <= max_degree


def export_ibd_pairs(
    out_path,
    relatedness: np.ndarray,
    iids: Sequence[str],
    pair_class_degree: Mapping[tuple, tuple],
    *,
    max_degree: int = 11,
) -> Path:
    '''
    Writes the ground-truth per-pair IBD table ``ibd_pairs.tsv`` (contract §4).
    Parameters:
        out_path: output path for the .tsv.
        relatedness (2D array): N x N realized relatedness (= 2 x kinship), e.g. from
            ``pedigree.ibd.compute_K_IBD``.
        iids (sequence of str): length-N namespaced IIDs, aligned with ``relatedness``.
        pair_class_degree (mapping): ``{(i, j): (rel_class, degree)}`` for pedigree-related index
            pairs (``i < j``). ``rel_class`` is one of H2EST_REL_CLASSES (see ``map_rel_class``);
            entries with a None ``rel_class`` or degree > ``max_degree`` are skipped.
    Returns:
        Path: the written path. Rows are canonicalized so ``ID1 <= ID2`` by Python string compare
            (matching h2est's canon()), one row per retained related pair.
    '''
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    relatedness = np.asarray(relatedness, dtype=float)
    N = len(iids)
    if relatedness.shape != (N, N):
        raise ValueError('`relatedness` must be an (N x N) matrix aligned with `iids`.')
    iids = list(map(str, iids))

    rows = []
    for (i, j), (rel_class, degree) in pair_class_degree.items():
        if rel_class is None or degree is None or degree < 1 or degree > max_degree:
            continue
        if rel_class not in H2EST_REL_CLASSES:
            raise ValueError(f'Unknown rel_class {rel_class!r}; expected one of {H2EST_REL_CLASSES}.')
        id_a, id_b = iids[i], iids[j]
        id1, id2 = (id_a, id_b) if id_a <= id_b else (id_b, id_a)
        rows.append((id1, id2, float(relatedness[i, j]), rel_class, int(degree)))

    # stable, deterministic ordering by (ID1, ID2)
    rows.sort(key=lambda r: (r[0], r[1]))
    with out_path.open('w', encoding='utf-8') as handle:
        handle.write('ID1\tID2\trelatedness\trel_class\tdegree\n')
        for id1, id2, rel, rel_class, degree in rows:
            handle.write(f'{id1}\t{id2}\t{rel:.10g}\t{rel_class}\t{degree}\n')
    return out_path


def export_pheno(out_path, y: Sequence[float], iids: Sequence[str],
                 fids: Optional[Sequence[str]] = None) -> Path:
    '''
    Writes the analyzed-cohort phenotype ``pheno.tsv`` as ``FID IID y`` with a header (contract
    §3); non-finite values are written as ``NA``.
    '''
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=float)
    iids = list(map(str, iids))
    if y.shape[0] != len(iids):
        raise ValueError('`y` and `iids` must have the same length.')
    fids = list(map(str, fids)) if fids is not None else list(iids)
    if len(fids) != len(iids):
        raise ValueError('`fids` must have the same length as `iids`.')
    with out_path.open('w', encoding='utf-8') as handle:
        handle.write('FID\tIID\ty\n')
        for fid, iid, val in zip(fids, iids, y):
            token = f'{val:.10g}' if np.isfinite(val) else 'NA'
            handle.write(f'{fid}\t{iid}\t{token}\n')
    return out_path


def export_variants_base(out_path, ids: Sequence[str], chrom: Sequence[int], pos: Sequence[int],
                         cm: Sequence[float], maf: Sequence[float], typed: Sequence[bool]) -> Path:
    '''
    Writes the population-level per-variant table ``variants.base.tsv`` with columns
    ``id chrom pos cm maf typed`` (contract §2).
    '''
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(map(str, ids))
    M = len(ids)
    cols = {'chrom': chrom, 'pos': pos, 'cm': cm, 'maf': maf, 'typed': typed}
    for name, col in cols.items():
        if len(col) != M:
            raise ValueError(f'`{name}` must have length M (= len(ids)).')
    chrom = np.asarray(chrom, dtype=int)
    pos = np.asarray(pos, dtype=int)
    cm = np.asarray(cm, dtype=float)
    maf = np.asarray(maf, dtype=float)
    typed = np.asarray(typed).astype(int)
    with out_path.open('w', encoding='utf-8') as handle:
        handle.write('id\tchrom\tpos\tcm\tmaf\ttyped\n')
        for k in range(M):
            handle.write(f'{ids[k]}\t{chrom[k]}\t{pos[k]}\t{cm[k]:.6g}\t{maf[k]:.6g}\t{typed[k]}\n')
    return out_path


def export_variants_effects(out_path, ids: Sequence[str], causal: Sequence[bool],
                            beta_std: Sequence[float],
                            beta_par_std: Optional[Sequence[float]] = None) -> Path:
    '''
    Writes the trait-level per-variant effect table ``variants.effects.tsv`` with columns
    ``id causal beta_std [beta_par_std]`` (contract §2). ``beta_par_std`` is omitted if None.
    '''
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(map(str, ids))
    M = len(ids)
    causal = np.asarray(causal).astype(int)
    beta_std = np.asarray(beta_std, dtype=float)
    if len(causal) != M or len(beta_std) != M:
        raise ValueError('`causal` and `beta_std` must have length M (= len(ids)).')
    has_par = beta_par_std is not None
    if has_par:
        beta_par_std = np.asarray(beta_par_std, dtype=float)
        if len(beta_par_std) != M:
            raise ValueError('`beta_par_std` must have length M when provided.')
    with out_path.open('w', encoding='utf-8') as handle:
        header = 'id\tcausal\tbeta_std' + ('\tbeta_par_std\n' if has_par else '\n')
        handle.write(header)
        for k in range(M):
            line = f'{ids[k]}\t{causal[k]}\t{beta_std[k]:.10g}'
            if has_par:
                line += f'\t{beta_par_std[k]:.10g}'
            handle.write(line + '\n')
    return out_path


def export_json(out_path, obj: Mapping[str, Any]) -> Path:
    '''
    Writes a mapping to pretty-printed JSON (used for ``truth_manifest.json`` and ``meta.json``).
    numpy scalars/arrays are converted to native Python types.
    '''
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f'Object of type {type(o).__name__} is not JSON serializable.')

    with out_path.open('w', encoding='utf-8') as handle:
        json.dump(obj, handle, indent=2, default=_default, sort_keys=False)
        handle.write('\n')
    return out_path

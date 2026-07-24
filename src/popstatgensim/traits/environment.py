"""Shared-environment effects keyed by relationship class.

A *shared environment* injects a known environmental covariance among pairs of individuals in a
relationship class (e.g. full sibs share a household). This is exactly the covariance the
class-level analysis matrices (C_FS / C_PO / C_eq2 / C_eq3) are designed to absorb, so it is the
first from-scratch confound for the simulation-validation effort.

Two mechanisms, both returning ordinary `RandomEffect`s:
- **cluster** (exact): individuals sharing a group label (e.g. a sibship) share one environmental
  draw → covariance == variance within the group, 0 across. Use for classes that partition the
  cohort (full sibs). Fast (no dense eigendecomposition).
- **pairs** (kernel): a class whose pairs do NOT form a partition (e.g. 3rd-degree cousins, whose
  neighbourhoods overlap) is injected via the pair-indicator correlation kernel K = I + 1[class],
  projected to the nearest PSD correlation matrix. The realized covariance is class-dominant but
  not perfectly class-pure (the projection smears some covariance to other classes); record the
  realized per-class covariance (see `realized_covariance_by_class`) as the scoring truth.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .effects import RandomEffect
from .random_effects import nearest_correlation_matrix


def _pairs_to_kernel(pairs: Iterable[tuple], N: int) -> np.ndarray:
    '''Build the correlation kernel K = I + 1[(i,j) in pairs], projected to nearest PSD.'''
    K = np.eye(N, dtype=float)
    for (i, j) in pairs:
        K[i, j] = 1.0
        K[j, i] = 1.0
    return nearest_correlation_matrix(K)


def shared_environment_effects(
    class_var: Mapping[str, float],
    N: int,
    *,
    cluster_labels: Optional[Mapping[str, Sequence]] = None,
    class_pairs: Optional[Mapping[str, Iterable[tuple]]] = None,
    name_prefix: str = 'c2_',
    seed: Optional[int] = None,
) -> dict:
    '''
    Build class-keyed shared-environment `RandomEffect`s.
    Parameters:
        class_var (mapping): {class_name: target variance} for each shared-env component.
        N (int): number of individuals in the (analyzed) cohort.
        cluster_labels (mapping): {class_name: length-N group labels} for classes injected as an
            exact categorical cluster effect (e.g. {'FS': cohort.relations['full_sibs']}). Takes
            precedence over `class_pairs` for the same class.
        class_pairs (mapping): {class_name: iterable of (i, j) index pairs} for classes injected
            via the PSD-projected pair-indicator kernel (e.g. deg3 cousin pairs).
        name_prefix (str): effect names are `f'{name_prefix}{class_name}'` (default 'c2_FS' etc.).
        seed (int): base seed; component k uses seed + k for reproducibility.
    Returns:
        dict: {effect_name: RandomEffect}, insertion order matching `class_var`.
    '''
    cluster_labels = dict(cluster_labels or {})
    class_pairs = dict(class_pairs or {})
    effects = {}
    for k, (cls, var) in enumerate(class_var.items()):
        name = f'{name_prefix}{cls}'
        eff_seed = None if seed is None else int(seed) + k
        if cls in cluster_labels:
            labels = np.asarray(cluster_labels[cls])
            if labels.shape[0] != N:
                raise ValueError(f'cluster_labels[{cls!r}] must have length N={N}.')
            eff = RandomEffect(name=name, var=float(var), source=labels, source_kind='array',
                               type='categorical', seed=eff_seed)
        elif cls in class_pairs:
            K = _pairs_to_kernel(class_pairs[cls], N)
            eff = RandomEffect.from_matrix(name=name, var=float(var), K=K, seed=eff_seed)
        else:
            raise ValueError(
                f'Class {cls!r} needs either cluster_labels or class_pairs to define its kernel.')
        effects[name] = eff
    return effects


def realized_covariance_by_class(values: np.ndarray,
                                 pair_class_degree: Mapping[tuple, tuple],
                                 classes: Sequence[str]) -> dict:
    '''
    Empirically measure the realized covariance of a component `values` (length N, mean ~0) within
    each relationship class — i.e. mean over class-c pairs of ``values[i] * values[j]``. This is the
    truth to record for a shared-env component (what covariance was actually injected per class),
    since the PSD projection can smear covariance across classes.
    Parameters:
        values (1D array): the realized environmental component (e.g. Trait.y_['c2_deg3']).
        pair_class_degree (mapping): {(i, j): (rel_class, degree)} from
            io.pair_class_degree_from_pedigree.
        classes (sequence of str): class names to summarize.
    Returns:
        dict: {class_name: mean pairwise product over that class's pairs (None if no such pairs)}.
    '''
    values = np.asarray(values, dtype=float)
    out = {}
    for cls in classes:
        prods = [values[i] * values[j] for (i, j), (c, d) in pair_class_degree.items() if c == cls]
        out[cls] = float(np.mean(prods)) if prods else None
    return out

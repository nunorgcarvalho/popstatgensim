"""Mate-graph construction: non-monogamous assortative mating by matching rounds.

The monogamous model pairs ``N/2`` mothers to ``N/2`` fathers one-to-one, so every pair
of siblings shares both parents and five of the eleven rows of the ``relative_covariance``
writeup's relationship table cannot be generated at all. This module replaces that with a
**mate graph**: an individual may have several mates, so half-siblings, step-siblings and
mate-chain relationships become simulatable.

**Vocabulary** follows the writeup: *mate* / *mates* and *offspring*, never "union" or
"children".

**Parameterization** (settled in ``DECISIONS.md`` D-0005). Non-monogamy is a **survival
array** ``q``, where ``q_k`` is the fraction of individuals with *at least* ``k`` mates::

    q_1 = 1               everyone mates at least once, whether or not they reproduce
    q non-increasing
    kappa = sum_k q_k     mean number of mates per individual
    U = kappa * N / 2     number of mate pairs

``q_k`` is exactly the fraction participating in matching **round k**. A pmf ``p_k`` (the
fraction with *exactly* ``k`` mates) is accepted as an alternative user-facing input and
converted internally via ``q_k = sum_{j>=k} p_j``.

**Why rounds** (D-0006). Each round is a plain monogamous matching, so it reuses the
existing mechanism, and each mate pair independently realizes the target ``rho_y`` as
``Primary-AM`` requires. Crucially, if ``i`` has mates ``j`` and ``k`` they come from
different rounds carrying independent noise, so ``y_j`` and ``y_k`` are conditionally
independent given ``y_i`` -- which makes ``Corr(y_j, y_k) = rho_y^2`` (the writeup's
``\\eqref{eq:chain-y}`` at ``N_mu = 2``) *fall out of the mechanism* rather than being
imposed. That is what makes the mate-chain formulas testable, so it is verified rather
than assumed; see :func:`measure_mate_chain_correlation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Survival array / pmf handling
# ---------------------------------------------------------------------------

def survival_from_pmf(pmf: Sequence[float]) -> np.ndarray:
    """Converts a mate-count pmf into the survival array ``q``.

    ``pmf[k]`` is the fraction of individuals with exactly ``k`` mates (index 0 allowed
    but must be zero -- ``q_1 = 1`` means everyone mates at least once). Returns
    ``q = [q_1, q_2, ...]`` with ``q_k = sum_{j>=k} pmf[j]``.
    """
    pmf = np.asarray(pmf, dtype=float)
    if pmf.ndim != 1 or pmf.size == 0:
        raise ValueError('`pmf` must be a non-empty 1D array.')
    if np.any(pmf < 0):
        raise ValueError('`pmf` must be non-negative.')
    total = pmf.sum()
    if not np.isclose(total, 1.0):
        raise ValueError(f'`pmf` must sum to 1, got {total:.6g}.')
    if pmf.size > 0 and not np.isclose(pmf[0], 0.0):
        raise ValueError(
            f'pmf[0] must be 0: q_1 = 1 means everyone mates at least once (got {pmf[0]:.6g}). '
            'Childlessness is expressed through the offspring distribution f_0, not here.'
        )
    # q_k = sum_{j>=k} pmf[j], for k = 1 ..
    tail = np.cumsum(pmf[::-1])[::-1]
    return np.asarray(tail[1:], dtype=float)


def normalize_survival(q: Optional[Sequence[float]] = None,
                       pmf: Optional[Sequence[float]] = None) -> np.ndarray:
    """Validates and returns the survival array ``q = [q_1, q_2, ...]``.

    Exactly one of ``q`` or ``pmf`` may be given; ``None`` for both means monogamy
    (``q = [1.0]``). Enforces ``q_1 = 1`` and ``q`` non-increasing, and drops a trailing
    run of zeros so ``len(q)`` is the maximum number of mates anyone can have.
    """
    if q is not None and pmf is not None:
        raise ValueError('Provide only one of `q` or `pmf`.')
    if pmf is not None:
        q = survival_from_pmf(pmf)
    if q is None:
        q = [1.0]

    q = np.asarray(q, dtype=float).ravel()
    if q.size == 0:
        raise ValueError('`q` must be non-empty.')
    if not np.isclose(q[0], 1.0):
        raise ValueError(f'q_1 must equal 1 (everyone mates at least once), got {q[0]:.6g}.')
    if np.any(q < 0) or np.any(q > 1):
        raise ValueError('every q_k must lie in [0, 1].')
    if np.any(np.diff(q) > 1e-12):
        raise ValueError('`q` must be non-increasing.')

    nonzero = np.flatnonzero(q > 0)
    return q[: nonzero[-1] + 1].copy()


def mean_n_mates(q: np.ndarray) -> float:
    """``kappa = sum_k q_k``, the mean number of mates per individual."""
    return float(np.sum(q))


def mean_offspring_per_pair(q: np.ndarray) -> float:
    """``m = 2 / kappa``, the only constraint the offspring pmf ``f`` must satisfy (D-0005).

    Follows from ``U * m = N`` with ``U = kappa * N / 2``. Nothing else about ``f``'s
    shape is constrained; ``f_0`` is the childless fraction and needs no separate
    parameter.
    """
    return 2.0 / mean_n_mates(q)


# ---------------------------------------------------------------------------
# Union-find, for the acyclicity (forest) constraint
# ---------------------------------------------------------------------------

class _UnionFind:
    """Union-find with path compression and union by size.

    ``No-mating-loops`` assumes the mate graph is a forest, which is what makes the
    chain between two individuals unique. An edge ``(i, j)`` closes a cycle exactly when
    ``find(i) == find(j)``, at ``O(alpha(N))`` per edge.
    """

    __slots__ = ('parent', 'size')

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, x: int) -> int:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        # path compression
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return int(root)

    def union(self, x: int, y: int) -> bool:
        """Merges the two components. Returns False if they were already joined."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        return True


# ---------------------------------------------------------------------------
# The mate graph
# ---------------------------------------------------------------------------

@dataclass
class MateGraph:
    """The mate edges of one generation.

    Replaces the length-``N`` ``relations['spouses']`` array, which structurally cannot
    represent a second mate.

    Attributes:
        edges: ``(U, 2)`` int32 array of mate pairs as ``(maternal_idx, paternal_idx)``.
        N: Number of individuals in the generation the edges index into.
        round_id: ``(U,)`` int32, which matching round each edge came from.
        n_rounds: How many rounds were run.
        rejections: Per-round count of constraint rejections that needed repair.
        round_rho_y: Per-round realized phenotypic correlation between mates.
    """

    edges: np.ndarray
    N: int
    round_id: np.ndarray
    n_rounds: int = 1
    rejections: Optional[np.ndarray] = None
    round_rho_y: Optional[np.ndarray] = None

    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=np.int32).reshape(-1, 2)
        self.round_id = np.asarray(self.round_id, dtype=np.int32).ravel()
        self._indptr = None
        self._indices = None

    @property
    def U(self) -> int:
        """Number of mate pairs."""
        return int(self.edges.shape[0])

    def _build_index(self):
        """Builds a CSR-style index so `mates_of` is O(degree)."""
        if self._indptr is not None:
            return
        flat_src = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        flat_dst = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
        order = np.argsort(flat_src, kind='stable')
        sorted_src = flat_src[order]
        self._indices = flat_dst[order].astype(np.int32, copy=False)
        counts = np.bincount(sorted_src, minlength=self.N)
        self._indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    def mates_of(self, i: int) -> np.ndarray:
        """Returns the indices of individual ``i``'s mates."""
        self._build_index()
        return self._indices[self._indptr[i]:self._indptr[i + 1]]

    def n_mates(self) -> np.ndarray:
        """Returns a length-``N`` array of each individual's number of mates."""
        self._build_index()
        return np.diff(self._indptr).astype(np.int32, copy=False)

    def has_cycle(self) -> bool:
        """Whether the mate graph contains a cycle (i.e. is not a forest)."""
        return self.count_cycles() > 0

    def count_cycles(self) -> int:
        """Number of independent cycles: ``U - (N - n_components)``.

        Counts only vertices that actually carry an edge, so isolated individuals do
        not inflate the component count.
        """
        uf = _UnionFind(self.N)
        extra = 0
        for i, j in self.edges:
            if not uf.union(int(i), int(j)):
                extra += 1
        return int(extra)

    def to_spouse_array(self) -> np.ndarray:
        """Back-compatible length-``N`` spouse array, keeping only each individual's FIRST mate.

        Lossy by construction -- it is exactly the representation the mate graph exists
        to replace. Provided only so monogamous consumers keep working; anything that
        cares about multiple mates must read :attr:`edges`.
        """
        spouse_idx = np.full(self.N, -1, dtype=np.int32)
        # iterate in reverse so the earliest round wins
        for k in range(self.U - 1, -1, -1):
            i, j = int(self.edges[k, 0]), int(self.edges[k, 1])
            spouse_idx[i] = j
            spouse_idx[j] = i
        return spouse_idx


def _match_one_round(i_females: np.ndarray, i_males: np.ndarray,
                     am_values: np.ndarray, rho_y: float,
                     rescale) -> tuple[np.ndarray, np.ndarray]:
    """Rank-matches one round of females to males at target correlation ``rho_y``.

    Same mechanism as the monogamous ``Population._pair_mates``: sort one sex on the raw
    AM value and the other on a noise-perturbed value. This is a rank-matching copula,
    not a bivariate normal, so the realized correlation lands close to but not exactly
    at ``rho_y``.
    """
    n_pairs = min(len(i_females), len(i_males))
    i_females = i_females[:n_pairs]
    i_males = i_males[:n_pairs]

    if rho_y == 0 or n_pairs == 0:
        return (np.random.permutation(i_females), np.random.permutation(i_males))

    noise = rescale(np.random.normal(size=len(am_values)), 1.0 - rho_y ** 2, 'mate noise')
    mate_values = rho_y * am_values + noise

    ordered_f = i_females[np.argsort(am_values[i_females], kind='stable')]
    ordered_m = i_males[np.argsort(mate_values[i_males], kind='stable')]
    return ordered_f, ordered_m


def build_mate_graph(N: int,
                     sex: np.ndarray,
                     am_values: np.ndarray,
                     rho_y: float = 0.0,
                     q: Optional[Sequence[float]] = None,
                     pmf: Optional[Sequence[float]] = None,
                     forbid_cycles: bool = True,
                     forbid_repeat_mates: bool = True,
                     relative_of=None,
                     rescale=None) -> MateGraph:
    """Builds a mate graph by successive independent matching rounds.

    Round 1 matches all ``N`` individuals into ``N/2`` mate pairs at target correlation
    ``rho_y``. Round ``k`` draws a random sex-balanced fraction ``q_k`` of the population
    and matches them again.

    Constraints (D-0006) are applied **after** the correlation-driven matching rather
    than simultaneously with it, then repaired by swapping mates with a nearby-ranked
    pair. Because the ranks are adjacent the swap barely perturbs the phenotypic
    correlation, so the correlation survives by construction. Rejections are ``~log N``
    out of ``kappa N / 2`` edges, because the mate graph is a union of matchings and its
    cycles behave like random-permutation cycles.

    Parameters:
        N: Population size.
        sex: Length-``N`` array, 0 = female, 1 = male.
        am_values: Length-``N`` values to assort on (already standardized).
        rho_y: Target phenotypic correlation between mates.
        q: Survival array; ``q_k`` = fraction with at least ``k`` mates. Default monogamy.
        pmf: Alternative input -- fraction with exactly ``k`` mates. Converted internally.
        forbid_cycles: Enforce the ``No-mating-loops`` forest constraint.
        forbid_repeat_mates: Forbid the same pair mating in two different rounds.
        relative_of: Optional ``callable(i, j) -> bool`` returning whether ``i`` and
            ``j`` are close blood relatives (``Outbred-mating-chain``). ``None`` disables
            the check.
        rescale: Optional ``callable(values, target_var, name)`` used to rescale the
            noise to an exact empirical variance. Defaults to a local implementation.

    Returns:
        :class:`MateGraph`.
    """
    if rescale is None:
        def rescale(values, target_var, name):
            values = np.asarray(values, dtype=float)
            values = values - values.mean()
            current = values.var()
            if np.isclose(current, 0.0):
                if np.isclose(target_var, 0.0):
                    return values
                raise ValueError(f'Cannot rescale zero-variance {name}.')
            return values * np.sqrt(target_var / current)

    q = normalize_survival(q=q, pmf=pmf)
    sex = np.asarray(sex).ravel()
    am_values = np.asarray(am_values, dtype=float).ravel()
    if sex.shape[0] != N or am_values.shape[0] != N:
        raise ValueError('`sex` and `am_values` must both have length N.')

    all_f = np.flatnonzero(sex == 0).astype(np.int32)
    all_m = np.flatnonzero(sex == 1).astype(np.int32)

    uf = _UnionFind(N) if forbid_cycles else None
    seen_pairs = set() if forbid_repeat_mates else None

    edges = []
    round_ids = []
    rejections = np.zeros(len(q), dtype=np.int32)
    round_rho = np.full(len(q), np.nan, dtype=float)

    for k, q_k in enumerate(q):
        if q_k <= 0:
            continue
        if k == 0:
            f_pool, m_pool = all_f, all_m
        else:
            # a random sex-balanced fraction q_k of the population participates
            n_f = int(round(q_k * len(all_f)))
            n_m = int(round(q_k * len(all_m)))
            n_take = min(n_f, n_m)
            if n_take < 1:
                continue
            f_pool = np.random.choice(all_f, n_take, replace=False)
            m_pool = np.random.choice(all_m, n_take, replace=False)

        ordered_f, ordered_m = _match_one_round(f_pool, m_pool, am_values, rho_y, rescale)

        accepted_f, accepted_m, n_rejected = _apply_constraints(
            ordered_f, ordered_m, uf, seen_pairs, relative_of,
        )
        rejections[k] = n_rejected

        if len(accepted_f) > 0:
            edges.append(np.stack((accepted_f, accepted_m), axis=1))
            round_ids.append(np.full(len(accepted_f), k, dtype=np.int32))
            a = am_values[accepted_f]
            b = am_values[accepted_m]
            if a.std() > 0 and b.std() > 0:
                round_rho[k] = float(np.corrcoef(a, b)[0, 1])

    if edges:
        edge_arr = np.concatenate(edges, axis=0).astype(np.int32, copy=False)
        round_arr = np.concatenate(round_ids).astype(np.int32, copy=False)
    else:
        edge_arr = np.zeros((0, 2), dtype=np.int32)
        round_arr = np.zeros(0, dtype=np.int32)

    return MateGraph(edges=edge_arr, N=N, round_id=round_arr, n_rounds=len(q),
                     rejections=rejections, round_rho_y=round_rho)


def _apply_constraints(ordered_f, ordered_m, uf, seen_pairs, relative_of):
    """Applies the four matching constraints, repairing by nearby-ranked swaps.

    Constraints: no self-pairing, no repeat mate pairs, no mate-graph cycles, no close
    blood relatives. A rejected pair swaps partners with a nearby-ranked pair; because
    ranks are adjacent this barely moves the phenotypic correlation.
    """
    n = len(ordered_f)
    f = np.array(ordered_f, dtype=np.int32, copy=True)
    m = np.array(ordered_m, dtype=np.int32, copy=True)

    def ok(idx_f, idx_m):
        i, j = int(idx_f), int(idx_m)
        if i == j:
            return False
        if seen_pairs is not None and (min(i, j), max(i, j)) in seen_pairs:
            return False
        if relative_of is not None and relative_of(i, j):
            return False
        if uf is not None and uf.find(i) == uf.find(j):
            return False
        return True

    accepted_f, accepted_m = [], []
    n_rejected = 0

    for idx in range(n):
        if ok(f[idx], m[idx]):
            chosen = idx
        else:
            # Repair: swap the male partner with a nearby-ranked pair. Only pairs NOT
            # yet processed are eligible -- swapping with an already-accepted pair would
            # re-use a male who has already been recorded as someone's mate, handing him
            # two mates inside a single round.
            chosen = None
            for cand in range(idx + 1, min(n, idx + 33)):
                if ok(f[idx], m[cand]):
                    m[idx], m[cand] = m[cand], m[idx]
                    chosen = idx
                    break
            n_rejected += 1
            if chosen is None:
                continue  # unrepairable; this individual simply does not mate this round

        i, j = int(f[chosen]), int(m[chosen])
        if uf is not None:
            uf.union(i, j)
        if seen_pairs is not None:
            seen_pairs.add((min(i, j), max(i, j)))
        accepted_f.append(i)
        accepted_m.append(j)

    return (np.array(accepted_f, dtype=np.int32),
            np.array(accepted_m, dtype=np.int32),
            n_rejected)


# ---------------------------------------------------------------------------
# Mechanism-fidelity check
# ---------------------------------------------------------------------------

def measure_mate_chain_correlation(graph: MateGraph, values: np.ndarray) -> dict:
    """Correlation between the values of one individual's two mates.

    The rounds mechanism predicts ``Corr(y_j, y_k) = rho_y(round of j) * rho_y(round of
    k)`` for two mates ``j, k`` of the same individual. Because different rounds carry
    independent noise, ``y_j`` and ``y_k`` are conditionally independent given ``y_i``,
    so this is a **prediction of the mechanism** rather than something imposed.
    Verifying it is the key mechanism-fidelity check (D-0006).

    The nominal target ``rho_y^2`` (the writeup's ``\\eqref{eq:chain-y}`` at
    ``N_mu = 2``) is correct only when every round realizes the SAME ``rho_y``. Round
    ``k > 1`` matches within a pool of ``q_k N / 2`` rather than ``N / 2``, and the
    constraint rejections and nearby-rank repairs concentrate there, so its realized
    correlation can attenuate. Comparing against the nominal square would then charge
    that calibration difference to the coupling. Both targets are returned so the two
    can be told apart:

    * ``corr`` vs ``rho_y_product`` isolates conditional independence -- the mechanism.
    * ``rho_y_product`` vs ``rho_y_squared`` isolates per-round calibration.
    """
    values = np.asarray(values, dtype=float).ravel()
    graph._build_index()

    # per-round realized rho_y, measured rather than assumed equal across rounds
    n_rounds = (int(graph.round_id.max()) + 1) if graph.U > 0 else 1
    round_rho = np.full(max(n_rounds, 1), np.nan, dtype=float)
    for r in range(n_rounds):
        sel = graph.round_id == r
        if int(sel.sum()) < 2:
            continue
        a = values[graph.edges[sel, 0]]
        b = values[graph.edges[sel, 1]]
        if a.std() > 0 and b.std() > 0:
            round_rho[r] = float(np.corrcoef(a, b)[0, 1])

    # each individual's mates, tagged with the round the edge came from
    mates_by_ind = [[] for _ in range(graph.N)]
    for e in range(graph.U):
        i0, j0 = int(graph.edges[e, 0]), int(graph.edges[e, 1])
        r = int(graph.round_id[e])
        mates_by_ind[i0].append((j0, r))
        mates_by_ind[j0].append((i0, r))

    left, right, prod = [], [], []
    for mates in mates_by_ind:
        if len(mates) < 2:
            continue
        for a_i in range(len(mates)):
            for b_i in range(a_i + 1, len(mates)):
                (ja, ra), (jb, rb) = mates[a_i], mates[b_i]
                left.append(ja)
                right.append(jb)
                prod.append(round_rho[ra] * round_rho[rb])

    if graph.U > 1:
        ya = values[graph.edges[:, 0]]
        yb = values[graph.edges[:, 1]]
        rho_y = (float(np.corrcoef(ya, yb)[0, 1])
                 if ya.std() > 0 and yb.std() > 0 else float('nan'))
    else:
        rho_y = float('nan')

    if len(left) < 2:
        return {'corr': float('nan'), 'n_triples': len(left),
                'rho_y': rho_y, 'rho_y_squared': rho_y ** 2,
                'rho_y_product': float('nan'), 'round_rho_y': round_rho}

    a = values[np.array(left, dtype=np.int32)]
    b = values[np.array(right, dtype=np.int32)]
    corr_val = (float(np.corrcoef(a, b)[0, 1])
                if a.std() > 0 and b.std() > 0 else float('nan'))

    return {'corr': corr_val, 'n_triples': len(left),
            'rho_y': rho_y, 'rho_y_squared': rho_y ** 2,
            'rho_y_product': float(np.nanmean(prod)),
            'round_rho_y': round_rho}

# ---------------------------------------------------------------------------
# Stage B -- offspring allocation
# ---------------------------------------------------------------------------

VALID_OFFSPRING_DISTS = ('poisson', 'deterministic', 'negative_binomial',
                         'zero_inflated_poisson')


def offspring_pmf(dist: str, m: float, n_max: int = None, **params) -> np.ndarray:
    """Builds an offspring-per-mate-pair pmf ``f`` with mean exactly ``m``.

    ``m = 2 / kappa`` is the only constraint on ``f`` (D-0005); its shape is otherwise
    free, and ``f_0`` is the childless fraction, which needs no separate parameter.

    Parameters:
        dist: One of ``'poisson'``, ``'deterministic'``, ``'negative_binomial'``,
            ``'zero_inflated_poisson'``. Each is auto-parameterized so its mean is ``m``.
        m: Target mean offspring per mate pair.
        n_max: Truncation point for the unbounded distributions. Defaults to a value
            comfortably beyond the mean.
        **params: ``overdispersion`` (> 1) for ``'negative_binomial'``; ``f_zero`` (the
            childless fraction) for ``'zero_inflated_poisson'``.

    Returns:
        1D array ``f`` with ``f[k]`` the probability of ``k`` offspring, normalized and
        with mean ``m`` to within floating point.
    """
    dist = str(dist).strip().lower()
    if dist not in VALID_OFFSPRING_DISTS:
        raise ValueError(f'`dist` must be one of {list(VALID_OFFSPRING_DISTS)}, got {dist!r}.')
    m = float(m)
    if m <= 0:
        raise ValueError(f'`m` must be positive, got {m}.')
    if n_max is None:
        n_max = max(8, int(np.ceil(m * 6)) + 4)

    k = np.arange(n_max + 1, dtype=float)

    if dist == 'deterministic':
        # counts as equal as possible: mass only on floor(m) and ceil(m)
        lo = int(np.floor(m))
        f = np.zeros(n_max + 1, dtype=float)
        if np.isclose(m, lo):
            f[lo] = 1.0
        else:
            f[lo] = 1.0 - (m - lo)
            f[lo + 1] = m - lo
        return f

    if dist == 'poisson':
        from scipy.stats import poisson as _poisson
        f = _poisson.pmf(k, m)

    elif dist == 'negative_binomial':
        overdispersion = float(params.get('overdispersion', 2.0))
        if overdispersion <= 1.0:
            raise ValueError('`overdispersion` must exceed 1 for a negative binomial.')
        # var = m * overdispersion  ->  r = m / (overdispersion - 1), p = 1/overdispersion
        from scipy.stats import nbinom as _nbinom
        r = m / (overdispersion - 1.0)
        p = 1.0 / overdispersion
        f = _nbinom.pmf(k, r, p)

    else:  # zero_inflated_poisson
        f_zero = float(params.get('f_zero', 0.2))
        if not 0.0 <= f_zero < 1.0:
            raise ValueError('`f_zero` must lie in [0, 1).')
        from scipy.stats import poisson as _poisson
        # inflate zeros but keep the overall mean at m: the Poisson part has mean m/(1-f_zero)
        lam = m / (1.0 - f_zero)
        f = (1.0 - f_zero) * _poisson.pmf(k, lam)
        f[0] += f_zero

    f = np.asarray(f, dtype=float)
    f = f / f.sum()
    # truncation shifts the mean slightly; nudge the tail so the mean is exactly m
    return _retarget_mean(f, m)


def _retarget_mean(f: np.ndarray, m: float) -> np.ndarray:
    """Rescales a truncated pmf so its mean is exactly ``m``, preserving shape."""
    k = np.arange(len(f), dtype=float)
    current = float(np.sum(k * f))
    if np.isclose(current, m, rtol=1e-12, atol=1e-15):
        return f
    # move mass between the low and high halves until the mean matches
    for _ in range(200):
        current = float(np.sum(k * f))
        err = m - current
        if abs(err) < 1e-13:
            break
        # tilt exponentially: f_k *= exp(theta * k), which preserves the family's shape
        var = float(np.sum((k - current) ** 2 * f))
        if var <= 0:
            break
        theta = err / var
        f = f * np.exp(theta * k)
        f = f / f.sum()
    return f


def pmf_mean(f: np.ndarray) -> float:
    """Mean of an offspring pmf."""
    f = np.asarray(f, dtype=float)
    return float(np.sum(np.arange(len(f), dtype=float) * f))


def pmf_dispersion(f: np.ndarray) -> float:
    """``D = (E[n^2] - m) / m^2``, the dispersion that enters the sibship design formula.

    ``D = 1`` for Poisson and ``1 - 1/m`` for deterministic counts.
    """
    f = np.asarray(f, dtype=float)
    k = np.arange(len(f), dtype=float)
    m = float(np.sum(k * f))
    e_k2 = float(np.sum(k ** 2 * f))
    return (e_k2 - m) / m ** 2


def validate_offspring_pmf(f: np.ndarray, kappa: float, tol: float = 1e-6) -> np.ndarray:
    """Validates a user-supplied ``f`` against the single constraint ``mean(f) = 2/kappa``."""
    f = np.asarray(f, dtype=float).ravel()
    if f.ndim != 1 or f.size == 0:
        raise ValueError('`f` must be a non-empty 1D array.')
    if np.any(f < 0):
        raise ValueError('`f` must be non-negative.')
    total = f.sum()
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f'`f` must sum to 1, got {total:.8g}.')
    target = 2.0 / float(kappa)
    mean = pmf_mean(f)
    if abs(mean - target) > tol:
        raise ValueError(
            f'mean(f) must equal 2/kappa = {target:.8g} (D-0005), got {mean:.8g}. '
            'This is the only constraint on f; adjust its shape to hit the mean.'
        )
    return f


def allocate_offspring(U: int, N_offspring: int,
                       f: Optional[np.ndarray] = None,
                       dist: str = 'poisson',
                       weights: Optional[np.ndarray] = None) -> dict:
    """Allocates exactly ``N_offspring`` offspring across ``U`` mate pairs.

    Population size is **deterministic, never random** (D-0005), so both modes return
    counts summing to exactly ``N_offspring``.

    Two modes:

    * **Poisson-shaped** (``f=None, dist='poisson'``): a multinomial allocation of
      exactly ``N_offspring`` across the pairs IS Poisson conditioned on the sum, so the
      exact total comes for free with no repair.
    * **Arbitrary ``f``**: sample iid from ``f``, then repair to exactly
      ``N_offspring`` by incrementing or decrementing randomly chosen pairs. The
      distortion this introduces is measured and returned rather than hidden.

    Parameters:
        U: Number of mate pairs.
        N_offspring: Exact number of offspring to allocate.
        f: Offspring pmf. ``None`` uses ``dist``.
        dist: Named distribution when ``f`` is not given.
        weights: Optional per-pair weight vector, default uniform. Fitness/selection is
            deliberately out of scope (D-0009, ``s = 0``); the existing
            ``W = exp(sum(log1p(G*s)))`` computation is just one possible producer of
            this vector.

    Returns:
        dict with ``counts`` (length-``U`` int array summing to ``N_offspring``),
        ``n_repaired``, ``tv_distortion`` (total-variation distance between the realized
        count distribution and the target ``f``, or 0 for the multinomial mode), and
        ``mode``.
    """
    if U <= 0:
        raise ValueError('`U` must be positive.')
    if N_offspring < 0:
        raise ValueError('`N_offspring` must be non-negative.')

    if weights is None:
        w = np.full(U, 1.0 / U)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != U:
            raise ValueError(f'`weights` must have length U = {U}, got {w.shape[0]}.')
        if np.any(w < 0):
            raise ValueError('`weights` must be non-negative.')
        s = w.sum()
        if s <= 0:
            raise ValueError('`weights` must sum to a positive value.')
        w = w / s

    if f is None and str(dist).strip().lower() == 'poisson':
        counts = np.random.multinomial(N_offspring, w).astype(np.int32, copy=False)
        return {'counts': counts, 'n_repaired': 0, 'tv_distortion': 0.0,
                'mode': 'multinomial(poisson)'}

    m = N_offspring / U
    if f is None:
        f = offspring_pmf(dist, m)
    f = np.asarray(f, dtype=float).ravel()
    f = f / f.sum()

    counts = np.random.choice(len(f), size=U, p=f).astype(np.int64, copy=False)

    # repair to exactly N_offspring
    n_repaired = 0
    diff = int(N_offspring - counts.sum())
    if diff > 0:
        idx = np.random.choice(U, size=diff, replace=True, p=w)
        np.add.at(counts, idx, 1)
        n_repaired = diff
    elif diff < 0:
        need = -diff
        n_repaired = need
        while need > 0:
            eligible = np.flatnonzero(counts > 0)
            if eligible.size == 0:
                raise ValueError('Cannot reduce offspring counts below zero; check f and N.')
            take = min(need, eligible.size)
            idx = np.random.choice(eligible, size=take, replace=False)
            counts[idx] -= 1
            need -= take

    counts = counts.astype(np.int32, copy=False)

    # how far the repair moved the realized distribution away from f
    n_max = max(len(f), int(counts.max()) + 1)
    realized = np.bincount(counts, minlength=n_max).astype(float) / U
    target = np.zeros(n_max, dtype=float)
    target[:len(f)] = f
    tv = 0.5 * float(np.abs(realized - target).sum())

    return {'counts': counts, 'n_repaired': n_repaired, 'tv_distortion': tv,
            'mode': 'iid+repair'}


def predict_halfsib_fullsib_ratio(q: np.ndarray, f: np.ndarray) -> float:
    """Design formula for the half-sib : full-sib pair ratio.

    ``ratio ~= 2 E[K(K-1)] / (kappa D)`` with ``E[K(K-1)] = 2 sum_{k>=2} (k-1) q_k`` and
    ``D`` the dispersion of ``f``. For the two-state case ``q = [1, q_2]`` with Poisson
    offspring this reduces to ``4 q_2 / (1 + q_2)``.

    This is an **approximation**, meant for aiming parameters at a target sibship
    mixture. Verify it empirically rather than trusting it.
    """
    q = np.asarray(q, dtype=float).ravel()
    kappa = float(np.sum(q))
    k_idx = np.arange(1, len(q) + 1, dtype=float)
    e_k_km1 = 2.0 * float(np.sum((k_idx - 1.0) * q))
    D = pmf_dispersion(f)
    if np.isclose(kappa * D, 0.0):
        return float('nan')
    return float(2.0 * e_k_km1 / (kappa * D))

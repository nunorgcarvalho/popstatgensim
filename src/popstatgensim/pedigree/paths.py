"""Paths through the pedigree, and the writeup's closed-form covariance predictor.

This mirrors section 2.3 of the ``relative_covariance`` writeup one-to-one, so theory
and simulation share one vocabulary.

The pedigree is a graph with two edge types: **undirected mating edges** and **directed
meiosis edges** (parent to offspring). The *mating graph* is the mating edges alone and
is a forest under ``No-mating-loops``, which makes the mating chain between any two
individuals unique; the *meiosis graph* is acyclic under ``Discrete-generations``.

* A **lineal chain** is the unique directed path between lineal relatives.
  ``N_lin`` counts its meioses, i.e. how many generations apart its ends are.
* A **mating chain** crosses ``N_mu`` matings.
* A **path** ``P(A,B)`` alternates lineal and mating chains and is itself a **chain
  digraph**: its nodes are chains, its edges join lineal to mating chains, oriented
  *older chain -> younger chain*. Proband ``A`` is a source and ``B`` a sink.
* ``N_o`` is the **out-degree** of a chain node, in ``{0, 1, 2}``. It was called ``N_U``
  in earlier drafts. Lineal chains always have in-degree and out-degree 1.

Per-chain properties (``N_lin``, ``N_mu``) are intrinsic. ``N_o`` and the proband cases
are **junction properties** of the digraph, derived from the arrangement rather than from
any chain alone -- which is why a signature carries them separately.

``N_S`` is deliberately **not implemented**. It is a presentational reparameterization
used in the writeup's prose; ``\\eqref{eq:chain-weight}``, the binomial sum over
``(N_mu, N_o)``, is the single source of truth and reproduces all three
``\\eqref{eq:chain-weight-cases}`` rows on its own, including the boundary that the
``N_S = 0`` row needs a side condition for (D-0008).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Optional, Sequence, Tuple

import numpy as np

#: The three chain-digraph positions a proband can occupy (``\eqref{eq:proband-correction}``).
PROBAND_CASES = ('sink', 'source', 'terminal_mating')


def _round_rho_y(mapping, round_id, default):
    """Looks up a matching round's realized rho_y, falling back to the pooled value."""
    try:
        value = mapping[round_id]
    except (KeyError, IndexError, TypeError):
        return default
    return default if value is None else value


def _is_zero(value) -> bool:
    """Zero test that tolerates symbolic inputs.

    The predictor is deliberately polymorphic: passing sympy symbols for rho_y/rho_g/V_A
    returns the closed form symbolically, which lets it be cross-checked directly against
    the writeup's algebra and against the pathMgr oracle rather than only numerically.
    """
    try:
        return bool(np.isclose(float(value), 0.0))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinealChain:
    """A directed run of meioses between lineal relatives.

    Attributes:
        young_end: Index of the younger individual (the descendant).
        old_end: Index of the older individual (the ancestor).
        N_lin: Number of meioses crossed, i.e. generations apart.
    """

    young_end: int
    old_end: int
    N_lin: int

    def __post_init__(self):
        if self.N_lin < 0:
            raise ValueError(f'N_lin must be non-negative, got {self.N_lin}.')


@dataclass(frozen=True)
class MatingChain:
    """A run of ``N_mu`` mating edges through the mating forest.

    ``N_mu = 0`` means the two ends are the same individual, which carries weight 1.

    Attributes:
        start: Index of the individual the chain starts at.
        end: Index of the individual it ends at.
        N_mu: Number of mating edges crossed.
        members: The individuals along the chain, in order, when known.
        round_ids: The matching round of each mating edge crossed, when known.
            Rounds matter because round ``k > 1`` can realize a slightly different
            ``rho_y`` -- see :func:`chain_weight`'s ``rho_y_per_mating``.
    """

    start: int
    end: int
    N_mu: int
    members: Tuple[int, ...] = ()
    round_ids: Tuple[int, ...] = ()

    def __post_init__(self):
        if self.N_mu < 0:
            raise ValueError(f'N_mu must be non-negative, got {self.N_mu}.')


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathSignature:
    """Everything ``\\eqref{eq:path-product}`` needs, and nothing else.

    A signature is what the predictor consumes, so :func:`predicted_cov` works on **any**
    path whether or not its class has a conventional name.

    Attributes:
        sum_N_lin: ``sum_i N_lin_i`` over the path's lineal chains.
        sum_N_o: ``sum_j N_o_j`` over its mating chains.
        mating_chains: One ``(N_mu, N_o)`` pair per mating chain.
        proband_A_case: Chain-digraph position of proband A.
        proband_B_case: Chain-digraph position of proband B.
        mating_round_ids: Optional per-mating-chain tuple of the matching round each
            mating edge came from, parallel to ``mating_chains``.
    """

    sum_N_lin: int
    sum_N_o: int
    mating_chains: Tuple[Tuple[int, int], ...]
    proband_A_case: str = 'sink'
    proband_B_case: str = 'sink'
    mating_round_ids: Optional[Tuple[Tuple[int, ...], ...]] = None

    def __post_init__(self):
        for case in (self.proband_A_case, self.proband_B_case):
            if case not in PROBAND_CASES:
                raise ValueError(
                    f'proband case must be one of {list(PROBAND_CASES)}, got {case!r}.'
                )
        for (N_mu, N_o) in self.mating_chains:
            if N_mu < 0:
                raise ValueError(f'N_mu must be non-negative, got {N_mu}.')
            if not 0 <= N_o <= 2:
                raise ValueError(f'N_o must be in {{0, 1, 2}}, got {N_o}.')
        if self.sum_N_o != sum(N_o for _, N_o in self.mating_chains):
            raise ValueError(
                f'sum_N_o ({self.sum_N_o}) must equal the sum of per-chain N_o '
                f'({sum(N_o for _, N_o in self.mating_chains)}).'
            )

    @property
    def n_mating_chains(self) -> int:
        return len(self.mating_chains)

    @property
    def key(self) -> tuple:
        """Hashable canonical key, for signature-to-name lookup."""
        return (self.sum_N_lin, self.sum_N_o, tuple(sorted(self.mating_chains)),
                tuple(sorted((self.proband_A_case, self.proband_B_case))))

    def systematic_name(self) -> str:
        """A name derived from the signature, for classes with no conventional one.

        Replaces the old single ``'unspecified'`` catch-all, so every path gets a stable
        label. Example: ``Lin6-Mu[2/0,3/2]-src+snk``.
        """
        chains = ','.join(f'{N_mu}/{N_o}' for N_mu, N_o in self.mating_chains)
        abbrev = {'sink': 'snk', 'source': 'src', 'terminal_mating': 'tmu'}
        ends = '+'.join(sorted((abbrev[self.proband_A_case],
                                abbrev[self.proband_B_case])))
        if not chains:
            return f'Lin{self.sum_N_lin}-Mu[]-{ends}'
        return f'Lin{self.sum_N_lin}-Mu[{chains}]-{ends}'


# ---------------------------------------------------------------------------
# The writeup's closed forms
# ---------------------------------------------------------------------------

def implied_covariance(N_mu: float, rho_y: float, rho_g: float,
                       rho_y_chain: Optional[Sequence[float]] = None) -> float:
    """``c(N_mu)`` from ``\\eqref{eq:chain-weight}``: the implied genetic covariance.

    ``c(0) = 1``; ``c(N_mu) = rho_g * rho_y^(N_mu - 1)`` for ``N_mu >= 1``.

    Parameters:
        N_mu: Number of matings crossed.
        rho_y: Phenotypic correlation between mates.
        rho_g: Genetic correlation between mates.
        rho_y_chain: Optional per-mating ``rho_y`` values. When given, the
            ``rho_y^(N_mu - 1)`` factor becomes the PRODUCT of the individual matings'
            realized correlations. The closed form assumes one common ``rho_y`` per
            mating, which the simulator does not quite deliver: matching round ``k > 1``
            draws from a smaller pool, so its realized ``rho_y`` attenuates slightly.
    """
    N_mu = int(N_mu)
    if N_mu == 0:
        return 1.0
    if rho_y_chain is None:
        return rho_g * rho_y ** (N_mu - 1)
    vals = list(rho_y_chain)
    if len(vals) < N_mu - 1:
        raise ValueError(
            f'rho_y_chain needs at least {N_mu - 1} entries for N_mu={N_mu}, got {len(vals)}.'
        )
    prod = 1.0
    for v in vals[: N_mu - 1]:
        prod = prod * v
    return rho_g * prod


def chain_weight(N_mu: int, N_o: int, rho_y: float, rho_g: float,
                 rho_y_chain: Optional[Sequence[float]] = None) -> float:
    """``w_j`` from ``\\eqref{eq:chain-weight}``.

    ``w_j = 2^-N_o * sum_{n=0}^{N_o} C(N_o, n) c(|N_mu - n|)`` -- the implied genetic
    covariance averaged over the possible entry points into the mating chain from the
    adjacent lineal chains.

    The binomial form is the single source of truth (D-0008). It reproduces all three
    named cases on its own and handles the boundary automatically, so the
    ``N_mu >= N_o + 1`` side condition of the ``N_S = 0`` row disappears::

        (N_mu=1, N_o=2) -> (1 + rho_g) / 2                      full sibs
        (N_mu=2, N_o=2) -> (1 + 2 rho_g + rho_g rho_y) / 4      half sibs
        (N_mu=3, N_o=2) -> rho_g ((1 + rho_y) / 2)^2            step sibs
    """
    if not 0 <= N_o <= 2:
        raise ValueError(f'N_o must be in {{0, 1, 2}}, got {N_o}.')
    if N_mu < 0:
        raise ValueError(f'N_mu must be non-negative, got {N_mu}.')
    total = 0.0
    for n in range(N_o + 1):
        total = total + comb(N_o, n) * implied_covariance(
            abs(N_mu - n), rho_y, rho_g, rho_y_chain=rho_y_chain,
        )
    return total / (2 ** N_o)


def proband_correction(case: str, rho_y: float, rho_g: float) -> float:
    """``C_X`` from ``\\eqref{eq:proband-correction}``.

    Switching a proband from its genetic to its phenotypic value replaces ``rho_g`` by
    ``rho_y`` in that proband's own mating and meiosis edges. The three cases are the
    three chain-digraph positions:

    * ``'sink'`` -- the youngest individual on its lineal chain: ``C = 1``
    * ``'source'`` -- the oldest individual on its lineal chain:
      ``C = (1 + rho_y) / (1 + rho_g)``
    * ``'terminal_mating'`` -- no lineal chain of its own at all:
      ``C = rho_y / rho_g = 1 / h^2``
    """
    if case not in PROBAND_CASES:
        raise ValueError(f'case must be one of {list(PROBAND_CASES)}, got {case!r}.')
    if case == 'sink':
        return 1.0
    if case == 'source':
        denom = 1 + rho_g
        if _is_zero(denom):
            raise ValueError('rho_g = -1 makes the source proband correction singular.')
        return (1 + rho_y) / denom
    if _is_zero(rho_g):
        raise ValueError(
            'rho_g = 0 makes the terminal-mating proband correction (rho_y/rho_g) singular.'
        )
    return rho_y / rho_g


def predicted_cov(signature: PathSignature, V_A: float, rho_y: float, rho_g: float,
                  phenotypic: bool = True, rho_y_by_round=None) -> float:
    """``\\eqref{eq:path-product}`` times ``\\eqref{eq:proband-correction}``.

    Genetic::

        Cov[g_A, g_B] = V_A ((1+rho_g)/2)^(sum N_lin - sum N_o) prod_j w_j

    Phenotypic (``\\eqref{eq:path-product-pheno}``)::

        Cov[y_A, y_B] = C_A C_B Cov[g_A, g_B]

    The ``(1+rho_g)/2`` exponent counts ascents and descents across all lineal chains,
    less the double-counting that occurs where lineal chains ascend into mating chains.

    Works on any signature, named or not -- the predictor never needs a class name.

    Parameters:
        signature: The path's :class:`PathSignature`.
        V_A: Additive variance. Pass the **measured** value (D-0009).
        rho_y: Phenotypic correlation between mates. Measured, not nominal.
        rho_g: Genetic correlation between mates. Measured, not nominal.
        phenotypic: Whether to apply the proband corrections. ``True`` returns
            ``Cov[y_A, y_B]``, ``False`` returns ``Cov[g_A, g_B]``.
        rho_y_by_round: How to treat a chain crossing several matching rounds (D-0014).
            A mapping from round index to that round's REALIZED ``rho_y`` gives the
            per-round product. The string ``'pooled'`` explicitly asks for the writeup's
            literal pooled form. ``None`` **raises** when a chain crosses two or more
            distinct rounds, because pooling them silently is a wrong number rather than
            an error -- there is deliberately no default in the ambiguous case. Single-
            round and monogamous chains are pooled without complaint, since both targets
            coincide exactly there.

    ``V_A``, ``rho_y`` and ``rho_g`` may be sympy symbols, in which case the closed form
    is returned symbolically -- useful for checking against the writeup's algebra and
    the pathMgr oracle directly rather than only at sampled numbers.
    """
    # An (N_mu = 1, N_o = 1) chain describes a path that cannot correspond to a real
    # pedigree: one mating edge feeding one descending lineal chain IS that descent,
    # already counted as a meiosis. `relate` excludes it during enumeration, but a
    # consumer building a signature BY HAND can reach here -- and would get 1.25 where
    # the truth is 0.75, silently. predicted_cov takes a path DESCRIPTION, not a weight,
    # so rejecting an impossible description belongs here. chain_weight(1, 1) is
    # separately well defined and is deliberately left alone.
    for N_mu, N_o in signature.mating_chains:
        if N_mu == 1 and N_o == 1:
            raise ValueError(
                'Signature contains an inadmissible (N_mu=1, N_o=1) mating chain. One '
                'mating edge feeding one descending lineal chain IS that descent, so '
                'the path double-counts the meiosis and the entering proband wrongly '
                'takes the terminal-mating correction 1/h^2 -- inflating the answer by '
                '(1+rho_g)/(h^2(1+rho_y)). The writeup\'s own make_figures.py excludes '
                'this cell ("N_mu = N_U = 1 is a meiosis, absorbed into a lineal chain"). '
                'Use the lineal signature instead: PathSignature(n, 0, ()).'
            )

    exponent = signature.sum_N_lin - signature.sum_N_o
    value = V_A * ((1 + rho_g) / 2) ** exponent

    # `mating_round_ids` holds round INDICES, not correlations. They become rho_y values
    # only once the caller supplies the realized per-round correlations (D-0014).
    per_chain_rounds = signature.mating_round_ids
    explicit_pooled = isinstance(rho_y_by_round, str) and rho_y_by_round == 'pooled'
    mapping = None if explicit_pooled else rho_y_by_round

    for idx, (N_mu, N_o) in enumerate(signature.mating_chains):
        entry = None
        if per_chain_rounds is not None and idx < len(per_chain_rounds):
            entry = per_chain_rounds[idx]

        rho_y_chain = None
        if entry is not None and len(entry) > 0:
            crosses_rounds = len(set(entry)) > 1
            if mapping is None and crosses_rounds and not explicit_pooled:
                raise ValueError(
                    f'Mating chain {idx} crosses matching rounds {sorted(set(entry))}, '
                    'whose realized rho_y can differ (round k > 1 matches in a smaller '
                    'pool and attenuates). Pooling them silently would give a wrong '
                    'NUMBER, not an error -- about 12% by N_mu = 4. Pass '
                    'rho_y_by_round={round: realized_rho_y, ...} for the per-round '
                    "product (D-0014), or rho_y_by_round='pooled' to ask for the "
                    "writeup's literal pooled form deliberately."
                )
            if mapping is not None:
                rho_y_chain = [_round_rho_y(mapping, r, rho_y) for r in entry]

        value = value * chain_weight(N_mu, N_o, rho_y, rho_g, rho_y_chain=rho_y_chain)

    if phenotypic:
        value = value * proband_correction(signature.proband_A_case, rho_y, rho_g)
        value = value * proband_correction(signature.proband_B_case, rho_y, rho_g)
    return value


def predicted_corr(signature: PathSignature, V_A: float, V_Y: float,
                   rho_y: float, rho_g: float, rho_y_by_round=None) -> float:
    """Phenotypic correlation implied by a path -- the quantity the 0.005 floor is in.

    The search floor is stated on the phenotypic correlation rather than on a
    generation count, so the truncation is expressed in the units of the thing being
    approximated.
    """
    if isinstance(V_Y, (int, float)) and V_Y <= 0:
        raise ValueError('V_Y must be positive.')
    return predicted_cov(signature, V_A, rho_y, rho_g, phenotypic=True,
                         rho_y_by_round=rho_y_by_round) / V_Y

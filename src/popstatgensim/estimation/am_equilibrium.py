"""Measured and predicted quantities for the assortative-mating equilibrium.

This module supplies the measurement side of testing the primary-phenotypic-AM
derivations of the ``relative_covariance`` writeup (its Section 2) against simulation.

Two halves, kept separate:

* **Measured** -- :func:`measure_am_quantities` reads realized quantities off a
  simulated :class:`~popstatgensim.simulation.population.Population`: ``V_A``, ``V_Y``,
  ``h2``, the realized mate correlations ``rho_y`` and ``rho_g``, and the segregation
  variance ``Var[eps_o]``.
* **Predicted** -- :func:`predict_rho_g_eq`, :func:`predict_VA_eq`,
  :func:`predict_h2_eq` and :func:`predict_seg_var` evaluate the writeup's closed forms.

Predictions are always evaluated at **measured** inputs, never at the nominal simulation
parameters. The reason is that ``Population._pair_mates`` sorts one sex on the raw AM
value and the other on a noise-perturbed value: that is a rank-matching copula, not a
bivariate normal, so the realized ``rho_y`` lands close to but not exactly at the
requested ``AM_r``, and the realized ``rho_g`` is not exactly ``rho_y * h2``. That
deviation is a quantity worth reporting, not one to hide -- :func:`compare_to_theory`
reports it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from ..traits.standardization import (
    BaseStandardization,
    effective_n_variants,
    get_genetic_effect,
    hardy_weinberg_departure,
    var_A_from_effects,
    var_A_observed,
    var_A_pinned,
)
from ..utils.stats import corr

# re-exported here so the M_e helper is importable alongside the predictions that use it
__all__ = [
    'AMMeasurements', 'compare_to_theory', 'effective_n_variants',
    'measure_am_quantities', 'measure_mate_correlations', 'measure_segregation_variance',
    'hardy_weinberg_departure',
    'predict_VA_eq', 'predict_VA_eq_corrected', 'predict_VA_eq_exact',
    'predict_h2_eq', 'predict_rho_g_eq', 'predict_seg_var', 'predict_seg_var_exact',
]


# ---------------------------------------------------------------------------
# Theory: the writeup's closed forms
# ---------------------------------------------------------------------------

def predict_rho_g_eq(rho_y: float, h2_0: float) -> float:
    """Equilibrium mate genetic correlation -- writeup ``\\eqref{eq:rhog-eq}``.

    ``rho_g_eq = (1 - sqrt(1 - 4 rho_y h2_0 (1 - h2_0))) / (2 (1 - h2_0))``

    The ``-`` root is the admissible one and the discriminant is never negative for
    ``rho_y`` in ``[0, 1]`` and ``h2_0`` in ``[0, 1]``.

    Parameters:
        rho_y: Phenotypic correlation between mates.
        h2_0: Base-population (random-mating) heritability.
    """
    rho_y = float(rho_y)
    h2_0 = float(h2_0)
    if np.isclose(h2_0, 1.0):
        # limit of the quadratic as h2_0 -> 1: the equation degenerates to rho_g = rho_y
        return rho_y
    disc = 1.0 - 4.0 * rho_y * h2_0 * (1.0 - h2_0)
    if disc < 0:
        raise ValueError(
            f'Negative discriminant ({disc:.6g}) in eq:rhog-eq for rho_y={rho_y}, h2_0={h2_0}.'
        )
    return float((1.0 - np.sqrt(disc)) / (2.0 * (1.0 - h2_0)))


def predict_VA_eq(VA_0: float, rho_g: float) -> float:
    """Equilibrium additive variance -- writeup ``\\eqref{eq:VA-eq}``.

    ``V_A_eq = V_A_0 / (1 - rho_g)``
    """
    rho_g = float(rho_g)
    if rho_g >= 1.0:
        raise ValueError(f'rho_g must be < 1 for eq:VA-eq, got {rho_g}.')
    return float(VA_0) / (1.0 - rho_g)


def predict_h2_eq(h2_0: float, rho_g: float) -> float:
    """Equilibrium heritability -- writeup ``\\eqref{eq:hsq-eq}``.

    ``h2_eq = h2_0 / (1 - rho_g (1 - h2_0))``
    """
    h2_0 = float(h2_0)
    rho_g = float(rho_g)
    denom = 1.0 - rho_g * (1.0 - h2_0)
    if np.isclose(denom, 0.0):
        raise ValueError(f'Degenerate denominator in eq:hsq-eq for h2_0={h2_0}, rho_g={rho_g}.')
    return float(h2_0 / denom)


def predict_seg_var(V_A: float, rho_g: float) -> float:
    """Segregation variance -- writeup ``\\eqref{eq:g-seg-var}``.

    ``Var[eps_o] = 0.5 V_A (1 - rho_g)``, the variance of an offspring's genetic value
    around its parents' mean. Under ``\\eqref{eq:VA-eq}`` this equals ``0.5 V_A_0``, so
    the residual genetic variance of a child around the parental mean is unaffected by
    assortative mating. ``V_A`` and ``rho_g`` here are the *equilibrium* values.
    """
    return 0.5 * float(V_A) * (1.0 - float(rho_g))


def predict_VA_eq_exact(VA_0: float, sum_beta2_a: float, rho_g: float) -> float:
    """``\\eqref{eq:VA-eq-exact}`` -- the equilibrium additive variance with NO approximation.

    ``V_A^(eq) = (V_A^(0) - 2 sum_k beta_k^2 a_kk) / (1 - rho_g)``

    ``\\eqref{eq:VA-eq}`` and ``\\eqref{eq:VA-eq-correction}`` both reach a closed form by
    replacing the drag term ``2 sum_k beta_k^2 a_kk`` with an approximation
    (``\\eqref{eq:VA-eq-drag}``, which goes through ``Uniform-inflation``). Feeding the
    MEASURED drag term here instead removes that approximation entirely, which separates
    "the drag term is right but Uniform-inflation mis-evaluates it" from "the drag term
    itself is wrong" -- a distinction no residual against the approximated forms can make.

    Parameters:
        VA_0: Base-generation additive variance.
        sum_beta2_a: Measured ``sum_k beta_k^2 a_kk``, from
            :func:`~popstatgensim.traits.standardization.hardy_weinberg_departure`.
            Note this is the UN-doubled sum; the factor of 2 is applied here.
        rho_g: Measured genetic correlation between mates.
    """
    rho_g = float(rho_g)
    if rho_g >= 1.0:
        raise ValueError(f'rho_g must be < 1 for eq:VA-eq-exact, got {rho_g}.')
    return (float(VA_0) - 2.0 * float(sum_beta2_a)) / (1.0 - rho_g)


def predict_seg_var_exact(VA_0: float, sum_beta2_a: float) -> float:
    """Exact segregation variance: half the numerator of ``\\eqref{eq:VA-eq-exact}``.

    ``Var[eps_o] = (V_A^(0) - 2 sum_k beta_k^2 a_kk) / 2``

    The writeup notes that half that numerator "will come up again as the variance of
    the deviation of offspring genetic values from their parents' average". So this is
    ``\\eqref{eq:g-seg-var}`` with no approximation, and it is a sharper test than
    comparing against ``0.5 V_A (1 - rho_g)``.
    """
    return (float(VA_0) - 2.0 * float(sum_beta2_a)) / 2.0


def predict_VA_eq_corrected(VA_0: float, rho_g: float, M_e: float) -> float:
    """Equilibrium additive variance keeping the term ``\\eqref{eq:VA-eq}`` drops.

    Writeup ``\\eqref{eq:VA-eq-correction}``::

        V_A_eq = V_A_0 / (1 - rho_g) * [1 - rho_g / (2 (1 - rho_g) M_e)]

    ``\\eqref{eq:VA-eq}`` is an *approximation*, and the writeup states the sign of its
    error: dropping the bracketed term **overstates** ``V_A_eq``, because a variant's
    own departure from Hardy-Weinberg drags its contribution down. So the simulation
    should sit predictably BELOW :func:`predict_VA_eq` by a computable amount, and the
    residual should shrink like ``1 / M_e``.

    That makes this a sharper test than "matches within Monte-Carlo error": comparing
    against this corrected form at two or three values of ``M`` tests the approximation
    itself rather than just its tolerance.

    Parameters:
        VA_0: Base-generation additive variance.
        rho_g: Genetic correlation between mates (measured).
        M_e: Effective number of variants, from
            :func:`~popstatgensim.traits.standardization.effective_n_variants`.
    """
    rho_g = float(rho_g)
    M_e = float(M_e)
    if rho_g >= 1.0:
        raise ValueError(f'rho_g must be < 1 for eq:VA-eq-correction, got {rho_g}.')
    if not np.isfinite(M_e) or M_e <= 0:
        raise ValueError(f'M_e must be a positive finite number, got {M_e}.')
    correction = 1.0 - rho_g / (2.0 * (1.0 - rho_g) * M_e)
    return float(VA_0) / (1.0 - rho_g) * correction


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class AMMeasurements:
    r"""Realized quantities read off one simulated generation.

    ``V_A``, ``V_Y`` and ``h2`` describe generation ``t``. ``rho_y`` and ``rho_g``
    describe the **mating that produced** generation ``t`` -- they are measured among
    the parents, in generation ``t - 1`` -- and ``var_eps_o`` is the variance of
    generation ``t``'s genetic values around their own parental means. At equilibrium
    the generations coincide, but the labelling is kept exact so a burn-in run can be
    read honestly.

    ``V_A`` is the **realized** additive variance, ``Var[y_['A']]`` -- the quantity the
    writeup's formulas predict. The three companion fields say where it came from and
    whether anything is off:

    * ``V_A_from_genotypes`` recomputes it from ``pop.G`` in pinned units. It should
      match ``V_A``; a gap means the realized component was rescaled, almost always a
      ``GeneticEffect`` with ``force_var=True``.
    * ``V_A_from_effects`` and ``V_A_from_effects_current_std`` are ``sum(beta^2)``
      under the pinned and the per-generation standardization respectively. The gap
      between *those* two is pure allele-frequency unit drift.

    Two heritabilities are reported and they differ by a covariance term:

    * ``h2`` is ``V_A / V_Y`` with ``V_Y`` measured empirically, so it absorbs the
      realized genetic-environmental covariance.
    * ``h2_cov_free`` is ``V_A / (V_A + V_E)``, which is the writeup's definition under
      ``GE-indep`` -- that assumption says the covariance is zero, so measuring ``V_Y``
      empirically adds it back in. ``h2_cov_free`` is both more faithful to the writeup
      and strictly quieter, and it makes the two boxed forms of ``\eqref{eq:hsq-eq}``
      agree by construction. **Prefer it when feeding the predictions.**

    ``corr_g_e`` reports the realized correlation the assumption sets to zero, so the
    thing being assumed away is visible rather than silently absorbed.

    ``standardization`` names the pinned standardization these were computed against, so
    a pinned figure is never confused with a per-generation one.
    """

    t: int
    trait: str
    component: str
    standardization: str
    V_A: float
    V_Y: float
    h2: float
    rho_y: Optional[float] = None
    rho_g: Optional[float] = None
    var_eps_o: Optional[float] = None
    V_A_from_genotypes: Optional[float] = None
    V_A_from_effects: Optional[float] = None
    V_A_from_effects_current_std: Optional[float] = None
    M_e: Optional[float] = None
    sum_beta4: Optional[float] = None
    sum_beta2_a: Optional[float] = None
    mean_a_kk: Optional[float] = None
    V_E: Optional[float] = None
    h2_cov_free: Optional[float] = None
    corr_g_e: Optional[float] = None
    parent_h2: Optional[float] = None
    parent_V_A: Optional[float] = None
    n_mate_pairs: Optional[int] = None
    N: Optional[int] = None
    drift: Optional[dict] = None

    def as_dict(self) -> dict:
        return asdict(self)


def _mate_pairs(parent_pop) -> tuple[np.ndarray, np.ndarray]:
    """Returns the (i, j) index arrays of each mate pair, each pair listed once.

    Prefers the full mate edge list. The length-N ``spouses`` array keeps only each
    individual's FIRST mate, so under non-monogamy reading it would silently drop every
    later mate pair and bias the measured mate correlations toward round 1.
    """
    edges = parent_pop.relations.get('mate_edges')
    if edges is not None:
        edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        if edges.shape[0] > 0:
            return edges[:, 0], edges[:, 1]

    if 'spouses' not in parent_pop.relations:
        raise ValueError(
            'No spouse relationship stored for the parent generation. '
            '`_pair_mates()` must have run (i.e. the offspring were generated with mating).'
        )
    spouse_idx = np.asarray(parent_pop.relations['spouses'], dtype=np.int32)
    i = np.flatnonzero((spouse_idx >= 0) & (np.arange(parent_pop.N) < spouse_idx))
    return i, spouse_idx[i]


def measure_mate_correlations(parent_pop, trait: str,
                              component: str = 'A') -> dict:
    """Realized phenotypic and genetic correlations between mates.

    Measured directly from the stored spouse pairing rather than assumed from the
    requested ``AM_r``. ``rho_g`` uses the realized genetic values, so it reflects the
    rank-matching pairing actually performed, not ``rho_y * h2``.

    Parameters:
        parent_pop: The generation in which the mating happened (the one holding
            ``relations['spouses']``).
        trait: Name of the trait mated on.
        component: Name of the genetic component, default ``'A'``.
    """
    i, j = _mate_pairs(parent_pop)
    if len(i) == 0:
        raise ValueError('No mate pairs found in the spouse relationship matrix.')

    trait_obj = parent_pop.traits[trait]
    y = np.asarray(trait_obj.y, dtype=float)
    g = np.asarray(trait_obj.y_[component], dtype=float)

    def _corr(v):
        a, b = v[i], v[j]
        if a.std() == 0 or b.std() == 0:
            return float('nan')
        return float(corr(a, b))

    return {'rho_y': _corr(y), 'rho_g': _corr(g), 'n_mate_pairs': int(len(i))}


def measure_segregation_variance(pop, trait: str, component: str = 'A') -> float:
    """Variance of offspring genetic value around the parental mean -- ``Var[eps_o]``.

    Computes ``g_o - 0.5 (g_m + g_p)`` for every offspring in ``pop``'s current
    generation using the realized genetic values of its two parents in ``pop.past[1]``,
    and returns the variance of that residual. Since ``effects_per_allele`` is fixed
    across generations, parental and offspring genetic values are automatically in the
    same units.
    """
    if 'parents' not in pop.relations:
        raise ValueError(
            "No parent relationship stored; `relations['parents']` is required to "
            'measure the segregation variance.'
        )
    parent_pop = _parent_population(pop)
    parents = np.asarray(pop.relations['parents'], dtype=np.int32)
    if parents.shape != (pop.N, 2):
        raise ValueError('Parent index array has incompatible shape for the current generation.')

    g_o = np.asarray(pop.traits[trait].y_[component], dtype=float)
    g_par = np.asarray(parent_pop.traits[trait].y_[component], dtype=float)

    valid = np.all(parents >= 0, axis=1)
    if not np.any(valid):
        raise ValueError('No offspring with two recorded parents.')

    g_m = g_par[parents[valid, 0]]
    g_p = g_par[parents[valid, 1]]
    eps = g_o[valid] - 0.5 * (g_m + g_p)
    return float(np.var(eps))


def _parent_population(pop):
    """Returns the population object holding this generation's parents."""
    parent_source = pop.relations.get('parent_source', 'past')
    if parent_source == 'current':
        return pop
    if getattr(pop, 'past', None) is None or len(pop.past) < 2 or pop.past[1] is None:
        raise ValueError(
            'Previous generation not available. Simulate with keep_past_generations >= 1 '
            'so `pop.past[1]` exists.'
        )
    return pop.past[1]


def measure_am_quantities(pop, trait: str,
                          standardization: Optional[BaseStandardization] = None,
                          component: str = 'A',
                          include_drift: bool = True) -> AMMeasurements:
    """Measures every quantity the equilibrium-tier theory predictions need.

    Parameters:
        pop: The simulated population, at the generation to measure.
        trait: Name of the trait.
        standardization: A pinned :class:`BaseStandardization` to express ``V_A`` in.
            Strongly recommended for multi-generation runs -- without it ``V_A`` is
            just the realized ``Var[y_['A']]`` of this generation, which is fine on its
            own but whose label will say so. Either way the ``standardization`` field
            of the result records which was used.
        component: Name of the genetic component, default ``'A'``.
        include_drift: Whether to attach the allele-frequency drift diagnostic
            (requires ``standardization``).

    Returns:
        :class:`AMMeasurements`.
    """
    trait_obj = pop.traits[trait]
    # touching the effect early gives a clear error if the component is missing/wrong type
    get_genetic_effect(trait_obj, component=component)

    V_Y = float(np.var(np.asarray(trait_obj.y, dtype=float)))
    # The realized genetic values are what the theory predicts, and they are already in
    # fixed units because `effects_per_allele` never changes.
    V_A = var_A_observed(pop, trait, component=component)
    if standardization is not None:
        V_A_geno = var_A_pinned(pop, trait, standardization, component=component)
        std_label = standardization.label
    else:
        V_A_geno = None
        std_label = f'realized@t={int(pop.t)}'

    h2 = float(V_A / V_Y) if V_Y > 0 else float('nan')

    # V_E is exact: y = g + e, so Var[y - g] = Var[e] with no covariance term.
    y_vec = np.asarray(trait_obj.y, dtype=float)
    g_vec = np.asarray(trait_obj.y_[component], dtype=float)
    e_vec = y_vec - g_vec
    V_E = float(np.var(e_vec))
    h2_cov_free = float(V_A / (V_A + V_E)) if (V_A + V_E) > 0 else float('nan')
    corr_g_e = (float(corr(g_vec, e_vec))
                if g_vec.std() > 0 and e_vec.std() > 0 else float('nan'))

    # sum(beta^2) under both standardizations -- the pair that exposes unit drift.
    VA_eff_pinned = (var_A_from_effects(pop, trait, standardization, component=component)
                     if standardization is not None else None)
    VA_eff_current = var_A_from_effects(pop, trait, None, component=component)

    # M_e controls the size of the term eq:VA-eq drops -- see predict_VA_eq_corrected.
    me = effective_n_variants(pop, trait, standardization, component=component)

    # a_kk: the departure from Hardy-Weinberg that eq:VA-eq-exact needs. Measuring it
    # removes the Uniform-inflation approximation from the equilibrium prediction.
    try:
        hw = hardy_weinberg_departure(pop, trait, standardization, component=component)
    except ValueError:
        hw = None

    rho_y = rho_g = var_eps_o = n_pairs = None
    parent_h2 = parent_V_A = None
    if 'parents' in pop.relations:
        try:
            parent_pop = _parent_population(pop)
        except ValueError:
            parent_pop = None
        if parent_pop is not None and parent_pop is not pop:
            try:
                mate = measure_mate_correlations(parent_pop, trait, component=component)
                rho_y, rho_g, n_pairs = mate['rho_y'], mate['rho_g'], mate['n_mate_pairs']
            except ValueError:
                pass
            try:
                var_eps_o = measure_segregation_variance(pop, trait, component=component)
            except ValueError:
                pass
            # parent-generation V_A/h2, in the same units as above
            try:
                parent_V_A = var_A_observed(parent_pop, trait, component=component)
                parent_V_Y = float(np.var(np.asarray(parent_pop.traits[trait].y, dtype=float)))
                parent_h2 = float(parent_V_A / parent_V_Y) if parent_V_Y > 0 else float('nan')
            except (ValueError, KeyError):
                pass

    drift = None
    if include_drift and standardization is not None:
        drift = standardization.allele_freq_drift(pop)

    return AMMeasurements(
        t=int(pop.t),
        trait=str(trait),
        component=str(component),
        standardization=std_label,
        V_A=V_A,
        V_Y=V_Y,
        h2=h2,
        rho_y=rho_y,
        rho_g=rho_g,
        var_eps_o=var_eps_o,
        V_A_from_genotypes=V_A_geno,
        V_A_from_effects=VA_eff_pinned,
        V_A_from_effects_current_std=VA_eff_current,
        M_e=me['M_e'],
        sum_beta4=me['sum_beta4'],
        sum_beta2_a=None if hw is None else hw['sum_beta2_a'],
        mean_a_kk=None if hw is None else hw['mean_a_kk'],
        V_E=V_E,
        h2_cov_free=h2_cov_free,
        corr_g_e=corr_g_e,
        parent_h2=parent_h2,
        parent_V_A=parent_V_A,
        n_mate_pairs=n_pairs,
        N=int(pop.N),
        drift=drift,
    )


def compare_to_theory(measured: AMMeasurements, VA_0: float, h2_0: float,
                      cov_free: bool = True) -> dict:
    r"""Evaluates the four equilibrium predictions at **measured** inputs.

    Each entry reports the predicted value, the measured value, and the relative error
    ``(measured - predicted) / predicted``.

    Parameters:
        measured: Output of :func:`measure_am_quantities`, at equilibrium.
        VA_0: Base-generation additive variance, in the SAME units as
            ``measured.V_A`` -- i.e. measured against the same pinned standardization.
        h2_0: Base-generation heritability. Pass the **covariance-free** value,
            ``V_A^(0) / (V_A^(0) + V_E)`` (``AMMeasurements.h2_cov_free``): the writeup's
            ``GE-indep`` sets the genetic-environmental covariance to zero, so an
            empirically measured ``V_Y^(0)`` adds it back in and injects noise larger
            than the effects being tested.
        cov_free: Whether to compare against ``measured.h2_cov_free`` rather than
            ``measured.h2``, so both sides of ``\eqref{eq:hsq-eq}`` use the same
            definition. Default True.

    Returns:
        A dict keyed by writeup equation label. ``eq:rhog-eq`` is predicted from
        ``rho_y`` and ``h2_0``; the rest are predicted from the **measured** ``rho_g``,
        so that a mismatch in ``eq:rhog-eq`` (which the rank-matching pairing can cause
        on its own) does not propagate into the other three and mask their accuracy.
    """
    out = {}

    def _entry(predicted, measured_value):
        if predicted is None or measured_value is None or not np.isfinite(predicted):
            return {'predicted': predicted, 'measured': measured_value, 'rel_err': None}
        rel = (measured_value - predicted) / predicted if predicted != 0 else None
        return {'predicted': float(predicted),
                'measured': float(measured_value),
                'rel_err': None if rel is None else float(rel)}

    rho_y = measured.rho_y
    rho_g = measured.rho_g

    if rho_y is not None and np.isfinite(rho_y):
        out['eq:rhog-eq'] = _entry(predict_rho_g_eq(rho_y, h2_0), rho_g)
    else:
        out['eq:rhog-eq'] = _entry(None, rho_g)

    if rho_g is not None and np.isfinite(rho_g) and rho_g < 1.0:
        out['eq:VA-eq'] = _entry(predict_VA_eq(VA_0, rho_g), measured.V_A)
        if measured.M_e is not None and np.isfinite(measured.M_e) and measured.M_e > 0:
            out['eq:VA-eq-correction'] = _entry(
                predict_VA_eq_corrected(VA_0, rho_g, measured.M_e), measured.V_A)
        measured_h2 = measured.h2_cov_free if (cov_free and measured.h2_cov_free is not None) else measured.h2
        out['eq:hsq-eq'] = _entry(predict_h2_eq(h2_0, rho_g), measured_h2)
        out['eq:g-seg-var'] = _entry(predict_seg_var(measured.V_A, rho_g), measured.var_eps_o)
    else:
        for key in ('eq:VA-eq', 'eq:VA-eq-correction', 'eq:hsq-eq', 'eq:g-seg-var'):
            out[key] = _entry(None, None)

    return out

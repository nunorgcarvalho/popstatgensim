"""Pinned base-generation genotype standardization.

By default a :class:`~popstatgensim.traits.effects.GeneticEffect` refreshes its
``effects_standardized`` from the *current* generation's genotype standard deviations
every generation (see ``GeneticEffect.update_G_std``). That is harmless for the
simulation itself -- realized genetic values are computed from the fixed
``effects_per_allele`` -- but it means any quantity read off ``effects_standardized``,
or off a genotype matrix restandardized on the current generation, silently changes
units from one generation to the next.

Theory that holds allele frequencies fixed (the ``Constant-freq`` assumption of the
``relative_covariance`` writeup, which standardizes by a fixed ``sqrt(P p (1 - p))``
taken from the base population) needs the *same* units in every generation. This module
pins a base generation's standardization so that quantities such as ``V_A^(0)`` and
``V_A^(eq)`` are directly comparable.

Every quantity returned here carries the standardization it was computed against, so a
pinned number is never mistaken for a per-generation one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

VALID_STD_METHODS = ('observed', 'binomial')


@dataclass(frozen=True)
class BaseStandardization:
    """A genotype standardization pinned to one (base) generation.

    Attributes:
        p: Length-``M`` allele frequencies of the base generation.
        G_std: Length-``M`` genotype standard deviations to divide by. These are
            *fixed* -- they are not refreshed as the population evolves.
        P: Ploidy of the genotypes the standardization was built from.
        t_base: The generation index (``Population.t``) this was pinned at.
        method: How ``G_std`` was derived -- ``'observed'`` (the base generation's
            empirical column SDs, matching ``get_G_std_for_effects``) or
            ``'binomial'`` (``sqrt(P p (1 - p))``, the writeup's literal
            ``sqrt(2 p (1 - p))`` under Hardy-Weinberg).
    """

    p: np.ndarray
    G_std: np.ndarray
    P: int
    t_base: int
    method: str

    @property
    def M(self) -> int:
        return int(self.p.shape[0])

    @property
    def label(self) -> str:
        """Short human-readable tag naming this standardization."""
        return f'pinned[{self.method}]@t={self.t_base}'

    @classmethod
    def from_population(cls, pop, method: str = 'observed',
                        p_min: float = 0.05) -> 'BaseStandardization':
        """Pins the standardization of ``pop``'s current generation.

        Parameters:
            pop: A :class:`~popstatgensim.simulation.population.Population`. Its
                *current* generation becomes the base generation.
            method: ``'observed'`` (default) uses the base generation's empirical
                genotype SDs -- this is what ``GeneticEffect`` used when it first
                converted standardized effects into per-allele effects, so it
                reproduces the original ``effects_standardized`` exactly.
                ``'binomial'`` uses ``sqrt(P p (1 - p))``, the writeup's form.
            p_min: Allele frequency assumed for variants that are monomorphic in the
                base generation, so their SD is never zero. Matches the convention in
                ``effect_sampling.get_G_std_for_effects``.
        """
        method = str(method).strip().lower()
        if method not in VALID_STD_METHODS:
            raise ValueError(f'`method` must be one of {list(VALID_STD_METHODS)}, got {method!r}.')

        G = np.asarray(pop.G)
        P = int(pop.P)
        p = np.asarray(pop.p, dtype=float).copy()

        if method == 'observed':
            G_std = G.std(axis=0).astype(float)
        else:
            G_std = np.sqrt(P * p * (1.0 - p))

        # monomorphic variants would otherwise divide by zero
        G_std = np.where(G_std == 0, np.sqrt(P * p_min * (1.0 - p_min)), G_std)

        return cls(p=p, G_std=G_std, P=P, t_base=int(pop.t), method=method)

    def _check_M(self, M: int):
        if M != self.M:
            raise ValueError(
                f'Expected {self.M} variants to match this standardization '
                f'({self.label}), got {M}.'
            )

    def standardize(self, G: np.ndarray) -> np.ndarray:
        """Standardizes a genotype matrix against the pinned base generation.

        Centers on the *base* allele frequencies and divides by the *base* SDs, so
        columns are only mean-0/variance-1 in the base generation itself. Any drift
        away from that is real signal, not an artifact -- it is exactly what
        :meth:`allele_freq_drift` reports.
        """
        G = np.asarray(G, dtype=float)
        if G.ndim != 2:
            raise ValueError('`G` must be a 2D array.')
        self._check_M(G.shape[1])
        return (G - self.P * self.p[None, :]) / self.G_std[None, :]

    def to_standardized_effects(self, effects_per_allele: np.ndarray) -> np.ndarray:
        """Converts fixed per-allele effects into per-standardized-allele effects in pinned units."""
        effects_per_allele = np.asarray(effects_per_allele, dtype=float)
        if effects_per_allele.ndim != 1:
            raise ValueError('`effects_per_allele` must be a 1D array.')
        self._check_M(effects_per_allele.shape[0])
        return effects_per_allele * self.G_std

    def allele_freq_drift(self, pop) -> dict:
        """Reports how far allele frequencies have drifted from the base generation.

        This is a *diagnostic*, not a correction: the ``Constant-freq`` assumption
        bundles the large-``N_e`` limit, and the intended use is to document that
        assuming no drift is reasonable at the ``N`` and generation counts in use.

        Returns a dict with the mean/max absolute and relative frequency change, the
        implied mean/max relative change in each variant's genotype SD (the quantity
        that actually propagates into ``V_A``), and how many variants have gone
        monomorphic since the base generation.

        Two different quantities are reported and they are NOT interchangeable:

        * ``mean_rel_dsd`` is an unsigned mean of ``|dSD|`` -- drift **magnitude**.
        * ``rel_dhet`` is the **signed** relative change in mean heterozygosity,
          ``mean(2 p_t (1-p_t)) / mean(2 p_0 (1-p_0)) - 1``.

        Use ``rel_dhet`` for anything testing the neutral prediction
        ``E[2 p_t (1-p_t)] = 2 p_0 (1-p_0) (1 - F)`` with ``F = t / (2N)``. The two
        scale differently -- ``|dp|`` accumulates like ``sqrt(t/(2N))`` while
        heterozygosity declines like ``t/(2N)`` -- so fitting the unsigned magnitude
        against the signed prediction gives a confident wrong slope.
        """
        p_now = np.asarray(pop.p, dtype=float)
        self._check_M(p_now.shape[0])

        d_abs = np.abs(p_now - self.p)
        # relative change guarded against base frequencies of exactly 0
        with np.errstate(divide='ignore', invalid='ignore'):
            d_rel = np.where(self.p > 0, d_abs / self.p, np.nan)

        sd_now = np.sqrt(self.P * p_now * (1.0 - p_now))
        sd_base = np.sqrt(self.P * self.p * (1.0 - self.p))
        with np.errstate(divide='ignore', invalid='ignore'):
            sd_rel = np.where(sd_base > 0, np.abs(sd_now - sd_base) / sd_base, np.nan)

        # SIGNED heterozygosity change. `mean_rel_dsd` above is an unsigned mean of
        # |dSD|, i.e. drift MAGNITUDE, and the two scale differently: |dp| accumulates
        # like sqrt(t/(2N)) while E[2p(1-p)] declines like t/(2N). Any test of the
        # neutral prediction E[2p_t(1-p_t)] = 2p_0(1-p_0)(1-F) with F = t/(2N) needs
        # this one -- regressing an unsigned magnitude against a signed prediction
        # yields a confident wrong slope.
        het_now = float(np.mean(2.0 * p_now * (1.0 - p_now)))
        het_base = float(np.mean(2.0 * self.p * (1.0 - self.p)))
        rel_het = (het_now / het_base - 1.0) if het_base > 0 else float('nan')

        return {
            'rel_dhet': rel_het,
            'mean_het_now': het_now,
            'mean_het_base': het_base,
            'standardization': self.label,
            't_base': self.t_base,
            't_now': int(pop.t),
            'generations_elapsed': int(pop.t) - self.t_base,
            'mean_abs_dp': float(d_abs.mean()),
            'max_abs_dp': float(d_abs.max()),
            'mean_rel_dp': float(np.nanmean(d_rel)),
            'max_rel_dp': float(np.nanmax(d_rel)),
            'mean_rel_dsd': float(np.nanmean(sd_rel)),
            'max_rel_dsd': float(np.nanmax(sd_rel)),
            'n_lost': int(np.sum((p_now == 0) | (p_now == 1))),
            'n_lost_base': int(np.sum((self.p == 0) | (self.p == 1))),
            'M': self.M,
        }


def get_genetic_effect(trait, component: str = 'A'):
    """Returns the :class:`GeneticEffect` named ``component`` from ``trait``.

    Raises a clear error rather than a ``KeyError`` when the trait has no such
    genetic component.
    """
    from .effects import GeneticEffect

    effect = trait.effects.get(component) if hasattr(trait, 'effects') else None
    if effect is None:
        raise ValueError(
            f"Trait '{getattr(trait, 'name', trait)}' has no effect named '{component}'."
        )
    if not isinstance(effect, GeneticEffect):
        raise ValueError(
            f"Effect '{component}' of trait '{getattr(trait, 'name', trait)}' is a "
            f'{type(effect).__name__}, not a GeneticEffect.'
        )
    return effect


def var_A_pinned(pop, trait: str, standardization: BaseStandardization,
                 component: str = 'A') -> float:
    """Additive variance recomputed from genotypes in pinned base-generation units.

    Evaluates ``Var[X_pinned @ beta_pinned]``, i.e. the additive variance the current
    genotypes and the fixed per-allele effects imply. Because ``effects_per_allele`` is
    held fixed across generations this is already in stable units; the pinned
    standardization makes those units explicit rather than changing the number.

    **This is a cross-check, not the headline V_A.** It recomputes from ``pop.G`` and so
    ignores any rescaling applied to the *realized* component -- in particular a
    ``GeneticEffect`` with ``force_var=True`` rescales ``y_['A']`` back to ``var_indep``
    every generation, which this function does not see. A material gap between this and
    :func:`var_A_observed` is therefore a symptom worth reading, usually that footgun.
    The quantity the writeup's theory predicts is the realized one,
    :func:`var_A_observed`.
    """
    trait_obj = pop.traits[trait]
    effect = get_genetic_effect(trait_obj, component=component)
    if effect.effects_per_allele is None:
        raise ValueError(f"Effect '{component}' has no per-allele effects stored.")
    X = standardization.standardize(pop.G)
    beta_std = standardization.to_standardized_effects(effect.effects_per_allele)
    return float(np.var(X @ beta_std))


def var_A_observed(pop, trait: str, component: str = 'A') -> float:
    """Additive variance of the realized genetic values in the current generation.

    ``Var[y_['A']]`` -- what the simulation actually realized and what drives both the
    phenotype and the mating, so this is the ``V_A`` the writeup's equilibrium formulas
    predict. Compare against :func:`var_A_pinned` to detect a ``force_var`` rescale.
    """
    trait_obj = pop.traits[trait]
    if component not in trait_obj.y_:
        raise ValueError(
            f"Trait '{trait}' has no realized component '{component}'."
        )
    return float(np.var(np.asarray(trait_obj.y_[component], dtype=float)))


def var_A_from_effects(pop, trait: str,
                       standardization: Optional[BaseStandardization] = None,
                       component: str = 'A') -> float:
    """Additive variance implied by the effect sizes alone: ``sum_k beta_k^2``.

    This is the "independent variants" proxy for ``V_A`` -- exact only when the causal
    standardized genotypes are uncorrelated and have unit variance, which is true in
    the base generation but not after assortative mating has induced correlations
    between causal variants.

    It is the quantity most exposed to the standardization drift this module exists to
    fix. With ``standardization=None`` it reads the effect's own
    ``effects_standardized``, which ``GeneticEffect.update_G_std()`` recomputes from the
    **current** generation's genotype SDs -- so it silently changes units every
    generation. With a pinned ``standardization`` it is held in fixed base-generation
    units and reproduces the base value exactly.

    Compare the two to see how much of an apparent change in ``V_A`` is unit drift
    rather than real inflation. Neither is the truth: that is
    :func:`var_A_pinned` / :func:`var_A_observed`.
    """
    effect = get_genetic_effect(pop.traits[trait], component=component)
    if standardization is None:
        if effect.effects_standardized is None:
            raise ValueError(f"Effect '{component}' has no standardized effects stored.")
        beta_std = np.asarray(effect.effects_standardized, dtype=float)
    else:
        if effect.effects_per_allele is None:
            raise ValueError(f"Effect '{component}' has no per-allele effects stored.")
        beta_std = standardization.to_standardized_effects(effect.effects_per_allele)
    return float(np.sum(beta_std ** 2))


def effective_n_variants(pop, trait: str,
                         standardization: Optional[BaseStandardization] = None,
                         component: str = 'A') -> dict:
    """Effective number of variants ``M_e`` -- writeup ``\\eqref{eq:Me-def}``.

    ``M_e := (sum_k beta_k^2)^2 / sum_k beta_k^4``, normalized so that ``M_e = M`` when
    every variant has the same effect size. It controls the size of the neglected term
    in ``\\eqref{eq:VA-eq}``, so it is what turns that approximation into a testable
    prediction rather than a tolerance check.

    ``M_e`` is invariant to a global rescaling of the effects but NOT to the
    per-variant rescaling that standardization drift applies, so pass the pinned
    ``standardization`` to keep it stable across generations.

    Returns a dict with ``M_e``, ``sum_beta2``, ``sum_beta4`` and the standardization
    label used.
    """
    effect = get_genetic_effect(pop.traits[trait], component=component)
    if standardization is None:
        if effect.effects_standardized is None:
            raise ValueError(f"Effect '{component}' has no standardized effects stored.")
        beta = np.asarray(effect.effects_standardized, dtype=float)
        label = f'current@t={int(pop.t)}'
    else:
        if effect.effects_per_allele is None:
            raise ValueError(f"Effect '{component}' has no per-allele effects stored.")
        beta = standardization.to_standardized_effects(effect.effects_per_allele)
        label = standardization.label

    sum_b2 = float(np.sum(beta ** 2))
    sum_b4 = float(np.sum(beta ** 4))
    M_e = float(sum_b2 ** 2 / sum_b4) if sum_b4 > 0 else float('nan')
    return {'M_e': M_e, 'sum_beta2': sum_b2, 'sum_beta4': sum_b4,
            'standardization': label, 'M': int(beta.shape[0])}


def hardy_weinberg_departure(pop, trait: str,
                             standardization: Optional[BaseStandardization] = None,
                             component: str = 'A') -> dict:
    """Within-individual departure from Hardy-Weinberg, ``a_kk``, and ``sum_k beta_k^2 a_kk``.

    ``a_kk = Cov[z^(m)_k, z^(p)_k]`` is the covariance between an individual's two
    haplotypes at variant ``k`` in standardized units, so ``Var[x_ik] = 1 + 2 a_kk``
    (the writeup's ``\\eqref{eq:varx-a}``). It is zero under Hardy-Weinberg and grows
    positive under assortative mating.

    This is what makes the writeup's **exact** equilibrium formula testable::

        V_A^(eq) = (V_A^(0) - 2 sum_k beta_k^2 a_kk) / (1 - rho_g)     eq:VA-eq-exact

    ``\\eqref{eq:VA-eq}`` reaches its closed form by replacing that drag term with an
    approximation (``\\eqref{eq:VA-eq-drag}``, via ``Uniform-inflation``). Measuring
    ``sum_k beta_k^2 a_kk`` directly separates two things a residual against
    ``\\eqref{eq:VA-eq-correction}`` otherwise confounds: whether the drag term itself
    is right, versus whether ``Uniform-inflation`` evaluates it accurately.

    Half the same numerator is the exact segregation variance -- see
    :func:`~popstatgensim.estimation.am_equilibrium.predict_seg_var_exact`.

    Returns a dict with ``sum_beta2_a`` (the drag term, un-doubled), ``mean_a_kk``,
    ``max_a_kk``, the ``V_A_0`` it should be compared against, and the standardization
    used. Requires haplotypes (``pop.H``).
    """
    H = getattr(pop, 'H', None)
    if H is None:
        raise ValueError('Haplotypes (pop.H) are required to measure a_kk.')
    H = np.asarray(H)
    if H.ndim != 3 or H.shape[2] != 2:
        raise ValueError('a_kk is defined for diploids; pop.H must be (N, M, 2).')

    effect = get_genetic_effect(pop.traits[trait], component=component)
    if standardization is None:
        p = np.asarray(pop.p, dtype=float)
        beta = np.asarray(effect.effects_standardized, dtype=float)
        label = f'current@t={int(pop.t)}'
    else:
        # Constant-freq pins the denominator to the BASE generation's frequencies
        p = standardization.p
        beta = standardization.to_standardized_effects(effect.effects_per_allele)
        label = standardization.label

    hap_m = H[:, :, 0].astype(float)
    hap_p = H[:, :, 1].astype(float)
    cov_raw = ((hap_m - hap_m.mean(axis=0)) * (hap_p - hap_p.mean(axis=0))).mean(axis=0)

    denom = 2.0 * p * (1.0 - p)
    with np.errstate(divide='ignore', invalid='ignore'):
        a_kk = np.where(denom > 0, cov_raw / denom, 0.0)

    return {
        'sum_beta2_a': float(np.sum(beta ** 2 * a_kk)),
        'mean_a_kk': float(np.mean(a_kk)),
        'max_a_kk': float(np.max(np.abs(a_kk))),
        'V_A_0': float(np.sum(beta ** 2)),
        'standardization': label,
        'M': int(a_kk.shape[0]),
    }

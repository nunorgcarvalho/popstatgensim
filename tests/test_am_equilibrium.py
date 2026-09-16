"""Tests for the AM-equilibrium measurement plumbing and theory predictions."""

import warnings

import numpy as np
import pytest

from popstatgensim import GeneticEffect, Population
from popstatgensim.estimation import (
    AMMeasurements,
    compare_to_theory,
    measure_am_quantities,
    measure_mate_correlations,
    measure_segregation_variance,
    predict_h2_eq,
    predict_rho_g_eq,
    predict_seg_var,
    predict_VA_eq,
    predict_VA_eq_corrected,
)
from popstatgensim.traits import (
    BaseStandardization,
    effective_n_variants,
    find_variance_pinned_genetic_effects,
    var_A_from_effects,
    var_A_observed,
    var_A_pinned,
)


def build_pop(N=200, M=60, h2_0=0.5, seed=0, force_var=False):
    """A small population with one additive trait, ready to mate."""
    np.random.seed(seed)
    pop = Population(N=N, M=M, seed=seed)
    pop.assign_sex()
    pop.add_trait(
        name="y",
        effects={"A": GeneticEffect(var_indep=h2_0, M=M, M_causal=M, name="A",
                                    force_var=force_var)},
        var_Eps=1.0 - h2_0,
    )
    return pop


# ---------------------------------------------------------------------------
# Theory predictions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho_y", [0.0, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("h2_0", [0.1, 0.3, 0.5, 0.9])
def test_rhog_eq_is_consistent_with_hsq_eq(rho_y, h2_0):
    """eq:rhog-eq must satisfy the defining relation rho_g = rho_y * h2_eq (eq:rhog-t)."""
    rho_g = predict_rho_g_eq(rho_y, h2_0)
    h2_eq = predict_h2_eq(h2_0, rho_g)
    assert rho_g == pytest.approx(rho_y * h2_eq, abs=1e-12)


@pytest.mark.parametrize("rho_y,h2_0", [(0.5, 0.5), (0.3, 0.8), (0.8, 0.2)])
def test_seg_var_reduces_to_half_base_VA(rho_y, h2_0):
    """eq:g-seg-var at equilibrium equals 0.5 * V_A_0, per the writeup's remark."""
    VA_0 = 1.0
    rho_g = predict_rho_g_eq(rho_y, h2_0)
    VA_eq = predict_VA_eq(VA_0, rho_g)
    assert predict_seg_var(VA_eq, rho_g) == pytest.approx(0.5 * VA_0, rel=1e-12)


def test_no_AM_predictions_are_degenerate():
    """With rho_y = 0 nothing inflates."""
    assert predict_rho_g_eq(0.0, 0.5) == pytest.approx(0.0)
    assert predict_VA_eq(1.0, 0.0) == pytest.approx(1.0)
    assert predict_h2_eq(0.5, 0.0) == pytest.approx(0.5)
    assert predict_seg_var(1.0, 0.0) == pytest.approx(0.5)


def test_predictions_reject_invalid_inputs():
    with pytest.raises(ValueError):
        predict_VA_eq(1.0, 1.0)
    with pytest.raises(ValueError):
        predict_rho_g_eq(2.0, 0.5)  # discriminant goes negative


def test_rhog_eq_h2_one_limit():
    """At h2_0 = 1 the quadratic degenerates and rho_g = rho_y."""
    assert predict_rho_g_eq(0.4, 1.0) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# BaseStandardization
# ---------------------------------------------------------------------------

def test_standardization_pins_base_generation():
    pop = build_pop(seed=1)
    std = BaseStandardization.from_population(pop, method="observed")
    assert std.M == pop.M
    assert std.t_base == pop.t
    assert std.method == "observed"
    assert "pinned[observed]" in std.label
    # base generation: columns are standardized to ~unit variance
    X = std.standardize(pop.G)
    assert np.allclose(X.var(axis=0), 1.0, atol=1e-8)
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-8)


def test_standardization_is_frozen_against_drift():
    """The pinned SDs do not change as the population evolves."""
    pop = build_pop(seed=2)
    std = BaseStandardization.from_population(pop)
    G_std_before = std.G_std.copy()
    p_before = std.p.copy()
    pop.set_params(AM_r=0.4, AM_trait="y", AM_type="phenotypic")
    pop.simulate_generations(4)
    assert np.array_equal(std.G_std, G_std_before)
    assert np.array_equal(std.p, p_before)


def test_binomial_and_observed_agree_in_base_generation():
    """Under HWE the two pinning methods agree up to sampling noise."""
    pop = build_pop(N=2000, M=200, seed=3)
    obs = BaseStandardization.from_population(pop, method="observed")
    binom = BaseStandardization.from_population(pop, method="binomial")
    assert np.allclose(obs.G_std, binom.G_std, rtol=0.15)


def test_standardization_rejects_wrong_M():
    pop = build_pop(seed=4)
    std = BaseStandardization.from_population(pop)
    with pytest.raises(ValueError, match="variants to match"):
        std.standardize(np.zeros((10, pop.M + 1)))
    with pytest.raises(ValueError):
        std.to_standardized_effects(np.zeros(pop.M + 1))


def test_invalid_standardization_method():
    pop = build_pop(seed=5)
    with pytest.raises(ValueError, match="method"):
        BaseStandardization.from_population(pop, method="nonsense")


def test_pinned_var_A_equals_realized_genetic_variance():
    """Because effects_per_allele is fixed, the pinned V_A IS the realized Var[g]."""
    pop = build_pop(seed=6)
    std = BaseStandardization.from_population(pop)
    pop.set_params(AM_r=0.5, AM_trait="y", AM_type="phenotypic")
    pop.simulate_generations(3)
    assert var_A_pinned(pop, "y", std) == pytest.approx(var_A_observed(pop, "y"), rel=1e-8)


def test_pinned_effect_variance_survives_generations_but_current_drifts():
    """sum(beta_std^2) is stable under pinning and drifts without it.

    This is the concrete failure the pinned standardization exists to prevent:
    `GeneticEffect.update_G_std()` rebuilds `effects_standardized` from the current
    generation's genotype SDs, so the per-generation figure moves even though the
    per-allele effects never change.
    """
    pop = build_pop(N=500, M=200, seed=7)
    std = BaseStandardization.from_population(pop)
    base_pinned = var_A_from_effects(pop, "y", std)
    base_current = var_A_from_effects(pop, "y", None)
    # in the base generation, with observed pinning, the two coincide
    assert base_pinned == pytest.approx(base_current, rel=1e-8)

    pop.set_params(AM_r=0.5, AM_trait="y", AM_type="phenotypic")
    pop.simulate_generations(8)

    # pinned value is bit-for-bit unchanged
    assert var_A_from_effects(pop, "y", std) == pytest.approx(base_pinned, rel=1e-10)
    # per-generation value has moved with allele-frequency drift
    assert var_A_from_effects(pop, "y", None) != pytest.approx(base_current, rel=1e-6)


def test_allele_freq_drift_diagnostic():
    pop = build_pop(N=500, M=200, seed=8)
    std = BaseStandardization.from_population(pop)
    d0 = std.allele_freq_drift(pop)
    assert d0["generations_elapsed"] == 0
    assert d0["mean_abs_dp"] == pytest.approx(0.0)
    assert d0["max_abs_dp"] == pytest.approx(0.0)

    pop.set_params(AM_r=0.4, AM_trait="y", AM_type="phenotypic")
    pop.simulate_generations(6)
    d = std.allele_freq_drift(pop)
    assert d["generations_elapsed"] == 6
    assert d["M"] == pop.M
    assert d["mean_abs_dp"] > 0
    assert d["max_abs_dp"] >= d["mean_abs_dp"]
    assert 0 <= d["n_lost"] <= pop.M


# ---------------------------------------------------------------------------
# Variance-pinning guard
# ---------------------------------------------------------------------------

def test_guard_finds_force_var_genetic_effect():
    pop = build_pop(seed=9, force_var=True)
    pinned = find_variance_pinned_genetic_effects(pop.traits)
    assert ("y", "A", "force_var=True") in pinned


def test_guard_ignores_noise_effect():
    """NoiseEffect defaults to force_var=True and that is correct -- V_E is held fixed."""
    pop = build_pop(seed=10, force_var=False)
    assert pop.traits["y"].effects["Eps"].force_var is True
    assert find_variance_pinned_genetic_effects(pop.traits) == []


def test_guard_warns_when_simulating_AM_with_pinned_genetic_variance():
    pop = build_pop(seed=11, force_var=True)
    pop.set_params(AM_r=0.5, AM_trait="y", AM_type="phenotypic")
    with pytest.warns(UserWarning, match="variance pinned"):
        pop.simulate_generations(1)


def test_guard_silent_without_AM():
    pop = build_pop(seed=12, force_var=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pop.simulate_generations(1)  # no AM configured -> no guard warning


def test_guard_silent_when_genetic_variance_is_free():
    pop = build_pop(seed=13, force_var=False)
    pop.set_params(AM_r=0.5, AM_trait="y", AM_type="phenotypic")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pop.simulate_generations(1)
    assert not any("variance pinned" in str(w.message) for w in caught)


def test_force_var_actually_destroys_VA_inflation():
    """The behaviour the guard warns about: pinned V_A does not inflate under AM."""
    pinned_pop = build_pop(N=600, M=200, seed=14, force_var=True)
    free_pop = build_pop(N=600, M=200, seed=14, force_var=False)

    for pop in (pinned_pop, free_pop):
        pop.set_params(AM_r=0.6, AM_trait="y", AM_type="phenotypic")

    std_pinned = BaseStandardization.from_population(pinned_pop)
    std_free = BaseStandardization.from_population(free_pop)
    VA0_pinned = var_A_pinned(pinned_pop, "y", std_pinned)
    VA0_free = var_A_pinned(free_pop, "y", std_free)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pinned_pop.simulate_generations(10)
    free_pop.simulate_generations(10)

    # The REALIZED variance is the one force_var pins, and the one the theory predicts.
    VA_pinned = var_A_observed(pinned_pop, "y")
    VA_free = var_A_observed(free_pop, "y")

    # force_var=True holds the realized genetic variance at var_indep
    assert VA_pinned == pytest.approx(VA0_pinned, rel=0.15)
    # force_var=False lets it inflate substantially
    assert VA_free > 1.2 * VA0_free

    # And the footgun's signature: recomputing V_A from genotypes bypasses the rescale,
    # so it still shows the inflation the realized values no longer have.
    assert var_A_pinned(pinned_pop, "y", std_pinned) > 1.2 * VA_pinned
    # With force_var=False the two agree, because nothing was rescaled.
    assert var_A_pinned(free_pop, "y", std_free) == pytest.approx(VA_free, rel=1e-8)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def test_mate_correlations_track_requested_AM_r():
    pop = build_pop(N=1000, M=100, seed=15)
    pop.set_params(AM_r=0.6, AM_trait="y", AM_type="phenotypic")
    pop.simulate_generations(1)
    mate = measure_mate_correlations(pop.past[1], "y")
    assert mate["n_mate_pairs"] == pop.past[1].N // 2
    # rank-matching pairing lands close to, but not exactly at, AM_r
    assert mate["rho_y"] == pytest.approx(0.6, abs=0.1)
    # rho_g is positive and smaller than rho_y (it is roughly rho_y * h2)
    assert 0.0 < mate["rho_g"] < mate["rho_y"]


def test_mate_correlations_zero_under_random_mating():
    pop = build_pop(N=1000, M=100, seed=16)
    pop.simulate_generations(1)  # AM_r defaults to 0
    mate = measure_mate_correlations(pop.past[1], "y")
    assert mate["rho_y"] == pytest.approx(0.0, abs=0.12)


def test_mate_correlations_require_spouses():
    pop = build_pop(seed=17)
    with pytest.raises(ValueError, match="spouse"):
        measure_mate_correlations(pop, "y")


def test_segregation_variance_matches_half_base_VA():
    """eq:g-seg-var under random mating: Var[eps_o] = 0.5 * V_A_0."""
    pop = build_pop(N=2000, M=300, seed=18)
    std = BaseStandardization.from_population(pop)
    VA_0 = var_A_pinned(pop, "y", std)
    pop.simulate_generations(1)
    seg = measure_segregation_variance(pop, "y")
    assert seg == pytest.approx(0.5 * VA_0, rel=0.15)


def test_segregation_variance_requires_parents():
    pop = build_pop(seed=19)
    with pytest.raises(ValueError, match="[Pp]arent|Previous generation"):
        measure_segregation_variance(pop, "y")


def test_measure_am_quantities_labels_its_standardization():
    pop = build_pop(seed=20)
    std = BaseStandardization.from_population(pop)
    with_std = measure_am_quantities(pop, "y", standardization=std)
    without = measure_am_quantities(pop, "y")
    assert with_std.standardization == std.label
    assert without.standardization.startswith("realized@")
    assert without.drift is None
    assert with_std.drift is not None
    # V_A is the realized variance either way; only the companion fields need pinning
    assert with_std.V_A == pytest.approx(without.V_A, rel=1e-12)
    assert with_std.V_A_from_genotypes is not None
    assert without.V_A_from_genotypes is None


def test_measure_am_quantities_base_generation_has_no_mating_fields():
    pop = build_pop(seed=21)
    m = measure_am_quantities(pop, "y")
    assert isinstance(m, AMMeasurements)
    assert m.t == 0
    assert m.rho_y is None and m.rho_g is None and m.var_eps_o is None
    assert m.h2 == pytest.approx(m.V_A / m.V_Y)


def test_measure_am_quantities_after_mating_is_complete():
    pop = build_pop(N=1000, M=200, seed=22)
    std = BaseStandardization.from_population(pop)
    pop.set_params(AM_r=0.5, AM_trait="y", AM_type="phenotypic")
    pop.simulate_generations(2)
    m = measure_am_quantities(pop, "y", standardization=std)
    for field in ("rho_y", "rho_g", "var_eps_o", "parent_h2", "parent_V_A"):
        assert getattr(m, field) is not None, field
    assert m.n_mate_pairs == pop.N // 2
    assert m.N == pop.N
    assert set(m.as_dict()) >= {"V_A", "V_Y", "h2", "rho_y", "rho_g", "var_eps_o"}


def test_measure_am_quantities_rejects_unknown_component():
    pop = build_pop(seed=23)
    with pytest.raises(ValueError, match="no effect named"):
        measure_am_quantities(pop, "y", component="nope")


def test_measure_am_quantities_rejects_non_genetic_component():
    pop = build_pop(seed=24)
    with pytest.raises(ValueError, match="not a GeneticEffect"):
        measure_am_quantities(pop, "y", component="Eps")


def test_VA_inflates_over_multiple_generations_of_AM():
    """The headline behaviour: V_A rises rather than staying pinned."""
    pop = build_pop(N=1000, M=300, seed=25)
    std = BaseStandardization.from_population(pop)
    VA_0 = var_A_pinned(pop, "y", std)

    pop.set_params(AM_r=0.6, AM_trait="y", AM_type="phenotypic")
    trajectory = []
    for _ in range(8):
        pop.simulate_generations(1)
        trajectory.append(var_A_pinned(pop, "y", std))

    assert trajectory[-1] > 1.25 * VA_0
    # monotone-ish rise: the last third sits above the first third
    assert np.mean(trajectory[-3:]) > np.mean(trajectory[:3])


def test_compare_to_theory_shapes_and_no_AM_case():
    pop = build_pop(N=800, M=200, seed=26)
    std = BaseStandardization.from_population(pop)
    base = measure_am_quantities(pop, "y", standardization=std)
    pop.simulate_generations(3)  # random mating
    m = measure_am_quantities(pop, "y", standardization=std)
    cmp = compare_to_theory(m, VA_0=base.V_A, h2_0=base.h2)
    assert set(cmp) == {"eq:rhog-eq", "eq:VA-eq", "eq:VA-eq-correction",
                        "eq:hsq-eq", "eq:g-seg-var"}
    for key, entry in cmp.items():
        assert set(entry) == {"predicted", "measured", "rel_err"}
    # under random mating V_A should be close to its base value
    assert cmp["eq:VA-eq"]["rel_err"] == pytest.approx(0.0, abs=0.25)


@pytest.mark.parametrize("AM_r,h2_0", [(0.4, 0.5), (0.6, 0.3)])
def test_equilibrium_predictions_hold_in_simulation(AM_r, h2_0):
    """End-to-end: all four equations reproduce within Monte-Carlo error.

    Small parameters and averaging over post-burn-in generations keep this cheap
    while still being a real test; tolerances are loose enough to be stable across
    seeds but tight enough to fail if a formula or a measurement is wrong.
    """
    pop = build_pop(N=1000, M=300, h2_0=h2_0, seed=27)
    std = BaseStandardization.from_population(pop)
    base = measure_am_quantities(pop, "y", standardization=std)

    pop.set_params(AM_r=AM_r, AM_trait="y", AM_type="phenotypic", s=0.0)
    pop.simulate_generations(10)  # burn-in to equilibrium

    acc = {k: [] for k in ("V_A", "h2", "rho_y", "rho_g", "var_eps_o")}
    for _ in range(12):
        pop.simulate_generations(1)
        m = measure_am_quantities(pop, "y", standardization=std, include_drift=False)
        for k in acc:
            acc[k].append(getattr(m, k))
    avg = {k: float(np.mean(v)) for k, v in acc.items()}

    rho_y, rho_g = avg["rho_y"], avg["rho_g"]

    # eq:rhog-eq
    assert predict_rho_g_eq(rho_y, base.h2) == pytest.approx(rho_g, rel=0.20)
    # eq:VA-eq
    assert predict_VA_eq(base.V_A, rho_g) == pytest.approx(avg["V_A"], rel=0.15)
    # eq:hsq-eq
    assert predict_h2_eq(base.h2, rho_g) == pytest.approx(avg["h2"], rel=0.12)
    # eq:g-seg-var -- the sharpest of the four
    assert predict_seg_var(avg["V_A"], rho_g) == pytest.approx(avg["var_eps_o"], rel=0.10)


# ---------------------------------------------------------------------------
# M_e and the eq:VA-eq correction term
# ---------------------------------------------------------------------------

def test_M_e_equals_M_for_equal_effect_sizes():
    """eq:Me-def is normalized so M_e = M when every variant has the same effect."""
    pop = build_pop(N=200, M=50, seed=30)
    effect = pop.traits["y"].effects["A"]
    # force all standardized effects equal
    effect.effects_standardized = np.full(pop.M, 0.1)
    effect.effects_per_allele = effect.effects_standardized / effect.G_std
    std = BaseStandardization.from_population(pop)
    # pinned units reproduce the equal effects exactly
    me = effective_n_variants(pop, "y", std)
    assert me["M_e"] == pytest.approx(pop.M, rel=1e-6)
    assert me["M"] == pop.M


def test_M_e_is_smaller_when_effects_are_concentrated():
    pop = build_pop(N=200, M=50, seed=31)
    effect = pop.traits["y"].effects["A"]
    concentrated = np.full(pop.M, 1e-6)
    concentrated[0] = 1.0
    effect.effects_standardized = concentrated
    effect.effects_per_allele = concentrated / effect.G_std
    std = BaseStandardization.from_population(pop)
    assert effective_n_variants(pop, "y", std)["M_e"] < 2.0


def test_M_e_is_invariant_to_global_rescaling():
    pop = build_pop(N=200, M=50, seed=32)
    std = BaseStandardization.from_population(pop)
    before = effective_n_variants(pop, "y", std)["M_e"]
    effect = pop.traits["y"].effects["A"]
    effect.effects_per_allele = effect.effects_per_allele * 7.0
    after = effective_n_variants(pop, "y", std)["M_e"]
    assert after == pytest.approx(before, rel=1e-10)


def test_M_e_pinned_is_stable_across_generations():
    pop = build_pop(N=500, M=200, seed=33)
    std = BaseStandardization.from_population(pop)
    before = effective_n_variants(pop, "y", std)["M_e"]
    pop.set_params(AM_r=0.5, AM_trait="y", AM_type="phenotypic")
    pop.simulate_generations(6)
    assert effective_n_variants(pop, "y", std)["M_e"] == pytest.approx(before, rel=1e-10)


def test_VA_eq_correction_is_negative_and_shrinks_like_one_over_Me():
    """The writeup states the sign: dropping the term OVERSTATES V_A_eq."""
    VA_0, rho_g = 1.0, 0.3
    plain = predict_VA_eq(VA_0, rho_g)
    errs = []
    for M_e in (10.0, 100.0, 1000.0):
        corrected = predict_VA_eq_corrected(VA_0, rho_g, M_e)
        assert corrected < plain  # signed error: plain overstates
        errs.append((plain - corrected) / plain)
    # relative error falls by ~10x for each 10x in M_e
    assert errs[0] / errs[1] == pytest.approx(10.0, rel=1e-6)
    assert errs[1] / errs[2] == pytest.approx(10.0, rel=1e-6)


def test_VA_eq_correction_vanishes_without_AM():
    assert predict_VA_eq_corrected(1.0, 0.0, 100.0) == pytest.approx(predict_VA_eq(1.0, 0.0))


def test_VA_eq_correction_rejects_bad_Me():
    with pytest.raises(ValueError, match="M_e"):
        predict_VA_eq_corrected(1.0, 0.3, 0.0)
    with pytest.raises(ValueError, match="M_e"):
        predict_VA_eq_corrected(1.0, 0.3, float("nan"))


def test_measurements_carry_Me_and_sum_beta4():
    pop = build_pop(N=400, M=150, seed=34)
    std = BaseStandardization.from_population(pop)
    m = measure_am_quantities(pop, "y", standardization=std)
    assert m.M_e is not None and m.M_e > 1.0
    assert m.sum_beta4 is not None and m.sum_beta4 > 0
    # M_e = (sum beta^2)^2 / sum beta^4
    assert m.M_e == pytest.approx(m.V_A_from_effects ** 2 / m.sum_beta4, rel=1e-9)


def test_compare_to_theory_includes_corrected_VA_row():
    pop = build_pop(N=600, M=200, seed=35)
    std = BaseStandardization.from_population(pop)
    base = measure_am_quantities(pop, "y", standardization=std)
    pop.set_params(AM_r=0.5, AM_trait="y", AM_type="phenotypic")
    pop.simulate_generations(4)
    m = measure_am_quantities(pop, "y", standardization=std)
    cmp = compare_to_theory(m, VA_0=base.V_A, h2_0=base.h2)
    assert "eq:VA-eq-correction" in cmp
    # the corrected prediction must sit below the plain one
    assert cmp["eq:VA-eq-correction"]["predicted"] < cmp["eq:VA-eq"]["predicted"]


# ---------------------------------------------------------------------------
# Covariance-free h2 (GE-indep) and the g-e diagnostic
# ---------------------------------------------------------------------------

def test_V_E_is_exact_and_pinned_by_noise_effect():
    """y = g + e, so Var[y - g] = Var[e] exactly -- no covariance term leaks in."""
    pop = build_pop(N=2000, M=200, h2_0=0.4, seed=40)
    m = measure_am_quantities(pop, "y")
    # NoiseEffect defaults to force_var=True, which is correct: the writeup holds V_E fixed
    assert m.V_E == pytest.approx(0.6, rel=1e-9)


def test_h2_cov_free_matches_writeup_definition():
    pop = build_pop(N=2000, M=200, h2_0=0.5, seed=41)
    m = measure_am_quantities(pop, "y")
    assert m.h2_cov_free == pytest.approx(m.V_A / (m.V_A + m.V_E), rel=1e-12)
    # the empirical-V_Y version differs by exactly the realized g-e covariance
    assert m.h2 == pytest.approx(m.V_A / m.V_Y, rel=1e-12)


def test_corr_g_e_is_small_and_reported():
    """GE-indep assumes this is zero; we report it rather than absorb it."""
    N = 4000
    pop = build_pop(N=N, M=200, h2_0=0.5, seed=42)
    m = measure_am_quantities(pop, "y")
    assert m.corr_g_e is not None
    # ordinary sampling noise: should sit within a few 1/sqrt(N)
    assert abs(m.corr_g_e) < 5.0 / np.sqrt(N)


def test_both_boxed_forms_of_hsq_eq_agree_under_cov_free_h2():
    """The two boxed forms of eq:hsq-eq are algebraically identical.

    Fed an empirically measured V_Y^(0) they disagree, because that absorbs the realized
    g-e covariance. Fed the covariance-free h2_0 they agree by construction -- which is
    the reason to prefer it.
    """
    pop = build_pop(N=4000, M=300, h2_0=0.5, seed=43)
    m = measure_am_quantities(pop, "y")
    rho_g = 0.28

    form1 = predict_h2_eq(m.h2_cov_free, rho_g)
    form2 = m.V_A / (m.V_A + (1.0 - rho_g) * m.V_E)
    assert form1 == pytest.approx(form2, rel=1e-12)

    # with the empirical-V_Y h2_0 the two forms do NOT agree
    form1_noisy = predict_h2_eq(m.h2, rho_g)
    assert form1_noisy != pytest.approx(form2, rel=1e-9)


def test_compare_to_theory_cov_free_flag_selects_the_measured_h2():
    pop = build_pop(N=1000, M=200, seed=44)
    std = BaseStandardization.from_population(pop)
    base = measure_am_quantities(pop, "y", standardization=std)
    pop.set_params(AM_r=0.5, AM_trait="y", AM_type="phenotypic")
    pop.simulate_generations(3)
    m = measure_am_quantities(pop, "y", standardization=std)

    free = compare_to_theory(m, VA_0=base.V_A, h2_0=base.h2_cov_free, cov_free=True)
    noisy = compare_to_theory(m, VA_0=base.V_A, h2_0=base.h2_cov_free, cov_free=False)
    assert free["eq:hsq-eq"]["measured"] == pytest.approx(m.h2_cov_free)
    assert noisy["eq:hsq-eq"]["measured"] == pytest.approx(m.h2)

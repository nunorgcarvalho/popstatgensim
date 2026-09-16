"""Tests for the path signature and the writeup's closed-form covariance predictor."""

import numpy as np
import pytest

from popstatgensim.pedigree.paths import (
    LinealChain,
    MatingChain,
    PathSignature,
    chain_weight,
    implied_covariance,
    predicted_corr,
    predicted_cov,
    proband_correction,
)

RHO_Y, RHO_G, V_A = 0.4, 0.2, 1.3


# ---------------------------------------------------------------------------
# eq:chain-weight
# ---------------------------------------------------------------------------

def test_implied_covariance_cases():
    """c(0) = 1; c(N_mu) = rho_g rho_y^(N_mu-1)."""
    assert implied_covariance(0, RHO_Y, RHO_G) == 1.0
    assert implied_covariance(1, RHO_Y, RHO_G) == pytest.approx(RHO_G)
    assert implied_covariance(2, RHO_Y, RHO_G) == pytest.approx(RHO_G * RHO_Y)
    assert implied_covariance(3, RHO_Y, RHO_G) == pytest.approx(RHO_G * RHO_Y ** 2)


def test_chain_weight_reproduces_the_three_named_cases():
    """eq:chain-weight must give eq:chain-weight-cases without a side condition (D-0008)."""
    # full sibs
    assert chain_weight(1, 2, RHO_Y, RHO_G) == pytest.approx((1 + RHO_G) / 2)
    # half sibs
    assert chain_weight(2, 2, RHO_Y, RHO_G) == pytest.approx(
        (1 + 2 * RHO_G + RHO_G * RHO_Y) / 4)
    # step sibs
    assert chain_weight(3, 2, RHO_Y, RHO_G) == pytest.approx(
        RHO_G * ((1 + RHO_Y) / 2) ** 2)


def test_chain_weight_no_mating_is_unity():
    assert chain_weight(0, 0, RHO_Y, RHO_G) == pytest.approx(1.0)


def test_chain_weight_validation():
    with pytest.raises(ValueError, match="N_o"):
        chain_weight(1, 3, RHO_Y, RHO_G)
    with pytest.raises(ValueError, match="N_mu"):
        chain_weight(-1, 1, RHO_Y, RHO_G)


def test_chain_weight_is_at_most_one():
    """The admissibility condition for best-first search: every factor is in (0, 1]."""
    for rho_y in (0.0, 0.3, 0.6, 0.9):
        for rho_g in (0.0, 0.2, 0.5):
            for N_mu in range(5):
                for N_o in range(3):
                    w = chain_weight(N_mu, N_o, rho_y, rho_g)
                    assert 0.0 <= w <= 1.0 + 1e-12, (N_mu, N_o, rho_y, rho_g, w)


def test_chain_weight_honours_per_mating_rho_y():
    """Round k > 1 can realize a different rho_y; the chain must use the product."""
    # N_mu = 3 crosses 3 matings, so 2 rho_y factors
    default = chain_weight(3, 0, RHO_Y, RHO_G)
    assert default == pytest.approx(RHO_G * RHO_Y ** 2)
    per_mating = chain_weight(3, 0, RHO_Y, RHO_G, rho_y_chain=(0.3, 0.5))
    assert per_mating == pytest.approx(RHO_G * 0.3 * 0.5)
    # passing the same value everywhere reproduces the pooled form
    same = chain_weight(3, 0, RHO_Y, RHO_G, rho_y_chain=(RHO_Y, RHO_Y))
    assert same == pytest.approx(default)


# ---------------------------------------------------------------------------
# eq:proband-correction
# ---------------------------------------------------------------------------

def test_proband_corrections():
    assert proband_correction('sink', RHO_Y, RHO_G) == 1.0
    assert proband_correction('source', RHO_Y, RHO_G) == pytest.approx(
        (1 + RHO_Y) / (1 + RHO_G))
    # terminal mating chain: rho_y/rho_g = 1/h^2
    assert proband_correction('terminal_mating', RHO_Y, RHO_G) == pytest.approx(
        RHO_Y / RHO_G)


def test_terminal_mating_correction_is_one_over_h2():
    """rho_g = rho_y h^2 (eq:rhog-t), so rho_y/rho_g is exactly 1/h^2."""
    h2 = 0.5
    rho_y = 0.4
    rho_g = rho_y * h2
    assert proband_correction('terminal_mating', rho_y, rho_g) == pytest.approx(1 / h2)


def test_proband_correction_validation():
    with pytest.raises(ValueError, match="case must be"):
        proband_correction('nonsense', RHO_Y, RHO_G)
    with pytest.raises(ValueError, match="singular"):
        proband_correction('terminal_mating', RHO_Y, 0.0)


# ---------------------------------------------------------------------------
# eq:path-product, against the writeup's own worked example
# ---------------------------------------------------------------------------

def example_signature():
    """fig:path-example: sum N_lin = 2+3+1 = 6, chains (N_mu=2,N_o=0) and (N_mu=3,N_o=2)."""
    return PathSignature(
        sum_N_lin=6, sum_N_o=2, mating_chains=((2, 0), (3, 2)),
        proband_A_case='source', proband_B_case='sink',
    )


def test_worked_example_genetic_covariance():
    """Cov[g_A,g_B] = V_A rho_g^2 rho_y ((1+rho_g)/2)^4 ((1+rho_y)/2)^2."""
    sig = example_signature()
    expected = (V_A * RHO_G ** 2 * RHO_Y
                * ((1 + RHO_G) / 2) ** 4 * ((1 + RHO_Y) / 2) ** 2)
    assert predicted_cov(sig, V_A, RHO_Y, RHO_G, phenotypic=False) == pytest.approx(expected)


def test_worked_example_phenotypic_covariance():
    """Cov[y_A,y_B] = V_A rho_g^2 rho_y ((1+rho_g)/2)^3 ((1+rho_y)/2)^3."""
    sig = example_signature()
    expected = (V_A * RHO_G ** 2 * RHO_Y
                * ((1 + RHO_G) / 2) ** 3 * ((1 + RHO_Y) / 2) ** 3)
    assert predicted_cov(sig, V_A, RHO_Y, RHO_G, phenotypic=True) == pytest.approx(expected)


def test_predictor_is_symbolic_when_given_symbols():
    """Polymorphic on purpose, so it can be checked against the algebra, not just numbers."""
    sympy = pytest.importorskip("sympy")
    rg, ry, va = sympy.symbols('rho_g rho_y V_A', positive=True)
    sig = example_signature()
    got = predicted_cov(sig, va, ry, rg, phenotypic=True)
    expected = va * rg ** 2 * ry * ((1 + rg) / 2) ** 3 * ((1 + ry) / 2) ** 3
    assert sympy.simplify(got - expected) == 0


@pytest.mark.parametrize("N_mu,N_o,expected_name", [
    (1, 2, "full sibs"), (2, 2, "half sibs"), (3, 2, "step sibs"),
])
def test_sibling_classes_via_the_full_predictor(N_mu, N_o, expected_name):
    """Sibling classes differ only in N_mu within a single mating chain."""
    sig = PathSignature(sum_N_lin=2, sum_N_o=N_o, mating_chains=((N_mu, N_o),),
                        proband_A_case='sink', proband_B_case='sink')
    got = predicted_cov(sig, V_A, RHO_Y, RHO_G, phenotypic=False)
    expected = V_A * ((1 + RHO_G) / 2) ** (2 - N_o) * chain_weight(N_mu, N_o, RHO_Y, RHO_G)
    assert got == pytest.approx(expected)


def test_full_sibs_reduce_to_the_known_coefficient():
    """Full sibs: sum N_lin = 2, N_o = 2, so the exponent vanishes and Cov = V_A (1+rho_g)/2."""
    sig = PathSignature(sum_N_lin=2, sum_N_o=2, mating_chains=((1, 2),))
    assert predicted_cov(sig, V_A, RHO_Y, RHO_G, phenotypic=False) == pytest.approx(
        V_A * (1 + RHO_G) / 2)


def test_parent_offspring_has_no_mating_chain():
    """Parent-offspring: one meiosis, no mating chain, so Cov = V_A (1+rho_g)/2."""
    sig = PathSignature(sum_N_lin=1, sum_N_o=0, mating_chains=())
    assert predicted_cov(sig, V_A, RHO_Y, RHO_G, phenotypic=False) == pytest.approx(
        V_A * (1 + RHO_G) / 2)


def test_self_pair_is_V_A():
    sig = PathSignature(sum_N_lin=0, sum_N_o=0, mating_chains=())
    assert predicted_cov(sig, V_A, RHO_Y, RHO_G, phenotypic=False) == pytest.approx(V_A)


def test_parent_parent_is_a_bare_mating_chain():
    """Two mates: no lineal chain at all, so both probands are terminal-mating."""
    sig = PathSignature(sum_N_lin=0, sum_N_o=0, mating_chains=((1, 0),),
                        proband_A_case='terminal_mating',
                        proband_B_case='terminal_mating')
    genetic = predicted_cov(sig, V_A, RHO_Y, RHO_G, phenotypic=False)
    assert genetic == pytest.approx(V_A * RHO_G)
    # phenotypically this must come back to V_A rho_y: (rho_y/rho_g)^2 V_A rho_g
    pheno = predicted_cov(sig, V_A, RHO_Y, RHO_G, phenotypic=True)
    assert pheno == pytest.approx(V_A * RHO_Y ** 2 / RHO_G)


def test_no_assortative_mating_collapses_to_classical_coefficients():
    """With rho_y = rho_g = 0 the predictions must be the textbook ones."""
    cases = {
        'self': (PathSignature(0, 0, ()), 1.0),
        'parent-offspring': (PathSignature(1, 0, ()), 0.5),
        'grandparent': (PathSignature(2, 0, ()), 0.25),
        'full sibs': (PathSignature(2, 2, ((1, 2),)), 0.5),
        'half sibs': (PathSignature(2, 2, ((2, 2),)), 0.25),
    }
    for name, (sig, expected) in cases.items():
        got = predicted_cov(sig, 1.0, 0.0, 0.0, phenotypic=False)
        assert got == pytest.approx(expected), name


# ---------------------------------------------------------------------------
# Signature mechanics
# ---------------------------------------------------------------------------

def test_signature_validates_sum_N_o():
    with pytest.raises(ValueError, match="sum_N_o"):
        PathSignature(sum_N_lin=2, sum_N_o=5, mating_chains=((1, 2),))


def test_signature_validates_N_o_range():
    with pytest.raises(ValueError, match="N_o"):
        PathSignature(sum_N_lin=2, sum_N_o=3, mating_chains=((1, 3),))


def test_signature_validates_proband_case():
    with pytest.raises(ValueError, match="proband case"):
        PathSignature(2, 2, ((1, 2),), proband_A_case='middle')


def test_systematic_name_is_stable_and_descriptive():
    sig = example_signature()
    name = sig.systematic_name()
    assert name == "Lin6-Mu[2/0,3/2]-snk+src"
    # the old catch-all was a single 'unspecified'; every path now gets a distinct label
    other = PathSignature(2, 2, ((1, 2),)).systematic_name()
    assert other != name


def test_signature_key_is_hashable_and_order_insensitive_at_the_ends():
    a = PathSignature(2, 2, ((1, 2),), proband_A_case='sink', proband_B_case='source')
    b = PathSignature(2, 2, ((1, 2),), proband_A_case='source', proband_B_case='sink')
    assert a.key == b.key
    assert len({a.key, b.key}) == 1


def test_predicted_corr_divides_by_V_Y():
    sig = PathSignature(2, 2, ((1, 2),))
    V_Y = 2.0
    assert predicted_corr(sig, V_A, V_Y, RHO_Y, RHO_G) == pytest.approx(
        predicted_cov(sig, V_A, RHO_Y, RHO_G, phenotypic=True) / V_Y)


def test_chain_objects_validate():
    with pytest.raises(ValueError, match="N_lin"):
        LinealChain(young_end=1, old_end=0, N_lin=-1)
    with pytest.raises(ValueError, match="N_mu"):
        MatingChain(start=0, end=1, N_mu=-1)


# ---------------------------------------------------------------------------
# Guards: inadmissible signatures and ambiguous multi-round scoring
# ---------------------------------------------------------------------------

def test_predicted_cov_rejects_an_inadmissible_1_1_chain():
    """(N_mu=1, N_o=1) describes a path no pedigree can realize.

    relate() excludes it during enumeration, but a consumer building a signature BY
    HAND reaches predicted_cov directly -- and would silently get 1.25 where the truth
    is 0.75. predicted_cov takes a path DESCRIPTION, not a weight, so the impossible
    description is refused here.
    """
    bad = PathSignature(1, 1, ((1, 1),), 'terminal_mating', 'sink')
    with pytest.raises(ValueError, match="inadmissible"):
        predicted_cov(bad, V_A, RHO_Y, RHO_G)
    with pytest.raises(ValueError, match="inadmissible"):
        predicted_corr(bad, V_A, 1.0, RHO_Y, RHO_G)


def test_chain_weight_1_1_is_still_defined():
    """The exclusion is about admissible PATHS, not about the weight function."""
    assert chain_weight(1, 1, RHO_Y, RHO_G) == pytest.approx((1 + RHO_G) / 2)


def test_other_N_o_1_chains_are_still_accepted():
    """Only N_mu = 1 collapses into the descent; N_mu >= 2 with N_o = 1 is fine."""
    ok = PathSignature(1, 1, ((2, 1),), 'terminal_mating', 'sink')
    assert predicted_cov(ok, V_A, RHO_Y, RHO_G) > 0


def test_multi_round_chain_refuses_to_pool_silently():
    """D-0014: no default in the ambiguous case."""
    sig = PathSignature(2, 2, ((2, 2),), mating_round_ids=((0, 1),))
    with pytest.raises(ValueError, match="crosses matching rounds"):
        predicted_cov(sig, V_A, RHO_Y, RHO_G)


def test_pooled_is_available_as_an_explicit_opt_in():
    """The writeup's literal form stays reachable, but must be asked for."""
    sig = PathSignature(2, 2, ((2, 2),), mating_round_ids=((0, 1),))
    plain = PathSignature(2, 2, ((2, 2),))
    assert predicted_cov(sig, V_A, RHO_Y, RHO_G, rho_y_by_round='pooled') == pytest.approx(
        predicted_cov(plain, V_A, RHO_Y, RHO_G))


def test_single_round_chains_pool_without_complaint():
    """Both targets coincide exactly there, so requiring a decision would be noise."""
    sig = PathSignature(2, 2, ((2, 2),), mating_round_ids=((0, 0),))
    assert predicted_cov(sig, V_A, RHO_Y, RHO_G) == pytest.approx(
        predicted_cov(PathSignature(2, 2, ((2, 2),)), V_A, RHO_Y, RHO_G))


def test_mapping_gives_the_per_round_product():
    sig = PathSignature(2, 2, ((2, 2),), mating_round_ids=((0, 1),))
    # N_mu = 2 uses one rho_y factor, taken from the first edge's round
    got = predicted_cov(sig, V_A, RHO_Y, RHO_G, rho_y_by_round={0: 0.3, 1: 0.9})
    expected = V_A * chain_weight(2, 2, RHO_Y, RHO_G, rho_y_chain=[0.3])
    assert got == pytest.approx(expected)

"""Tests for the mate-graph model (non-monogamous assortative mating by rounds)."""

import numpy as np
import pytest

from popstatgensim.simulation.mating import (
    MateGraph,
    build_mate_graph,
    mean_n_mates,
    mean_offspring_per_pair,
    measure_mate_chain_correlation,
    normalize_survival,
    survival_from_pmf,
)


@pytest.fixture
def pop_inputs():
    """A sex-balanced population with standardized AM values."""
    rng = np.random.default_rng(0)
    N = 2000
    sex = np.zeros(N, dtype=int)
    sex[N // 2:] = 1
    sex = rng.permutation(sex)
    y = rng.normal(size=N)
    y = (y - y.mean()) / y.std()
    return N, sex, y


# ---------------------------------------------------------------------------
# Survival array / pmf
# ---------------------------------------------------------------------------

def test_survival_from_pmf():
    assert np.allclose(survival_from_pmf([0.0, 0.5, 0.5]), [1.0, 0.5])
    assert np.allclose(survival_from_pmf([0.0, 1.0]), [1.0])
    assert np.allclose(survival_from_pmf([0.0, 0.2, 0.3, 0.5]), [1.0, 0.8, 0.5])


def test_survival_from_pmf_rejects_childless_at_index_zero():
    """q_1 = 1: everyone mates at least once. Childlessness lives in f_0, not here."""
    with pytest.raises(ValueError, match="pmf\\[0\\] must be 0"):
        survival_from_pmf([0.1, 0.4, 0.5])


def test_survival_from_pmf_requires_normalized():
    with pytest.raises(ValueError, match="sum to 1"):
        survival_from_pmf([0.0, 0.5, 0.2])


def test_normalize_survival_defaults_to_monogamy():
    q = normalize_survival()
    assert np.allclose(q, [1.0])
    assert mean_n_mates(q) == pytest.approx(1.0)
    assert mean_offspring_per_pair(q) == pytest.approx(2.0)


def test_normalize_survival_validations():
    with pytest.raises(ValueError, match="q_1 must equal 1"):
        normalize_survival(q=[0.9, 0.5])
    with pytest.raises(ValueError, match="non-increasing"):
        normalize_survival(q=[1.0, 0.3, 0.6])
    with pytest.raises(ValueError, match="only one"):
        normalize_survival(q=[1.0], pmf=[0.0, 1.0])


def test_normalize_survival_drops_trailing_zeros():
    assert np.allclose(normalize_survival(q=[1.0, 0.4, 0.0, 0.0]), [1.0, 0.4])


def test_kappa_and_mean_offspring_relation():
    """D-0005: m = 2 / kappa is the only constraint on the offspring pmf."""
    for q2 in (0.0, 0.25, 0.5, 1.0):
        q = normalize_survival(q=[1.0, q2]) if q2 > 0 else normalize_survival()
        kappa = mean_n_mates(q)
        assert mean_offspring_per_pair(q) == pytest.approx(2.0 / kappa)


# ---------------------------------------------------------------------------
# Monogamy must be reproduced exactly
# ---------------------------------------------------------------------------

def test_monogamy_gives_one_mate_each(pop_inputs):
    """Acceptance criterion: q = [1] reproduces the current monogamous behaviour."""
    N, sex, y = pop_inputs
    g = build_mate_graph(N, sex, y, rho_y=0.5, q=[1.0])
    assert g.U == N // 2
    assert np.all(g.n_mates() == 1)
    assert g.count_cycles() == 0
    assert g.n_rounds == 1


def test_monogamy_spouse_array_round_trip(pop_inputs):
    N, sex, y = pop_inputs
    g = build_mate_graph(N, sex, y, rho_y=0.5, q=[1.0])
    spouse = g.to_spouse_array()
    assert spouse.shape == (N,)
    assert np.all(spouse >= 0)
    # spouse relation is symmetric and fixed-point free
    assert np.all(spouse[spouse] == np.arange(N))
    assert np.all(spouse != np.arange(N))


# ---------------------------------------------------------------------------
# Non-monogamy: mate counts and pair counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q2", [0.2, 0.4, 0.6, 1.0])
def test_pair_count_matches_kappa(pop_inputs, q2):
    N, sex, y = pop_inputs
    g = build_mate_graph(N, sex, y, rho_y=0.5, q=[1.0, q2])
    kappa = 1.0 + q2
    assert g.U == pytest.approx(kappa * N / 2, rel=0.02)
    assert g.n_mates().mean() == pytest.approx(kappa, rel=0.02)
    assert g.n_mates().max() <= 2


def test_three_rounds(pop_inputs):
    N, sex, y = pop_inputs
    g = build_mate_graph(N, sex, y, rho_y=0.4, q=[1.0, 0.5, 0.25])
    kappa = 1.75
    assert g.U == pytest.approx(kappa * N / 2, rel=0.03)
    assert g.n_mates().max() <= 3
    assert g.n_rounds == 3


def test_pmf_input_matches_equivalent_survival(pop_inputs):
    N, sex, y = pop_inputs
    g = build_mate_graph(N, sex, y, rho_y=0.3, pmf=[0.0, 0.5, 0.5])
    # pmf [0, .5, .5] -> q = [1, .5] -> kappa 1.5
    assert g.U == pytest.approx(1.5 * N / 2, rel=0.03)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q2", [0.3, 0.6, 1.0])
def test_mate_graph_is_a_forest_when_requested(pop_inputs, q2):
    """No-mating-loops: the mate graph must be acyclic, which makes chains unique."""
    N, sex, y = pop_inputs
    g = build_mate_graph(N, sex, y, rho_y=0.5, q=[1.0, q2], forbid_cycles=True)
    assert g.count_cycles() == 0
    assert not g.has_cycle()


def test_cycles_appear_when_constraint_is_off(pop_inputs):
    """With the forest constraint off, cycles appear -- and are ~log N, not many."""
    N, sex, y = pop_inputs
    g = build_mate_graph(N, sex, y, rho_y=0.5, q=[1.0, 1.0], forbid_cycles=False)
    n_cycles = g.count_cycles()
    assert n_cycles > 0
    # union of matchings -> cycles behave like random-permutation cycles, so ~log N
    assert n_cycles < 20 * np.log(N)


def test_no_self_pairing_and_no_repeat_mates(pop_inputs):
    N, sex, y = pop_inputs
    g = build_mate_graph(N, sex, y, rho_y=0.5, q=[1.0, 1.0], forbid_repeat_mates=True)
    assert np.all(g.edges[:, 0] != g.edges[:, 1])
    pairs = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in g.edges}
    assert len(pairs) == g.U


def test_relative_constraint_is_respected(pop_inputs):
    """Outbred-mating-chain: close blood relatives must not be matched."""
    N, sex, y = pop_inputs

    # declare anyone in the same block of 4 indices a close relative
    def relative_of(i, j):
        return (i // 4) == (j // 4)

    g = build_mate_graph(N, sex, y, rho_y=0.5, q=[1.0], relative_of=relative_of)
    assert not any((int(a) // 4) == (int(b) // 4) for a, b in g.edges)


def test_edges_are_female_male_ordered(pop_inputs):
    N, sex, y = pop_inputs
    g = build_mate_graph(N, sex, y, rho_y=0.5, q=[1.0, 0.5])
    assert np.all(sex[g.edges[:, 0]] == 0)
    assert np.all(sex[g.edges[:, 1]] == 1)


# ---------------------------------------------------------------------------
# Assortment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho_y", [0.0, 0.3, 0.6])
def test_realized_rho_y_tracks_target_in_every_round(pop_inputs, rho_y):
    N, sex, y = pop_inputs
    g = build_mate_graph(N, sex, y, rho_y=rho_y, q=[1.0, 0.6])
    realized = g.round_rho_y[~np.isnan(g.round_rho_y)]
    assert len(realized) == 2
    # rank matching lands close to but not exactly at the target
    assert np.all(np.abs(realized - rho_y) < 0.08)


def test_random_mating_gives_no_assortment(pop_inputs):
    N, sex, y = pop_inputs
    g = build_mate_graph(N, sex, y, rho_y=0.0, q=[1.0])
    ya, yb = y[g.edges[:, 0]], y[g.edges[:, 1]]
    assert np.corrcoef(ya, yb)[0, 1] == pytest.approx(0.0, abs=0.06)


# ---------------------------------------------------------------------------
# The key mechanism-fidelity check (D-0006)
# ---------------------------------------------------------------------------

def test_mate_chain_correlation_is_near_rho_y_squared(pop_inputs):
    """eq:chain-y at N_mu = 2: Corr(y_j, y_k) should EMERGE as rho_y^2.

    Two mates of the same individual come from different rounds carrying independent
    noise, so this is a prediction of the rounds mechanism rather than something
    imposed. Tolerance is loose because the triples are not independent (individuals
    are reused across triples), which inflates the sampling error well beyond
    1/sqrt(n_triples).
    """
    N, sex, y = pop_inputs
    g = build_mate_graph(N, sex, y, rho_y=0.6, q=[1.0, 1.0])
    out = measure_mate_chain_correlation(g, y)
    assert out['n_triples'] > 100
    assert out['corr'] == pytest.approx(out['rho_y_squared'], abs=0.06)


def test_mate_chain_correlation_needs_multiple_mates(pop_inputs):
    N, sex, y = pop_inputs
    g = build_mate_graph(N, sex, y, rho_y=0.5, q=[1.0])  # monogamous -> no triples
    out = measure_mate_chain_correlation(g, y)
    assert out['n_triples'] == 0
    assert np.isnan(out['corr'])


# ---------------------------------------------------------------------------
# MateGraph mechanics
# ---------------------------------------------------------------------------

def test_mates_of_is_consistent_with_edges():
    edges = np.array([[0, 1], [0, 3], [2, 3]], dtype=np.int32)
    g = MateGraph(edges=edges, N=4, round_id=np.array([0, 1, 0]))
    assert sorted(g.mates_of(0).tolist()) == [1, 3]
    assert sorted(g.mates_of(3).tolist()) == [0, 2]
    assert g.mates_of(1).tolist() == [0]
    assert g.n_mates().tolist() == [2, 1, 1, 2]
    assert g.U == 3


def test_count_cycles_detects_a_square():
    # 0-1, 1-2, 2-3, 3-0 is one cycle
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    g = MateGraph(edges=edges, N=4, round_id=np.zeros(4))
    assert g.count_cycles() == 1
    assert g.has_cycle()


def test_count_cycles_zero_for_a_tree():
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    g = MateGraph(edges=edges, N=4, round_id=np.zeros(3))
    assert g.count_cycles() == 0
    assert not g.has_cycle()

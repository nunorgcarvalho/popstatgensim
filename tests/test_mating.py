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


# ---------------------------------------------------------------------------
# Integration: the mate graph driving an actual Population
# ---------------------------------------------------------------------------

from collections import defaultdict

from popstatgensim import GeneticEffect, Population
from popstatgensim.simulation.mating import (
    mean_offspring_per_pair,
    offspring_pmf,
    predict_halfsib_fullsib_ratio,
)


def build_am_pop(N=600, M=80, seed=7, q=None, n_offspring_dist="poisson",
                 offspring_f=None, AM_r=0.5):
    np.random.seed(seed)
    pop = Population(N=N, M=M, seed=seed)
    pop.assign_sex()
    pop.add_trait(
        name="y",
        effects={"A": GeneticEffect(var_indep=0.5, M=M, M_causal=M, name="A")},
        var_Eps=0.5,
    )
    pop.set_params(AM_r=AM_r, AM_trait="y", AM_type="phenotypic",
                   mate_q=q, n_offspring_dist=n_offspring_dist,
                   offspring_f=offspring_f)
    return pop


def count_sib_pairs(pop):
    """Returns (n_full_sib_pairs, n_half_sib_pairs) in the current generation."""
    par = np.asarray(pop.relations["parents"])
    by_parent = defaultdict(list)
    for i in range(pop.N):
        by_parent[par[i, 0]].append(i)
        by_parent[par[i, 1]].append(i)
    full = half = 0
    seen = set()
    for kids in by_parent.values():
        for a in range(len(kids)):
            for b in range(a + 1, len(kids)):
                i, j = kids[a], kids[b]
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                shared = len(set(par[i]) & set(par[j]))
                if shared == 2:
                    full += 1
                elif shared == 1:
                    half += 1
    return full, half


def test_monogamy_produces_no_half_sibs():
    """Acceptance criterion: q = [1] with the old defaults reproduces monogamy exactly."""
    pop = build_am_pop(q=None)
    pop.simulate_generations(1)
    full, half = count_sib_pairs(pop)
    assert half == 0
    assert full > 0


@pytest.mark.parametrize("q2", [0.3, 0.6, 1.0])
def test_non_monogamy_produces_half_sibs(q2):
    pop = build_am_pop(q=[1.0, q2])
    pop.simulate_generations(1)
    full, half = count_sib_pairs(pop)
    assert half > 0, "multi-mate individuals must produce half-sibs"
    assert full > 0


@pytest.mark.parametrize("q2", [None, 0.3, 0.6, 1.0])
def test_population_size_is_exactly_N_every_generation(q2):
    """Population size is deterministic, never random (D-0005)."""
    N = 600
    pop = build_am_pop(N=N, q=(None if q2 is None else [1.0, q2]))
    for _ in range(3):
        pop.simulate_generations(1)
        assert pop.N == N
        assert pop.G.shape[0] == N


@pytest.mark.parametrize("q2", [None, 0.5])
def test_population_size_exact_for_arbitrary_f_allocation(q2):
    """The iid+repair mode must also land on exactly N."""
    N = 600
    q = normalize_survival(q=([1.0, q2] if q2 else None))
    f = offspring_pmf("negative_binomial", mean_offspring_per_pair(q), overdispersion=2.0)
    pop = build_am_pop(N=N, q=(None if q2 is None else [1.0, q2]), offspring_f=f)
    pop.simulate_generations(1)
    assert pop.N == N
    alloc = pop._last_offspring_allocation
    assert alloc["counts"].sum() == N
    assert alloc["mode"] == "iid+repair"


def test_mate_pair_count_matches_kappa_in_population():
    N = 600
    for q2 in (0.3, 0.6):
        pop = build_am_pop(N=N, q=[1.0, q2])
        pop.simulate_generations(1)
        U = pop.past[1].relations["mate_edges"].shape[0]
        assert U == pytest.approx((1.0 + q2) * N / 2, rel=0.02)


def test_mate_graph_is_stored_on_the_parent_generation():
    """The mating happened in the parent generation, so the graph belongs there."""
    pop = build_am_pop(q=[1.0, 0.5])
    pop.simulate_generations(1)
    assert pop.past[1].relations["mate_edges"].shape[0] > 0
    # the offspring generation has not mated yet
    assert pop.relations.get("mate_edges", np.zeros((0, 2))).shape[0] == 0


@pytest.mark.parametrize("q2", [0.3, 0.6, 1.0])
def test_halfsib_fullsib_ratio_tracks_the_design_formula(q2):
    """The design formula is an approximation -- verified empirically, not trusted."""
    pop = build_am_pop(N=1000, M=100, q=[1.0, q2])
    pop.simulate_generations(1)
    full, half = count_sib_pairs(pop)
    q = normalize_survival(q=[1.0, q2])
    f = offspring_pmf("poisson", mean_offspring_per_pair(q))
    predicted = predict_halfsib_fullsib_ratio(q, f)
    observed = half / full
    assert observed == pytest.approx(predicted, rel=0.15)


def test_full_sibs_relation_still_identifies_full_sibships():
    """relations['full_sibs'] is the mate-pair id, which generalizes correctly."""
    pop = build_am_pop(q=[1.0, 0.6])
    pop.simulate_generations(1)
    par = np.asarray(pop.relations["parents"])
    fam = np.asarray(pop.relations["full_sibs"])
    # same family id must mean both parents shared
    for fid in np.unique(fam[fam >= 0])[:20]:
        members = np.flatnonzero(fam == fid)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                assert set(par[members[a]]) == set(par[members[b]])
    # and _get_full_sibships must still work
    sibships = pop._get_full_sibships()
    assert sum(len(s) for s in sibships) == pop.N


def test_spouse_array_still_available_for_monogamous_consumers():
    pop = build_am_pop(q=[1.0, 0.5])
    pop.simulate_generations(1)
    spouse = np.asarray(pop.past[1].relations["spouses"])
    assert spouse.shape == (pop.past[1].N,)
    paired = spouse >= 0
    # symmetric where defined
    assert np.all(spouse[spouse[paired]] == np.flatnonzero(paired))


def test_flatten_generations_shifts_mate_edges():
    """flatten_generations must carry the mate graph across, shifted per generation."""
    pop = build_am_pop(N=400, M=60, q=[1.0, 0.5])
    pop.set_params(keep_past_generations=2, track_pedigree=True)
    pop.simulate_generations(2)
    flat = pop.flatten_generations(generations=2)

    edges = np.asarray(flat.relations["mate_edges"])
    assert edges.shape[0] > 0
    # every index must be in range for the combined population
    assert edges.min() >= 0 and edges.max() < flat.N
    # rounds carry across with matching length
    assert flat.relations["mate_round_id"].shape[0] == edges.shape[0]


def test_subset_individuals_drops_mate_edges_with_a_removed_partner():
    pop = build_am_pop(N=400, M=60, q=[1.0, 0.5])
    pop.simulate_generations(1)
    parent = pop.past[1]
    n_before = parent.relations["mate_edges"].shape[0]
    assert n_before > 0

    keep = np.arange(0, parent.N, 2)  # keep half
    subset = parent.subset_individuals(keep)
    edges = np.asarray(subset.relations["mate_edges"])
    # dropping half the individuals must drop edges, and every survivor index is valid
    assert edges.shape[0] < n_before
    if edges.shape[0] > 0:
        assert edges.min() >= 0 and edges.max() < subset.N


def test_mate_correlations_use_all_mates_not_just_the_first():
    """Reading the lossy spouse array would drop every non-first mate pair.

    Under non-monogamy that would bias the measured mate correlations toward round 1,
    so the measurement must read the full mate edge list.
    """
    from popstatgensim.estimation import measure_mate_correlations
    from popstatgensim.estimation.am_equilibrium import _mate_pairs

    pop = build_am_pop(N=600, M=80, q=[1.0, 0.8])
    pop.simulate_generations(1)
    parent = pop.past[1]

    i, j = _mate_pairs(parent)
    n_edges = parent.relations["mate_edges"].shape[0]
    assert len(i) == n_edges

    # the spouse array keeps strictly fewer pairs
    spouse = np.asarray(parent.relations["spouses"], dtype=np.int32)
    n_spouse_pairs = int(np.sum((spouse >= 0) & (np.arange(parent.N) < spouse)))
    assert n_spouse_pairs < n_edges

    out = measure_mate_correlations(parent, "y")
    assert out["n_mate_pairs"] == n_edges


def test_n_mates_never_exceeds_the_number_of_rounds():
    """Regression: repair once re-used an already-matched partner, giving 2 mates in 1 round."""
    N, sex, y = 2000, None, None
    rng = np.random.default_rng(3)
    sex = np.zeros(N, dtype=int); sex[N // 2:] = 1; sex = rng.permutation(sex)
    y = rng.normal(size=N); y = (y - y.mean()) / y.std()
    for q in ([1.0], [1.0, 0.2], [1.0, 0.6], [1.0, 0.5, 0.25]):
        g = build_mate_graph(N, sex, y, rho_y=0.5, q=q)
        assert g.n_mates().max() <= len(q), f"q={q} produced {g.n_mates().max()} mates"


def test_round_rho_y_is_measured_per_round():
    N = 4000
    rng = np.random.default_rng(11)
    sex = np.zeros(N, dtype=int); sex[N // 2:] = 1; sex = rng.permutation(sex)
    y = rng.normal(size=N); y = (y - y.mean()) / y.std()
    g = build_mate_graph(N, sex, y, rho_y=0.5, q=[1.0, 0.6])
    out = measure_mate_chain_correlation(g, y)
    assert out["round_rho_y"].shape[0] == 2
    assert np.all(np.abs(out["round_rho_y"] - 0.5) < 0.08)


def test_per_round_product_is_the_correct_chain_target():
    """eq:chain-y's rho_y^2 assumes every round realizes the SAME rho_y.

    Round k > 1 matches inside a pool of q_k*N/2 rather than N/2, and the constraint
    rejections concentrate there, so its realized rho_y can attenuate. The product of
    the measured per-round correlations is the right target; comparing against the
    nominal square would charge that calibration difference to the coupling.
    """
    N, REPS = 4000, 15

    def mean_round_ratio(q2):
        """Mean rho_y(round 2) / rho_y(round 1) over several draws.

        Averaged deliberately: the attenuation is a few percent while the per-draw
        spread is larger, so a single draw says nothing about the direction.
        """
        ratios = []
        for r in range(REPS):
            rng = np.random.default_rng(500 + r)
            np.random.seed(500 + r)
            sex = np.zeros(N, dtype=int); sex[N // 2:] = 1
            sex = rng.permutation(sex)
            y = rng.normal(size=N); y = (y - y.mean()) / y.std()
            out = measure_mate_chain_correlation(
                build_mate_graph(N, sex, y, rho_y=0.5, q=[1.0, q2]), y)
            ratios.append(out["round_rho_y"][1] / out["round_rho_y"][0])
        return float(np.mean(ratios))

    small_pool = mean_round_ratio(0.3)
    large_pool = mean_round_ratio(1.0)

    # attenuation is a pool-size effect: worse in the smaller round-2 pool
    assert small_pool < large_pool
    # and it is mild -- a few percent, not a broken mechanism
    assert 0.90 < small_pool < 1.0
    # with a full-size round 2 the two targets converge
    assert large_pool == pytest.approx(1.0, abs=0.02)


def test_chain_correlation_matches_both_targets_within_error():
    """The headline mechanism check, against both targets."""
    N = 4000
    rng = np.random.default_rng(13)
    sex = np.zeros(N, dtype=int); sex[N // 2:] = 1; sex = rng.permutation(sex)
    y = rng.normal(size=N); y = (y - y.mean()) / y.std()
    out = measure_mate_chain_correlation(
        build_mate_graph(N, sex, y, rho_y=0.7, q=[1.0, 0.6]), y)
    assert out["corr"] == pytest.approx(out["rho_y_product"], rel=0.10)
    assert out["corr"] == pytest.approx(out["rho_y_squared"], rel=0.10)


def test_dropped_pairs_are_recorded_not_silent():
    """D-0013 theme: small-and-silent becomes large-and-silent without anyone noticing."""
    N = 2000
    rng = np.random.default_rng(21)
    sex = np.zeros(N, dtype=int); sex[N // 2:] = 1; sex = rng.permutation(sex)
    y = rng.normal(size=N); y = (y - y.mean()) / y.std()
    g = build_mate_graph(N, sex, y, rho_y=0.5, q=[1.0, 1.0])
    assert g.dropped is not None
    assert g.dropped.shape[0] == 2
    assert g.n_dropped == int(g.dropped.sum())
    # it should stay tiny -- flag loudly if this ever grows
    assert g.n_dropped < 0.01 * g.U


def test_get_spouse_corr_raises_under_non_monogamy():
    """D-0013: it must refuse rather than return a first-mate-only figure."""
    pop = build_am_pop(N=600, M=60, q=[1.0, 0.8])
    pop.simulate_generations(1)
    with pytest.raises(Exception, match="monogamy-only"):
        pop.past[1].get_spouse_corr("y")


def test_get_spouse_corr_still_works_under_monogamy():
    pop = build_am_pop(N=600, M=60, q=None)
    pop.simulate_generations(1)
    r = pop.past[1].get_spouse_corr("y")
    assert 0.2 < r < 0.8  # AM_r = 0.5


def test_measure_mate_correlations_is_the_multi_mate_replacement():
    from popstatgensim.estimation import measure_mate_correlations
    pop = build_am_pop(N=600, M=60, q=[1.0, 0.8])
    pop.simulate_generations(1)
    out = measure_mate_correlations(pop.past[1], "y")
    assert out["n_mate_pairs"] == pop.past[1].relations["mate_edges"].shape[0]
    assert 0.2 < out["rho_y"] < 0.8

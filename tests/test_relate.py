"""Tests for the local pedigree graph, on-demand relate(), and class enumeration."""

import numpy as np
import pytest

from popstatgensim import GeneticEffect, Population
from popstatgensim.pedigree.graph import PedigreeGraph
from popstatgensim.pedigree.paths import PathSignature, predicted_cov
from popstatgensim.pedigree.relate import (
    VALID_CLASS_SPECS,
    enumerate_class,
    relate,
    signature_name,
)

RHO_Y, RHO_G, V_A, V_Y = 0.50, 0.28, 0.68, 1.18
# D-0014: these fixtures use q2 = 0.8, i.e. two matching rounds, so every
# relate() call must state how multi-round chains are scored. "pooled" is the
# writeup's literal form and is what these structural tests want.
POOLED = "pooled"


def build_pop(N=600, M=60, gens=3, q2=0.8, seed=5):
    np.random.seed(seed)
    pop = Population(N=N, M=M, seed=seed)
    pop.assign_sex()
    pop.add_trait(
        name="y",
        effects={"A": GeneticEffect(var_indep=0.5, M=M, M_causal=M, name="A")},
        var_Eps=0.5,
    )
    pop.set_params(AM_r=0.5, AM_trait="y", AM_type="phenotypic",
                   mate_q=[1.0, q2], keep_past_generations=gens,
                   n_offspring_dist="poisson")
    pop.simulate_generations(gens)
    return pop


@pytest.fixture(scope="module")
def graph():
    return PedigreeGraph.from_population(build_pop())


# ---------------------------------------------------------------------------
# The graph itself
# ---------------------------------------------------------------------------

def test_graph_is_local_and_linear_in_size(graph):
    """Only O(N) per generation is stored -- no all-pairs structure (D-0007)."""
    assert graph.parents.shape == (graph.N, 2)
    assert graph.generation.shape == (graph.N,)
    assert graph.mate_pair_id.shape == (graph.N,)
    # the mate edge list is O(kappa N / 2), not O(N^2)
    assert graph.mate_edges.shape[0] < graph.N


def test_graph_generations_are_ordered_oldest_first(graph):
    assert graph.generation.min() == 0
    assert graph.generation.max() >= 1
    # a child is always exactly one generation younger than its parents
    has_par = graph.parents[:, 0] >= 0
    child_gen = graph.generation[has_par]
    par_gen = graph.generation[graph.parents[has_par, 0]]
    assert np.all(child_gen == par_gen + 1)


def test_offspring_index_is_the_inverse_of_parents(graph):
    for child in range(0, graph.N, 97):
        for p in graph.parents_of(child):
            assert child in [int(x) for x in graph.offspring_of(int(p))]


def test_mate_graph_is_a_forest(graph):
    """No-mating-loops is what makes the mating chain unique."""
    assert graph.mate_graph_is_forest()


def test_mating_chain_is_symmetric_and_counts_edges(graph):
    edge = graph.mate_edges[0]
    a, b = int(edge[0]), int(edge[1])
    chain = graph.mating_chain(a, b)
    assert chain is not None
    assert chain["N_mu"] == 1
    assert set(chain["members"]) == {a, b}
    back = graph.mating_chain(b, a)
    assert back["N_mu"] == 1


def test_mating_chain_to_self_is_zero(graph):
    chain = graph.mating_chain(0, 0)
    assert chain["N_mu"] == 0


def test_dropping_generations_truncates_rather_than_corrupting():
    """Local-only state means a path through a dropped generation is simply unreachable."""
    deep = PedigreeGraph.from_population(build_pop(gens=3))
    shallow = PedigreeGraph.from_population(build_pop(gens=1))
    assert shallow.generation.max() < deep.generation.max()
    # the oldest retained generation has no parents recorded
    oldest = shallow.generation == 0
    assert np.all(shallow.parents[oldest] == -1)


# ---------------------------------------------------------------------------
# enumerate_class
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", VALID_CLASS_SPECS)
def test_every_class_is_enumerable(graph, spec):
    pairs = list(enumerate_class(graph, spec, limit=50))
    assert len(pairs) > 0, f'{spec} produced no pairs'
    for a, b in pairs:
        assert 0 <= a < graph.N and 0 <= b < graph.N
        assert a != b


def test_enumerate_respects_limit(graph):
    assert len(list(enumerate_class(graph, "full_sibs", limit=7))) == 7


def test_enumerate_rejects_unknown_spec(graph):
    with pytest.raises(ValueError, match="spec"):
        list(enumerate_class(graph, "not_a_class"))


def test_full_sibs_share_both_parents(graph):
    for a, b in list(enumerate_class(graph, "full_sibs", limit=40)):
        assert set(graph.parents_of(a)) == set(graph.parents_of(b))


def test_half_sibs_share_exactly_one_parent(graph):
    for a, b in list(enumerate_class(graph, "half_sibs", limit=40)):
        shared = set(graph.parents_of(a)) & set(graph.parents_of(b))
        assert len(shared) == 1, (a, b, shared)


def test_step_sibs_share_no_parent(graph):
    """Step sibs are linked only through the mating graph, never by blood."""
    for a, b in list(enumerate_class(graph, "step_sibs", limit=40)):
        assert not (set(graph.parents_of(a)) & set(graph.parents_of(b)))


def test_parent_parent_pairs_are_mates(graph):
    mate_set = {(min(int(i), int(j)), max(int(i), int(j))) for i, j in graph.mate_edges}
    for a, b in list(enumerate_class(graph, "parent_parent", limit=40)):
        assert (min(a, b), max(a, b)) in mate_set


def test_parent_offspring_pairs_really_are(graph):
    for a, b in list(enumerate_class(graph, "parent_offspring", limit=40)):
        assert a in [int(x) for x in graph.parents_of(b)]


# ---------------------------------------------------------------------------
# relate: every class must get the writeup's own coefficient
# ---------------------------------------------------------------------------

EXPECTED_SIGNATURES = {
    "full_sibs": PathSignature(2, 2, ((1, 2),)),
    "half_sibs": PathSignature(2, 2, ((2, 2),)),
    "step_sibs": PathSignature(2, 2, ((3, 2),)),
    "parent_offspring": PathSignature(1, 0, (), "source", "sink"),
    "grandparent": PathSignature(2, 0, (), "source", "sink"),
    "avuncular": PathSignature(3, 2, ((1, 2),)),
    "first_cousins": PathSignature(4, 2, ((1, 2),)),
    "parent_parent": PathSignature(0, 0, ((1, 0),), "terminal_mating", "terminal_mating"),
    "married_couple_parents": PathSignature(2, 0, ((1, 0),), "source", "source"),
}


@pytest.mark.parametrize("spec", sorted(EXPECTED_SIGNATURES))
def test_relate_reproduces_the_writeup_coefficient_for_each_class(graph, spec):
    """All the named rows of the writeup's table: enumerated, signed, and predicted."""
    expected_sig = EXPECTED_SIGNATURES[spec]
    expected_cov = predicted_cov(expected_sig, V_A, RHO_Y, RHO_G, phenotypic=True)

    pairs = list(enumerate_class(graph, spec, limit=10))
    assert pairs, spec
    hits = 0
    for a, b in pairs:
        r = relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, rho_y_by_round=POOLED)
        if r.signature is not None and r.signature.key == expected_sig.key:
            assert r.cov == pytest.approx(expected_cov)
            hits += 1
    assert hits > 0, f'{spec}: no pair got the expected signature'


@pytest.mark.parametrize("spec,expected_name", [
    ("full_sibs", "full siblings"),
    ("half_sibs", "half siblings"),
    ("step_sibs", "step siblings"),
    ("parent_parent", "parent--parent (mates)"),
])
def test_relate_names_the_named_classes(graph, spec, expected_name):
    """At least one enumerated pair of each class must get the conventional name.

    Scanning rather than taking the first pair is deliberate: a pair enumerated as step
    sibs may ALSO be joined by a shorter, stronger path -- two people who share a mate
    are co-mates, which dominates the step-sib path. relate() reporting the dominant
    relationship there is correct, not a misclassification.
    """
    found = [relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, rho_y_by_round=POOLED).name
             for a, b in enumerate_class(graph, spec, limit=40)]
    assert expected_name in found, f'{spec}: got {set(found)}'


def test_sibling_classes_are_ordered_full_above_half_above_step(graph):
    """They differ only in N_mu within the single mating chain -- 1, 2, 3."""
    covs = {}
    for spec in ("full_sibs", "half_sibs", "step_sibs"):
        sig = EXPECTED_SIGNATURES[spec]
        covs[spec] = predicted_cov(sig, V_A, RHO_Y, RHO_G, phenotypic=False)
    assert covs["full_sibs"] > covs["half_sibs"] > covs["step_sibs"]


def test_unrelated_pair_reports_unrelated(graph):
    """Two founders in the oldest generation with no mating chain between them."""
    oldest = np.flatnonzero(graph.generation == 0)
    found = False
    for a in oldest[:40]:
        for b in oldest[:40]:
            if a == b:
                continue
            if graph.mating_chain(int(a), int(b)) is None:
                r = relate(graph, int(a), int(b), rho_y=RHO_Y, rho_g=RHO_G,
                           V_A=V_A, V_Y=V_Y, rho_y_by_round=POOLED)
                if not r.related:
                    assert r.cov == 0.0
                    assert r.name == "unrelated"
                    found = True
                    break
        if found:
            break
    assert found, "expected at least one genuinely unrelated pair"


def test_self_pair_is_V_A(graph):
    r = relate(graph, 5, 5, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, rho_y_by_round=POOLED)
    assert r.name == "self"
    assert r.cov == pytest.approx(V_A)


def test_relate_is_symmetric(graph):
    for spec in ("full_sibs", "half_sibs", "avuncular"):
        a, b = next(iter(enumerate_class(graph, spec, limit=1)))
        fwd = relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, rho_y_by_round=POOLED)
        rev = relate(graph, b, a, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, rho_y_by_round=POOLED)
        assert fwd.cov == pytest.approx(rev.cov)


def test_correlation_floor_prunes(graph):
    """The floor is on the phenotypic correlation, not a generation horizon."""
    a, b = next(iter(enumerate_class(graph, "first_cousins", limit=1)))
    loose = relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, floor=1e-6, rho_y_by_round=POOLED)
    strict = relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, floor=0.9, rho_y_by_round=POOLED)
    assert loose.related
    assert not strict.related


def test_return_all_is_opt_in_and_lists_candidates(graph):
    a, b = next(iter(enumerate_class(graph, "full_sibs", limit=1)))
    default = relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, rho_y_by_round=POOLED)
    assert default.all_paths is None
    diag = relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y,
                  return_all=True, rho_y_by_round=POOLED)
    assert diag.all_paths is not None and len(diag.all_paths) >= 1
    # the dominant path must be the first one popped
    assert diag.all_paths[0][0].key == diag.signature.key


def test_label_and_prediction_are_reported_separately(graph):
    """Labeling is parameter-free; prediction uses measured rho_y/rho_g. Disagreements
    are flagged rather than silently resolved."""
    a, b = next(iter(enumerate_class(graph, "full_sibs", limit=1)))
    r = relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, rho_y_by_round=POOLED)
    assert r.label_name is not None
    assert isinstance(r.label_agrees, bool)


def test_rho_y_by_round_changes_the_prediction_only_when_supplied(graph):
    """mating_round_ids are ROUND INDICES; they become rho_y only via the mapping."""
    a, b = next(iter(enumerate_class(graph, "half_sibs", limit=1)))
    pooled = relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, rho_y_by_round=POOLED)
    per_round = relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y,
                       rho_y_by_round={0: RHO_Y, 1: RHO_Y})
    # supplying the SAME value for each round must reproduce the pooled prediction
    assert per_round.cov == pytest.approx(pooled.cov)
    # halving EVERY round's rho_y must move the prediction, whichever rounds the
    # chain happens to cross
    attenuated = relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y,
                        rho_y_by_round={0: 0.5 * RHO_Y, 1: 0.5 * RHO_Y})
    assert attenuated.cov != pytest.approx(pooled.cov)


# ---------------------------------------------------------------------------
# Exactness: relate() vs exhaustive enumeration
# ---------------------------------------------------------------------------

def test_dominant_path_equals_the_best_of_all_candidates(graph):
    """The first complete path popped must be the true maximum over all candidates.

    This is the acceptance criterion that relate() agrees with exhaustive search: with
    return_all=True every candidate above the floor is listed, so the dominant one can
    be checked against the whole set rather than trusted.
    """
    checked = 0
    for spec in ("full_sibs", "half_sibs", "step_sibs", "avuncular", "first_cousins"):
        for a, b in list(enumerate_class(graph, spec, limit=6)):
            r = relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y,
                       return_all=True, floor=1e-9, rho_y_by_round=POOLED)
            if not r.related:
                continue
            best = max(c for _, c in r.all_paths)
            assert r.corr == pytest.approx(best), (spec, a, b)
            checked += 1
    assert checked > 10


def test_per_query_cost_does_not_blow_up_with_N():
    """The N^2 structure is gone: per-query cost is flat in N, build is linear."""
    import time
    timings = {}
    for N in (400, 1600):
        g = PedigreeGraph.from_population(build_pop(N=N, M=40, gens=2))
        pairs = list(enumerate_class(g, "full_sibs", limit=60))
        t0 = time.time()
        for a, b in pairs:
            relate(g, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, rho_y_by_round=POOLED)
        timings[N] = (time.time() - t0) / len(pairs)
    # a 4x increase in N must not produce anything like a 16x increase per query
    assert timings[1600] < 4 * timings[400]


# ---------------------------------------------------------------------------
# Hand-built pedigrees: the classes too rare to sample at feasible N
# ---------------------------------------------------------------------------

def hand_built_mating_chain(n_matings: int):
    """A pedigree whose two probands are joined by exactly ``n_matings`` matings.

    Builds a mate chain P0 -- P1 -- ... -- Pn, gives the FIRST mating a child A and the
    LAST a child B, so A and B are joined by one mating chain of length ``n_matings``
    and by no blood path at all. n_matings = 1/2/3/4 gives full / half / step /
    step-step siblings respectively.

    Step-step sibs need four matings on one chain and are effectively unsampleable from
    a random population at feasible N -- the task anticipates this and asks that their
    signature and prediction be checked on a hand-built pedigree instead.
    """
    n_parents = n_matings + 1
    N = n_parents + 2  # the chain, plus the two probands
    A, B = n_parents, n_parents + 1

    parents = np.full((N, 2), -1, dtype=np.int32)
    parents[A] = [0, 1]                            # child of the first mating
    parents[B] = [n_parents - 2, n_parents - 1]    # child of the last mating

    mate_edges = np.array([[k, k + 1] for k in range(n_matings)], dtype=np.int32)
    generation = np.zeros(N, dtype=np.int32)
    generation[[A, B]] = 1

    mate_pair_id = np.full(N, -1, dtype=np.int32)
    mate_pair_id[A] = 0
    mate_pair_id[B] = n_matings - 1

    return PedigreeGraph(
        N=N, parents=parents, generation=generation, mate_edges=mate_edges,
        mate_round_id=np.zeros(n_matings, dtype=np.int32),
        mate_pair_id=mate_pair_id,
    ), A, B


@pytest.mark.parametrize("n_matings,expected_name", [
    (1, "full siblings"),
    (2, "half siblings"),
    (3, "step siblings"),
    (4, "step-step siblings"),
])
def test_sibling_ladder_on_hand_built_pedigrees(n_matings, expected_name):
    """The four sibling classes differ only in N_mu within one mating chain: 1,2,3,4."""
    g, A, B = hand_built_mating_chain(n_matings)
    r = relate(g, A, B, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, floor=1e-12, rho_y_by_round=POOLED)
    assert r.related
    assert r.signature.mating_chains == ((n_matings, 2),)
    assert r.signature.sum_N_lin == 2
    assert r.name == expected_name
    expected = predicted_cov(PathSignature(2, 2, ((n_matings, 2),)),
                             V_A, RHO_Y, RHO_G, phenotypic=True)
    assert r.cov == pytest.approx(expected)


def test_step_step_sibs_prediction_matches_the_closed_form():
    """Step-step sibs: N_mu = 4, so w = (1/4)[c(4) + 2c(3) + c(2)]."""
    g, A, B = hand_built_mating_chain(4)
    r = relate(g, A, B, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, floor=1e-12, rho_y_by_round=POOLED)
    w = 0.25 * (RHO_G * RHO_Y ** 3 + 2 * RHO_G * RHO_Y ** 2 + RHO_G * RHO_Y)
    assert r.cov == pytest.approx(V_A * w)


def test_sibling_ladder_is_monotonically_weaker():
    covs = []
    for n in (1, 2, 3, 4):
        g, A, B = hand_built_mating_chain(n)
        covs.append(relate(g, A, B, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y,
                           floor=1e-12, rho_y_by_round=POOLED).cov)
    assert covs[0] > covs[1] > covs[2] > covs[3]


def test_relate_matches_brute_force_on_a_small_pedigree():
    """relate() must agree with exhaustive enumeration where exhaustive is feasible."""
    g, A, B = hand_built_mating_chain(3)
    exhaustive = relate(g, A, B, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y,
                        floor=1e-12, return_all=True, rho_y_by_round=POOLED)
    fast = relate(g, A, B, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, floor=1e-12, rho_y_by_round=POOLED)
    assert fast.signature.key == exhaustive.signature.key
    assert fast.corr == pytest.approx(max(c for _, c in exhaustive.all_paths))


def test_nmu1_no1_cell_is_excluded():
    """(N_mu = 1, N_o = 1): one mating edge feeding one descent IS that descent.

    Admitting it double-counts the meiosis, and the entering proband then sits on a
    terminal mating chain and picks up C = 1/h^2 rather than (1+rho_y)/(1+rho_g). The
    inflation factor is (1+rho_g)/(h^2 (1+rho_y)), about 1.48 at h^2 = 0.58 -- which
    turns parent-offspring from 0.43 into 0.66. Best-first search PREFERS the inflated
    route, so excluding it is load-bearing rather than tidy.

    The printed rules in the writeup admit the configuration; the exclusion matches the
    writeup's own make_figures.py, which skips the same cell.
    """
    g, A, B = hand_built_mating_chain(1)   # A and B are full sibs via parents 0 -- 1
    # parent 0 and child B: a genuine parent-offspring pair inside the same pedigree
    r = relate(g, 0, B, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y, floor=1e-12, rho_y_by_round=POOLED)
    assert r.related
    # must be the lineal answer, not the inflated mating route
    expected = predicted_cov(PathSignature(1, 0, (), "source", "sink"),
                             V_A, RHO_Y, RHO_G, phenotypic=True)
    assert r.cov == pytest.approx(expected)
    assert r.name == "parent-offspring"
    # no candidate anywhere in the search may carry that cell
    diag = relate(g, 0, B, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y,
                  floor=1e-12, return_all=True, rho_y_by_round=POOLED)
    for sig, _ in diag.all_paths:
        assert (1, 1) not in sig.mating_chains


def test_excluded_cell_would_have_won_if_admitted():
    """Quantifies why the exclusion matters: the spurious route scores HIGHER."""
    from popstatgensim.pedigree.paths import chain_weight, proband_correction
    # the route A -- Q -> B: sum_N_lin = 1, N_o = 1, N_mu = 1
    spurious = (V_A * ((1 + RHO_G) / 2) ** (1 - 1)
                * chain_weight(1, 1, RHO_Y, RHO_G)
                * proband_correction("terminal_mating", RHO_Y, RHO_G)
                * proband_correction("sink", RHO_Y, RHO_G))
    correct = predicted_cov(PathSignature(1, 0, (), "source", "sink"),
                            V_A, RHO_Y, RHO_G, phenotypic=True)
    assert spurious > correct
    # the inflation is (1+rho_g)/(h^2 (1+rho_y)) with h^2 = rho_g/rho_y
    h2 = RHO_G / RHO_Y
    assert spurious / correct == pytest.approx((1 + RHO_G) / (h2 * (1 + RHO_Y)))


def test_relate_refuses_multi_round_pedigree_without_a_decision(graph):
    """D-0014: relate states the ambiguity once, up front, not mid-search."""
    a, b = next(iter(enumerate_class(graph, "half_sibs", limit=1)))
    with pytest.raises(ValueError, match="matching rounds"):
        relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y)


def test_relate_accepts_pooled_or_a_mapping(graph):
    a, b = next(iter(enumerate_class(graph, "half_sibs", limit=1)))
    pooled = relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y,
                    rho_y_by_round="pooled")
    mapped = relate(graph, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y,
                    rho_y_by_round={0: RHO_Y, 1: RHO_Y})
    assert pooled.related and mapped.related
    assert pooled.cov == pytest.approx(mapped.cov)


def test_monogamous_pedigree_needs_no_decision():
    """One matching round has no ambiguity, so it must not be made to ask."""
    g = PedigreeGraph.from_population(_monogamous_pop(N=400, M=40, gens=2))
    a, b = next(iter(enumerate_class(g, "full_sibs", limit=1)))
    r = relate(g, a, b, rho_y=RHO_Y, rho_g=RHO_G, V_A=V_A, V_Y=V_Y)
    assert r.name == "full siblings"


def _monogamous_pop(N, M, gens):
    np.random.seed(9)
    pop = Population(N=N, M=M, seed=9)
    pop.assign_sex()
    pop.add_trait(
        name="y",
        effects={"A": GeneticEffect(var_indep=0.5, M=M, M_causal=M, name="A")},
        var_Eps=0.5,
    )
    pop.set_params(AM_r=0.5, AM_trait="y", AM_type="phenotypic",
                   keep_past_generations=gens, n_offspring_dist="poisson")
    pop.simulate_generations(gens)
    return pop


# ---------------------------------------------------------------------------
# Backward compatibility of the PedPath removal
# ---------------------------------------------------------------------------

def test_pedigree_shim_cannot_be_constructed_but_can_be_unpickled():
    """The removed layer's class survives only so old pickles load.

    Population.__init__ used to assign `self.ped = Pedigree(self.N)` for EVERY
    population, tracked or not, so every pre-removal pickle embeds this class and
    deleting it outright would make existing scratch bundles unloadable.
    """
    import pickle
    from popstatgensim.pedigree.pedigree import Pedigree

    with pytest.raises(TypeError, match="removed"):
        Pedigree(10)

    # unpickling does not call __init__, which is what makes the shim work
    obj = Pedigree.__new__(Pedigree)
    obj.__dict__["paths"] = {(0, 1): (1, -1)}
    restored = pickle.loads(pickle.dumps(obj))
    assert restored.__dict__["paths"] == {(0, 1): (1, -1)}
    assert "removed layer" in repr(restored)


def test_population_without_pedigree_graph_attribute_still_works():
    """A Population restored from a pre-removal pickle has no `_pedigree_graph`.

    Unpickling does not call __init__, so the attribute is simply absent and a bare
    `self._pedigree_graph` would raise AttributeError on every restored bundle. The
    property must tolerate that. Caught by the round-trip check, not by reasoning.
    """
    pop = build_pop(N=300, M=40, gens=2)
    del pop.__dict__["_pedigree_graph"]          # simulate the pre-removal shape
    assert "_pedigree_graph" not in pop.__dict__
    g = pop.pedigree_graph                        # must not raise
    assert g.N > 0
    assert pop.pedigree_graph is g                # and caches thereafter


def test_new_populations_carry_no_pedigree_object():
    """Nothing builds a Pedigree any more; pickles get smaller, not larger."""
    pop = build_pop(N=200, M=40, gens=1)
    assert not hasattr(pop, "ped")


def test_pedigree_graph_is_lazy_and_invalidated_on_new_generations():
    pop = build_pop(N=300, M=40, gens=1)
    assert pop.__dict__["_pedigree_graph"] is None      # not built until asked for
    first = pop.pedigree_graph
    assert pop.pedigree_graph is first                  # cached
    pop.simulate_generations(1)
    assert pop.__dict__["_pedigree_graph"] is None      # staled by the new generation
    rebuilt = pop.pedigree_graph
    assert rebuilt is not first                          # and rebuilt, not reused
    # N is capped by keep_past_generations, so it does NOT grow without bound -- the
    # graph spans only the retained generations, which is what makes dropping old ones
    # coherent (D-0007): a path through a dropped generation is simply unreachable.
    assert rebuilt.N == first.N
    assert not np.array_equal(rebuilt.parents, first.parents)

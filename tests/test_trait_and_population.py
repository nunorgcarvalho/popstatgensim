import numpy as np
import pytest

import popstatgensim as psg
from popstatgensim.genome import compute_PCA
from popstatgensim.traits import (
    FixedEffect,
    GeneticEffect,
    RandomEffect,
    simulate_pgs_standardized_weights,
)
from popstatgensim.utils import summarize_matrix_error


def test_population_trait_can_reference_existing_population_trait():
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=0)
    base_values = np.arange(pop.N, dtype=float)
    pop.add_trait_from_fixed_values(name="base", y=base_values)

    effects = {
        "base_copy": FixedEffect(
            name="base_copy",
            beta=2.0,
            input_name="base",
        )
    }
    pop.add_trait(name="y", effects=effects)

    np.testing.assert_allclose(pop.traits["y"].y, 2.0 * base_values)


def test_fixed_effect_raises_when_input_name_is_ambiguous_between_trait_and_input():
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=31)
    pop.add_trait_from_fixed_values(name="base", y=np.arange(pop.N, dtype=float))

    with pytest.raises(ValueError, match="Set is_trait explicitly"):
        pop.add_trait(
            name="y",
            effects={"base_copy": FixedEffect(name="base_copy", beta=1.0, input_name="base")},
            inputs={"base": np.ones(pop.N, dtype=float)},
        )


def test_fixed_effect_can_use_parent_mean_of_same_trait_from_previous_generation():
    pop = psg.Population(
        N=6,
        M=4,
        p_init=0.3,
        seed=32,
        params=psg.PopulationParams(keep_past_generations=1),
    )
    base_values = np.linspace(-1.0, 1.0, pop.N)
    pop.add_trait_from_fixed_values(name="y", y=base_values, trait_type="permanent")
    pop.simulate_generations(generations=1, related_offspring=True, trait_updates=False)

    pop.add_trait(
        name="z",
        effects={
            "PT": FixedEffect(
                name="PT",
                beta=2.0,
                input_name="y",
                relation="parents",
                reduce="mean",
            )
        },
    )

    parent_ids = pop.relations["parents"]
    expected = 2.0 * base_values[parent_ids].mean(axis=1)
    np.testing.assert_allclose(pop.traits["z"].y, expected)


def test_fixed_effect_parent_trait_uses_zeros_when_past_generation_is_unavailable():
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=320)
    pop.add_trait_from_fixed_values(name="y", y=np.linspace(-1.0, 1.0, pop.N), trait_type="permanent")

    with pytest.warns(UserWarning, match="Using zeros for this component"):
        pop.add_trait(
            name="z",
            effects={
                "PT": FixedEffect(
                    name="PT",
                    var=1.5,
                    input_name="y",
                    relation="parents",
                    reduce="mean",
                )
            },
        )

    np.testing.assert_allclose(pop.traits["z"].y, np.zeros(pop.N))


def test_fixed_effect_can_use_parent_component_and_var_scaling():
    pop = psg.Population(
        N=6,
        M=5,
        p_init=0.3,
        seed=33,
        params=psg.PopulationParams(keep_past_generations=1),
    )
    pop.set_params(R_type="blocks")
    x = np.linspace(0.0, 1.0, pop.N)
    pop.add_trait(
        name="base",
        effects={"base_component": FixedEffect(name="base_component", beta=1.0, input_name="x")},
        inputs={"x": x},
        var_Eps=0.2,
    )
    parent_component = pop.traits["base"].y_["base_component"].copy()

    pop.simulate_generations(generations=1, related_offspring=True, trait_updates=False)
    pop.add_trait(
        name="child",
        effects={
            "PT": FixedEffect(
                name="PT",
                var=1.5,
                input_name="base",
                input_component="base_component",
                relation="parents",
                reduce="sum",
            )
        },
    )

    parent_ids = pop.relations["parents"]
    raw = parent_component[parent_ids].sum(axis=1)
    expected = raw * np.sqrt(1.5 / raw.var())
    np.testing.assert_allclose(pop.traits["child"].y, expected)


def test_trait_reports_when_effect_requires_per_generation_updates():
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=333)
    pop.add_trait_from_fixed_values(name="y", y=np.linspace(-1.0, 1.0, pop.N), trait_type="permanent")
    pop.add_trait(
        name="z",
        effects={
            "PT": FixedEffect(
                name="PT",
                beta=1.0,
                input_name="y",
                relation="parents",
                reduce="mean",
            )
        },
    )

    assert pop.traits["z"].requires_per_generation_updates() is True
    assert pop.traits["z"].required_per_generation_update_effects() == ["PT"]


def test_multigeneration_simulation_requires_trait_updates_for_parent_phenotype_effects():
    pop = psg.Population(
        N=6,
        M=4,
        p_init=0.3,
        seed=334,
        params=psg.PopulationParams(keep_past_generations=1),
    )
    pop.add_trait_from_fixed_values(name="y", y=np.linspace(-1.0, 1.0, pop.N), trait_type="permanent")
    pop.add_trait(
        name="z",
        effects={
            "PT": FixedEffect(
                name="PT",
                beta=1.0,
                input_name="y",
                relation="parents",
                reduce="mean",
            )
        },
    )

    with pytest.raises(ValueError, match="require per-generation updates"):
        pop.simulate_generations(generations=2, related_offspring=True, trait_updates=False)


def test_multigeneration_simulation_allows_trait_updates_for_parent_phenotype_effects():
    pop = psg.Population(
        N=6,
        M=4,
        p_init=0.3,
        seed=335,
        params=psg.PopulationParams(keep_past_generations=1),
    )
    pop.add_trait_from_fixed_values(name="y", y=np.linspace(-1.0, 1.0, pop.N), trait_type="permanent")
    pop.add_trait(
        name="z",
        effects={
            "PT": FixedEffect(
                name="PT",
                beta=1.0,
                input_name="y",
                relation="parents",
                reduce="mean",
            )
        },
    )

    pop.simulate_generations(generations=2, related_offspring=True, trait_updates=True)
    assert pop.traits["z"].y.shape == (pop.N,)


def test_population_trait_with_genetic_effect_runs_through_refactored_modules():
    pop = psg.Population(N=8, M=6, p_init=0.4, seed=1)
    effect = GeneticEffect(var_indep=1.0, M=pop.M, M_causal=3, name="A")
    pop.add_trait(name="genetic_trait", effects={"A": effect}, var_Eps=0.5)

    trait = pop.traits["genetic_trait"]
    assert trait.y.shape == (pop.N,)
    assert set(trait.y_.keys()) == {"A", "Eps"}
    assert trait.inputs["G"].shape == (pop.N, pop.M)


def test_generate_causal_effects_accepts_point_mass_alias():
    np.random.seed(10)
    effects, j_causal = psg.traits.generate_causal_effects(
        M=8,
        M_causal=4,
        var_G=2.0,
        dist="point_mass",
    )

    assert j_causal.shape == (4,)
    np.testing.assert_allclose(effects[j_causal], np.sqrt(2.0 / 4))
    assert np.count_nonzero(effects) == 4
    assert np.isclose(np.sum(effects ** 2), 2.0)


def test_generate_genetic_effects_supports_constant_distribution():
    np.random.seed(11)
    effects = psg.traits.generate_genetic_effects(
        var_A=0.8,
        var_A_par=0.2,
        r=1.0,
        M=10,
        M_causal=5,
        dist="constant",
    )

    effect_A = effects["A"].effects_standardized
    effect_A_par = effects["A_par"].effects_standardized
    j_causal_A = effects["A"].j_causal
    j_causal_A_par = effects["A_par"].j_causal

    np.testing.assert_array_equal(j_causal_A, j_causal_A_par)
    np.testing.assert_allclose(effect_A[j_causal_A], np.sqrt(0.8 / 5))
    np.testing.assert_allclose(effect_A_par[j_causal_A], np.sqrt(0.2 / 5))
    assert np.count_nonzero(effect_A) == 5
    assert np.count_nonzero(effect_A_par) == 5
    assert np.isclose(np.sum(effect_A ** 2), 0.8)
    assert np.isclose(np.sum(effect_A_par ** 2), 0.2)


def test_generate_causal_effects_supports_constant_symmetric_distribution():
    np.random.seed(12)
    effects, j_causal = psg.traits.generate_causal_effects(
        M=9,
        M_causal=5,
        var_G=2.0,
        dist="constant_symmetric",
    )

    causal_values = effects[j_causal]
    assert np.sum(causal_values < 0) == 2
    assert np.sum(causal_values > 0) == 3
    np.testing.assert_allclose(np.abs(causal_values), np.full(5, np.sqrt(2.0 / 5)))
    assert np.isclose(np.sum(causal_values ** 2), 2.0)


def test_generate_genetic_effects_supports_constant_symmetric_distribution():
    np.random.seed(13)
    effects = psg.traits.generate_genetic_effects(
        var_A=0.8,
        var_A_par=0.2,
        r=-1.0,
        M=10,
        M_causal=5,
        dist="constant_symmetric",
    )

    effect_A = effects["A"].effects_standardized
    effect_A_par = effects["A_par"].effects_standardized
    j_causal = effects["A"].j_causal

    np.testing.assert_array_equal(j_causal, effects["A_par"].j_causal)
    assert np.sum(effect_A[j_causal] < 0) == 2
    assert np.sum(effect_A[j_causal] > 0) == 3
    np.testing.assert_allclose(np.abs(effect_A[j_causal]), np.full(5, np.sqrt(0.8 / 5)))
    np.testing.assert_allclose(np.abs(effect_A_par[j_causal]), np.full(5, np.sqrt(0.2 / 5)))
    np.testing.assert_allclose(
        np.sign(effect_A_par[j_causal]),
        -np.sign(effect_A[j_causal]),
    )
    assert np.isclose(np.sum(effect_A ** 2), 0.8)
    assert np.isclose(np.sum(effect_A_par ** 2), 0.2)


def test_trait_add_popstrat_adds_cluster_constant_component_after_join():
    pops = [
        psg.Population(N=4, M=5, p_init=0.3, seed=idx)
        for idx in range(3)
    ]
    spop = psg.SuperPopulation(pops)
    spop.add_subpop_trait()
    spop.join_populations()

    joined = spop.pops[-1]
    x = np.linspace(-1.0, 1.0, joined.N)
    joined.add_trait(
        name="y",
        effects={"base": FixedEffect(name="base", beta=1.0, input_name="x")},
        inputs={"x": x},
    )

    trait = joined.traits["y"]
    trait.add_popstrat(variance=2.0, force_var=True)

    popstrat = trait.y_["popstrat"]
    subpop = joined.traits["subpop"].y.astype(int)
    for label in np.unique(subpop):
        values = popstrat[subpop == label]
        np.testing.assert_allclose(values, values[0])
    assert np.isclose(popstrat.var(), 2.0)

    base_component = trait.y_["base"].copy()
    trait.remove_effect("popstrat")
    np.testing.assert_allclose(trait.y, base_component)


def test_join_populations_can_randomly_sample_Ns_before_joining():
    pops = [
        psg.Population(N=6, M=5, p_init=0.3, seed=idx)
        for idx in range(3)
    ]
    spop = psg.SuperPopulation(pops)
    spop.add_subpop_trait()

    np.random.seed(12)
    spop.join_populations(Ns=[2, 4, 2])

    joined = spop.pops[-1]
    assert joined.N == 8
    assert spop.active == [False, False, False, True]
    subpop = joined.traits["subpop"].y.astype(int)
    np.testing.assert_array_equal(
        np.bincount(subpop, minlength=3),
        np.array([2, 4, 2]),
    )


def test_join_populations_can_return_joined_population_object():
    pops = [
        psg.Population(N=4, M=5, p_init=0.3, seed=idx)
        for idx in range(2)
    ]
    spop = psg.SuperPopulation(pops)
    n_before = len(spop.pops)
    active_before = spop.active.copy()
    graph_before = spop.graph.copy()

    joined = spop.join_populations(return_obj=True)

    assert joined.N == 8
    assert len(spop.pops) == n_before
    assert spop.active == active_before
    np.testing.assert_array_equal(spop.graph, graph_before)


def test_join_populations_inherits_first_population_params_and_warns_on_mismatch():
    pops = [
        psg.Population(N=4, M=5, p_init=0.3, seed=idx)
        for idx in range(2)
    ]
    pops[0].set_params(mu=0.02, R_type="uniform")
    pops[1].set_params(mu=0.05, R_type="blocks")
    spop = psg.SuperPopulation(pops)

    with pytest.warns(UserWarning, match="inherit the first population's params"):
        joined = spop.join_populations(return_obj=True)

    assert joined.params is not pops[0].params
    assert joined.params.mu == pops[0].params.mu
    assert joined.params.R_type == pops[0].params.R_type
    np.testing.assert_allclose(joined.params.R, pops[0].params.R)


def test_join_populations_defaults_keep_past_generations_from_first_population():
    pops = [
        psg.Population(
            N=4,
            M=5,
            p_init=0.3,
            seed=idx,
            params=psg.PopulationParams(keep_past_generations=1),
        )
        for idx in range(2)
    ]
    for pop in pops:
        pop.simulate_generations(generations=1, related_offspring=True, trait_updates=False)
    spop = psg.SuperPopulation(pops)

    joined = spop.join_populations(return_obj=True)

    assert joined.keep_past_generations == 1
    assert joined.past is not None
    assert len(joined.past) >= 2
    assert joined.past[1] is not None


def test_join_populations_rejects_odd_Ns_total():
    spop = psg.SuperPopulation([
        psg.Population(N=5, M=4, p_init=0.3, seed=1),
        psg.Population(N=5, M=4, p_init=0.3, seed=2),
    ])

    with pytest.raises(ValueError, match="even"):
        spop.join_populations(Ns=[2, 3])


def test_join_populations_rejects_odd_N_new():
    spop = psg.SuperPopulation([
        psg.Population(N=5, M=4, p_init=0.3, seed=1),
        psg.Population(N=5, M=4, p_init=0.3, seed=2),
    ])

    with pytest.raises(ValueError, match="even"):
        spop.join_populations(N_new=5, admix_fractions=[0.4, 0.6])


def test_join_populations_can_sample_by_admix_fractions():
    pops = [
        psg.Population(N=8, M=5, p_init=0.3, seed=idx)
        for idx in range(3)
    ]
    spop = psg.SuperPopulation(pops)
    spop.add_subpop_trait()

    np.random.seed(13)
    spop.join_populations(
        N_new=10,
        admix_fractions=[0.2, 0.5, 0.3],
    )

    joined = spop.pops[-1]
    assert joined.N == 10
    subpop = joined.traits["subpop"].y.astype(int)
    np.testing.assert_array_equal(
        np.bincount(subpop, minlength=3),
        np.array([2, 5, 3]),
    )


def test_join_populations_requires_admix_arguments_together():
    spop = psg.SuperPopulation([
        psg.Population(N=5, M=4, p_init=0.3, seed=1),
        psg.Population(N=5, M=4, p_init=0.3, seed=2),
    ])

    with pytest.raises(ValueError, match="provided together"):
        spop.join_populations(N_new=4)

    with pytest.raises(ValueError, match="provided together"):
        spop.join_populations(admix_fractions=[0.5, 0.5])


def test_join_populations_rejects_Ns_with_admix_arguments():
    spop = psg.SuperPopulation([
        psg.Population(N=5, M=4, p_init=0.3, seed=1),
        psg.Population(N=5, M=4, p_init=0.3, seed=2),
    ])

    with pytest.raises(ValueError, match="cannot be used"):
        spop.join_populations(Ns=[2, 2], N_new=4, admix_fractions=[0.5, 0.5])


def test_join_populations_rejects_admix_request_larger_than_source_population():
    spop = psg.SuperPopulation([
        psg.Population(N=2, M=4, p_init=0.3, seed=1),
        psg.Population(N=8, M=4, p_init=0.3, seed=2),
    ])

    with pytest.raises(ValueError, match="only contains"):
        spop.join_populations(N_new=8, admix_fractions=[0.5, 0.5])


def test_add_popstrat_requires_more_than_one_populated_subpopulation():
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=21)
    spop = psg.SuperPopulation([pop])
    spop.add_subpop_trait()
    pop.add_trait(
        name="y",
        effects={"base": FixedEffect(name="base", beta=1.0, input_name="x")},
        inputs={"x": np.arange(pop.N, dtype=float)},
        var_Eps=0.5,
    )

    with pytest.raises(ValueError, match="only one populated cluster"):
        pop.traits["y"].add_popstrat(variance=0.4)


def test_superpopulation_from_FST_builds_subpops_and_adds_subpop_trait():
    p0 = np.linspace(0.1, 0.9, 6)
    spop = psg.SuperPopulation.from_FST(n_pops=3, N=4, FST=0.2, p0=p0, seed=0)

    assert len(spop.pops) == 3
    assert spop.active == [True, True, True]
    for idx, pop in enumerate(spop.pops):
        assert pop.N == 4
        assert pop.M == len(p0)
        assert "subpop" in pop.traits
        np.testing.assert_array_equal(pop.traits["subpop"].y, np.full(pop.N, idx))


def test_add_subpop_trait_can_override_existing_subpop_trait():
    pop = psg.Population(N=4, M=5, p_init=0.3, seed=41)
    pop.add_trait_from_fixed_values(
        name="subpop",
        y=np.full(pop.N, 99, dtype=int),
        trait_type="permanent",
    )
    spop = psg.SuperPopulation([pop])

    spop.add_subpop_trait()
    np.testing.assert_array_equal(pop.traits["subpop"].y, np.full(pop.N, 99))

    spop.add_subpop_trait(override=True)
    np.testing.assert_array_equal(pop.traits["subpop"].y, np.zeros(pop.N, dtype=int))


def test_superpopulation_set_params_updates_selected_or_active_populations():
    pops = [
        psg.Population(N=4, M=5, p_init=0.3, seed=idx)
        for idx in range(3)
    ]
    spop = psg.SuperPopulation(pops)
    spop.inactivate_population(1)

    spop.set_params(mu=0.02)
    assert spop.pops[0].params.mu == 0.02
    assert spop.pops[2].params.mu == 0.02
    assert spop.pops[1].params.mu == 0.0

    spop.set_params(pop_i=[1], R_type="uniform", AM_r=0.3)
    assert spop.pops[1].params.R_type == "uniform"
    assert spop.pops[1].params.AM_r == 0.3
    np.testing.assert_allclose(
        spop.pops[1].params.R,
        psg.genome.generate_recombination_rates(spop.pops[1].M, R_type="uniform"),
    )


def test_superpopulation_from_FST_accepts_per_population_FST_with_warning():
    p0 = np.linspace(0.2, 0.8, 5)
    fst = np.array([0.01, 0.05])

    with pytest.warns(UserWarning, match="passed directly to draw_p_FST"):
        spop = psg.SuperPopulation.from_FST(n_pops=2, N=[3, 4], FST=fst, p0=p0, seed=1)

    assert [pop.N for pop in spop.pops] == [3, 4]
    for pop in spop.pops:
        assert pop.M == len(p0)


def test_superpopulation_from_FST_checks_per_population_FST_length():
    p0 = np.linspace(0.2, 0.8, 5)

    with pytest.raises(ValueError, match="length `n_pops`"):
        psg.SuperPopulation.from_FST(n_pops=3, N=4, FST=np.array([0.01, 0.02]), p0=p0)


def test_superpopulation_from_FST_requires_M_when_p0_missing():
    with pytest.raises(ValueError, match="Must provide `M`"):
        psg.SuperPopulation.from_FST(n_pops=2, N=3)


def test_validate_moves_eps_component_to_the_end():
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=11)
    pop.add_trait(
        name="y",
        effects={"base": FixedEffect(name="base", beta=1.0, input_name="x")},
        inputs={"x": np.arange(pop.N, dtype=float)},
        var_Eps=0.5,
    )

    trait = pop.traits["y"]
    trait.y_ = {"Eps": trait.y_["Eps"], "base": trait.y_["base"]}
    trait.validate()

    assert list(trait.y_.keys()) == ["base", "Eps"]


def test_random_effect_seed_is_deterministic_and_static_kernel_is_dropped_on_subset():
    pop = psg.Population(N=8, M=5, p_init=0.3, seed=2)
    pop.set_params(mu=0.02, AM_r=0.1, R_type="uniform")
    x = np.arange(pop.N, dtype=float)
    pop.add_trait(
        name="y",
        effects={
            "base": FixedEffect(name="base", beta=1.0, input_name="x"),
            "dyn": RandomEffect(name="dyn", var=1.0, seed=123),
            "static": RandomEffect.from_matrix(
                name="static",
                var=1.0,
                K=np.eye(pop.N),
                seed=456,
            ),
        },
        inputs={"x": x},
    )

    trait = pop.traits["y"]
    dyn_initial = trait.y_["dyn"].copy()
    trait.generate_trait()
    np.testing.assert_allclose(trait.y_["dyn"], dyn_initial)

    with pytest.warns(UserWarning, match="removed random effect"):
        subset = pop.subset_individuals(np.arange(pop.N - 2))

    subset_trait = subset.traits["y"]
    assert "dyn" in subset_trait.effects
    assert "static" not in subset_trait.effects
    assert subset_trait.y_["dyn"].shape == (pop.N - 2,)
    assert not np.allclose(subset_trait.y_["dyn"], dyn_initial[: subset.N])
    assert subset.params is not pop.params
    assert subset.params.mu == pop.params.mu
    assert subset.params.AM_r == pop.params.AM_r
    assert subset.params.R_type == pop.params.R_type
    np.testing.assert_allclose(subset.params.R, pop.params.R)


def test_subset_individuals_defaults_keep_past_generations_from_params():
    pop = psg.Population(
        N=8,
        M=5,
        p_init=0.3,
        seed=12,
        params=psg.PopulationParams(keep_past_generations=1),
    )
    pop.simulate_generations(generations=1, related_offspring=True, trait_updates=False)

    subset = pop.subset_individuals(np.arange(pop.N - 2))

    assert subset.keep_past_generations == 1
    assert subset.past is not None
    assert len(subset.past) >= 2
    assert subset.past[1] is not None


def test_simulating_new_generation_drops_static_random_effects():
    pop = psg.Population(N=6, M=4, p_init=0.25, seed=4)
    x = np.linspace(0.0, 1.0, pop.N)
    pop.add_trait(
        name="y",
        effects={
            "base": FixedEffect(name="base", beta=1.0, input_name="x"),
            "dyn": RandomEffect(name="dyn", var=1.0, seed=5),
            "static": RandomEffect.from_matrix(
                name="static",
                var=1.0,
                K=np.eye(pop.N),
                seed=6,
            ),
        },
        inputs={"x": x},
    )

    with pytest.warns(UserWarning, match="removed random effect"):
        pop.simulate_generations(generations=1, related_offspring=False, trait_updates=True)

    trait = pop.traits["y"]
    assert "dyn" in trait.effects
    assert "static" not in trait.effects


def test_rescale_empirical_variance_matches_requested_variance_exactly():
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=40)
    values = np.array([1.0, -2.0, 3.0, 0.5, -1.5, 2.5], dtype=float)

    standardized = pop._rescale_empirical_variance(values, target_var=1.0, name="x")
    noise = pop._rescale_empirical_variance(values, target_var=0.36, name="x")

    assert np.isclose(standardized.mean(), 0.0)
    assert np.isclose(np.var(standardized), 1.0)
    assert np.isclose(noise.mean(), 0.0)
    assert np.isclose(np.var(noise), 0.36)


def test_pair_mates_uses_exact_empirical_noise_variance_under_am(monkeypatch):
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=41)
    pop.traits["sex"].y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    pop.add_trait_from_fixed_values(name="am", y=np.array([-2.0, -0.5, 1.0, -1.5, 0.5, 2.0]))

    calls = []
    original_rescale = pop._rescale_empirical_variance

    def wrapped_rescale(values, target_var, name):
        out = original_rescale(values, target_var=target_var, name=name)
        calls.append((name, float(target_var), float(np.var(out))))
        return out

    monkeypatch.setattr(pop, "_rescale_empirical_variance", wrapped_rescale)
    monkeypatch.setattr(
        np.random,
        "choice",
        lambda a, size, replace=False: np.asarray(a)[:size],
    )

    pop._pair_mates(AM_r=0.8, AM_trait="am")

    assert calls[0][0] == "AM_values"
    assert np.isclose(calls[0][1], 1.0)
    assert np.isclose(calls[0][2], 1.0)
    assert calls[1][0] == "mate noise"
    assert np.isclose(calls[1][1], 1.0 - 0.8 ** 2)
    assert np.isclose(calls[1][2], 1.0 - 0.8 ** 2)


def test_random_effect_can_reference_existing_component():
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=3)
    x = np.arange(pop.N, dtype=float)
    pop.add_trait(
        name="y",
        effects={
            "base": FixedEffect(name="base", beta=1.0, input_name="x"),
            "corr": RandomEffect(
                name="corr",
                var=1.0,
                r=0.4,
                reference_component="base",
                seed=7,
            ),
        },
        inputs={"x": x},
    )

    assert pop.traits["y"].y_["corr"].shape == (pop.N,)


def test_remove_effect_validates_trait_state():
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=12)
    pop.add_trait(
        name="y",
        effects={"base": FixedEffect(name="base", beta=1.0, input_name="x")},
        inputs={"x": np.arange(pop.N, dtype=float)},
        var_Eps=0.5,
    )

    trait = pop.traits["y"]
    trait.remove_effect("base")

    assert list(trait.y_.keys()) == ["Eps"]
    assert np.allclose(trait.y, trait.y_["Eps"])


def test_simulate_pgs_standardized_weights_uses_r2_formula():
    beta = np.array([0.2, -0.1, 0.0, 0.3], dtype=float)
    h2 = 0.35
    r2 = 0.2
    seed = 123

    observed = simulate_pgs_standardized_weights(beta=beta, h2=h2, R2=r2, seed=seed)

    v_eps = (h2 / beta.size) * ((h2 / r2) - 1.0)
    expected = beta + np.random.default_rng(seed).normal(0.0, np.sqrt(v_eps), size=beta.size)
    np.testing.assert_allclose(observed, expected)


def test_simulate_pgs_standardized_weights_uses_n_formula():
    beta = np.array([0.2, -0.1, 0.0, 0.3], dtype=float)
    h2 = 0.35
    n = 5000
    seed = 456

    observed = simulate_pgs_standardized_weights(beta=beta, h2=h2, N=n, seed=seed)

    expected = beta + np.random.default_rng(seed).normal(0.0, np.sqrt(1.0 / n), size=beta.size)
    np.testing.assert_allclose(observed, expected)


def test_simulate_pgs_standardized_weights_requires_exactly_one_of_r2_or_n():
    beta = np.array([0.1, 0.2], dtype=float)

    with pytest.raises(ValueError, match="exactly one of R2 or N"):
        simulate_pgs_standardized_weights(beta=beta, h2=0.2)

    with pytest.raises(ValueError, match="exactly one of R2 or N"):
        simulate_pgs_standardized_weights(beta=beta, h2=0.2, R2=0.1, N=1000)


def test_simulate_pgs_standardized_weights_rejects_r2_above_h2():
    beta = np.array([0.1, 0.2], dtype=float)

    with pytest.raises(ValueError, match="less than or equal to h2"):
        simulate_pgs_standardized_weights(beta=beta, h2=0.2, R2=0.3, seed=0)


def test_trait_simulate_pgs_weights_wraps_additive_effects():
    pop = psg.Population(N=10, M=5, p_init=0.35, seed=8)
    effect = GeneticEffect(var_indep=0.4, M=pop.M, M_causal=5, name="A")
    pop.add_trait(name="y", effects={"A": effect}, var_Eps=0.6)

    trait = pop.traits["y"]
    beta = trait.effects["A"].effects_standardized.copy()
    h2 = trait.get_h2(method="additive_effects")
    seed = 789

    observed = trait.simulate_PGS_weights(R2=0.1, seed=seed)
    expected = simulate_pgs_standardized_weights(beta=beta, h2=h2, R2=0.1, seed=seed)

    np.testing.assert_allclose(observed, expected)


def test_trait_run_gwas_matches_manual_ols_with_covariates():
    pop = psg.Population(N=8, M=3, p_init=0.3, seed=41)
    x = np.linspace(-1.5, 1.5, pop.N)
    beta = np.array([0.4, -0.2, 0.1], dtype=float)
    y = 1.2 + 0.7 * x + pop.G @ beta
    pop.add_trait_from_fixed_values(name="y", y=y)

    out = pop.traits["y"].run_GWAS(
        covariates=x[:, None],
        standardize_y=False,
        standardize_geno=False,
        detailed_output=True,
    )

    assert out.trait_name == "y"
    assert out.N == pop.N
    assert out.M == pop.M
    assert out.n_covariates == 1
    assert out.detailed_output is True

    expected_intercepts = np.empty(pop.M, dtype=float)
    expected_intercept_se = np.empty(pop.M, dtype=float)
    expected_covar = np.empty((pop.M, 1), dtype=float)
    expected_covar_se = np.empty((pop.M, 1), dtype=float)
    expected_beta = np.empty(pop.M, dtype=float)
    expected_beta_se = np.empty(pop.M, dtype=float)
    expected_vcov = np.empty((pop.M, 3, 3), dtype=float)
    expected_residual_var = np.empty(pop.M, dtype=float)
    expected_rss = np.empty(pop.M, dtype=float)

    for j in range(pop.M):
        X = np.column_stack((np.ones(pop.N), x, pop.G[:, j]))
        xtx_inv = np.linalg.pinv(X.T @ X)
        coef = xtx_inv @ (X.T @ y)
        resid = y - X @ coef
        sigma2 = float(resid @ resid) / (pop.N - X.shape[1])
        vcov = sigma2 * xtx_inv
        se = np.sqrt(np.diag(vcov))
        expected_intercepts[j] = coef[0]
        expected_intercept_se[j] = se[0]
        expected_covar[j, 0] = coef[1]
        expected_covar_se[j, 0] = se[1]
        expected_beta[j] = coef[2]
        expected_beta_se[j] = se[2]
        expected_vcov[j, :, :] = vcov
        expected_residual_var[j] = sigma2
        expected_rss[j] = float(resid @ resid)

    np.testing.assert_allclose(out.intercept_est, expected_intercepts)
    np.testing.assert_allclose(out.intercept_se, expected_intercept_se)
    np.testing.assert_allclose(out.covar_est, expected_covar)
    np.testing.assert_allclose(out.covar_se, expected_covar_se)
    np.testing.assert_allclose(out.beta_est, expected_beta)
    np.testing.assert_allclose(out.beta_se, expected_beta_se)
    np.testing.assert_allclose(out.coef_vcov, expected_vcov)
    np.testing.assert_allclose(out.residual_var, expected_residual_var)
    np.testing.assert_allclose(out.rss, expected_rss)
    assert out.dof_resid == pop.N - 3
    assert np.isclose(out.y_mean, y.mean())
    assert np.isclose(out.y_var, y.var())


def test_trait_run_gwas_uses_standardized_genotypes_and_standardized_trait():
    pop = psg.Population(N=7, M=2, p_init=0.3, seed=42)
    y = np.array([0.5, 1.1, -0.2, 2.4, -1.3, 0.7, 1.8], dtype=float)
    pop.add_trait_from_fixed_values(name="y", y=y)

    out = pop.traits["y"].run_GWAS()

    y_std = (y - y.mean()) / y.std()
    expected_beta = np.empty(pop.M, dtype=float)
    expected_beta_se = np.empty(pop.M, dtype=float)
    for j in range(pop.M):
        X = np.column_stack((np.ones(pop.N), np.asarray(pop.X[:, j], dtype=float)))
        xtx_inv = np.linalg.pinv(X.T @ X)
        coef = xtx_inv @ (X.T @ y_std)
        resid = y_std - X @ coef
        sigma2 = float(resid @ resid) / (pop.N - X.shape[1])
        vcov = sigma2 * xtx_inv
        se = np.sqrt(np.diag(vcov))
        expected_beta[j] = coef[1]
        expected_beta_se[j] = se[1]

    np.testing.assert_allclose(out.beta_est, expected_beta)
    np.testing.assert_allclose(out.beta_se, expected_beta_se)
    assert out.detailed_output is False
    assert out.intercept_est is None
    assert out.intercept_se is None
    assert out.covar_est is None
    assert out.covar_se is None
    assert out.coef_vcov is None
    assert out.residual_var is None
    assert out.rss is None
    assert out.gamma_est is None
    assert out.gamma_se is None
    assert out.within_family is None
    assert out.dof_resid == pop.N - 2
    assert np.isclose(out.y_mean, y_std.mean())
    assert np.isclose(out.y_var, y_std.var())


def test_trait_run_gwas_adjust_pcs_adds_sample_pcs_as_covariates():
    pop = psg.Population(N=8, M=4, p_init=0.3, seed=52)
    y = np.array([0.2, 1.1, -0.4, 0.9, 1.7, -1.2, 0.5, 1.3], dtype=float)
    pop.add_trait_from_fixed_values(name="y", y=y)

    out = pop.traits["y"].run_GWAS(
        adjust_PCs=2,
        standardize_y=False,
        standardize_geno=True,
        detailed_output=True,
    )

    pca = compute_PCA(X=np.asarray(pop.X, dtype=float), n_components=2)
    pcs = np.asarray(pca.scores[:, :2], dtype=float)
    expected_covar = np.empty((pop.M, 2), dtype=float)
    expected_covar_se = np.empty((pop.M, 2), dtype=float)
    for j in range(pop.M):
        X = np.column_stack((np.ones(pop.N), pcs, np.asarray(pop.X[:, j], dtype=float)))
        xtx_inv = np.linalg.pinv(X.T @ X)
        coef = xtx_inv @ (X.T @ y)
        resid = y - X @ coef
        sigma2 = float(resid @ resid) / (pop.N - X.shape[1])
        vcov = sigma2 * xtx_inv
        se = np.sqrt(np.diag(vcov))
        expected_covar[j, :] = coef[1:3]
        expected_covar_se[j, :] = se[1:3]

    assert out.n_covariates == 2
    np.testing.assert_allclose(out.covar_est, expected_covar)
    np.testing.assert_allclose(out.covar_se, expected_covar_se)


def test_trait_run_gwas_adjust_pcs_appends_to_user_covariates():
    pop = psg.Population(N=8, M=3, p_init=0.3, seed=53)
    y = np.array([1.2, -0.3, 0.4, 1.0, -1.1, 0.2, 0.8, -0.6], dtype=float)
    x = np.linspace(-1.0, 1.0, pop.N)
    pop.add_trait_from_fixed_values(name="y", y=y)

    out = pop.traits["y"].run_GWAS(
        covariates=x[:, None],
        adjust_PCs=1,
        standardize_y=False,
        standardize_geno=False,
        detailed_output=True,
    )

    pca = compute_PCA(X=np.asarray(pop.X, dtype=float), n_components=1)
    pcs = np.asarray(pca.scores[:, :1], dtype=float)
    expected_covar = np.empty((pop.M, 2), dtype=float)
    for j in range(pop.M):
        X = np.column_stack((np.ones(pop.N), x[:, None], pcs, np.asarray(pop.G[:, j], dtype=float)))
        coef = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        expected_covar[j, :] = coef[1:3]

    assert out.n_covariates == 2
    np.testing.assert_allclose(out.covar_est, expected_covar)


def test_trait_run_gwas_within_family_gpar_reports_gamma_separately():
    pop = psg.Population(N=8, M=3, p_init=0.3, seed=43)
    x = np.linspace(-1.0, 1.0, pop.N)
    gpar = pop.G + np.array([0.5, 1.0, -0.5], dtype=float)[None, :]
    pop.G_par = gpar
    beta = np.array([0.3, -0.1, 0.2], dtype=float)
    gamma = np.array([0.4, 0.2, -0.3], dtype=float)
    y = 0.8 + 0.5 * x + pop.G @ beta + gpar @ gamma
    pop.add_trait_from_fixed_values(name="y", y=y)

    out = pop.traits["y"].run_GWAS(
        covariates=x[:, None],
        within_family='Gpar',
        standardize_y=False,
        standardize_geno=False,
        detailed_output=True,
    )

    assert out.within_family == 'Gpar'
    assert out.gamma_est is not None
    assert out.gamma_se is not None
    assert out.n_covariates == 1

    expected_intercepts = np.empty(pop.M, dtype=float)
    expected_intercept_se = np.empty(pop.M, dtype=float)
    expected_covar = np.empty((pop.M, 1), dtype=float)
    expected_covar_se = np.empty((pop.M, 1), dtype=float)
    expected_gamma = np.empty(pop.M, dtype=float)
    expected_gamma_se = np.empty(pop.M, dtype=float)
    expected_beta = np.empty(pop.M, dtype=float)
    expected_beta_se = np.empty(pop.M, dtype=float)

    for j in range(pop.M):
        X = np.column_stack((np.ones(pop.N), x, gpar[:, j], pop.G[:, j]))
        xtx_inv = np.linalg.pinv(X.T @ X)
        coef = xtx_inv @ (X.T @ y)
        resid = y - X @ coef
        sigma2 = float(resid @ resid) / (pop.N - X.shape[1])
        vcov = sigma2 * xtx_inv
        se = np.sqrt(np.diag(vcov))
        expected_intercepts[j] = coef[0]
        expected_intercept_se[j] = se[0]
        expected_covar[j, 0] = coef[1]
        expected_covar_se[j, 0] = se[1]
        expected_gamma[j] = coef[2]
        expected_gamma_se[j] = se[2]
        expected_beta[j] = coef[3]
        expected_beta_se[j] = se[3]

    np.testing.assert_allclose(out.intercept_est, expected_intercepts)
    np.testing.assert_allclose(out.intercept_se, expected_intercept_se)
    np.testing.assert_allclose(out.covar_est, expected_covar)
    np.testing.assert_allclose(out.covar_se, expected_covar_se)
    np.testing.assert_allclose(out.gamma_est, expected_gamma)
    np.testing.assert_allclose(out.gamma_se, expected_gamma_se)
    np.testing.assert_allclose(out.beta_est, expected_beta)
    np.testing.assert_allclose(out.beta_se, expected_beta_se)


def test_prune_sibs_applies_min_and_max_filters_deterministically():
    pop = psg.Population(N=6, M=5, p_init=0.3, seed=17)
    pop.relations["full_sibs"] = np.array([0, 0, 0, 1, 1, -1], dtype=np.int32)

    pruned = pop.prune_sibs(min_n_sibs=2, max_n_sibs=2, seed=7, keep_past_generations=0)

    expected_first_family = np.sort(
        np.random.default_rng(7).choice(np.array([0, 1, 2]), size=2, replace=False)
    )
    expected_keep = np.concatenate([expected_first_family, np.array([3, 4])])

    assert pruned.N == 4
    np.testing.assert_array_equal(pruned.G, pop.G[expected_keep])
    np.testing.assert_array_equal(
        np.sort(np.bincount(pruned.relations["full_sibs"])),
        np.array([2, 2]),
    )


def test_impute_gpar_sibs_linear_returns_family_mean_and_warns_on_singletons():
    pop = psg.Population(N=4, M=4, p_init=0.35, seed=18)
    pop.relations["full_sibs"] = np.array([0, 0, 1, -1], dtype=np.int32)

    with pytest.warns(UserWarning, match="n_sibs=1"):
        result = pop.impute_Gpar(method="sibs_linear")

    expected = np.empty((pop.N, pop.M), dtype=float)
    expected[[0, 1]] = 2.0 * pop.G[[0, 1]].mean(axis=0, dtype=float)
    expected[2] = 2.0 * pop.G[2]
    expected[3] = 2.0 * pop.G[3]

    np.testing.assert_allclose(result["Gpar"], expected)
    np.testing.assert_array_equal(result["n_sibs"], np.array([2, 2, 1, 1], dtype=np.int32))


def test_impute_gpar_af_pop_matches_population_frequency_formula():
    pop = psg.Population(N=5, M=6, p_init=np.linspace(0.2, 0.7, 6), seed=19)

    result = pop.impute_Gpar(method="AF_pop")

    expected = pop.G + 2.0 * pop.p[None, :]
    np.testing.assert_allclose(result["Gpar"], expected)


def test_impute_gpar_af_pcs_matches_pca_reconstruction_formula():
    pop = psg.Population(N=8, M=6, p_init=np.linspace(0.15, 0.75, 6), seed=20)
    pca = pop.compute_PCA(n_components=2)

    result = pop.impute_Gpar(method="AF_PCs", n_components=2, pca=pca)

    scores = np.asarray(pca.scores[:, :2], dtype=float)
    eigenvalues = np.asarray(pca.eigenvalues[:2], dtype=float)
    loadings = (np.asarray(pop.X, dtype=float).T @ scores) / eigenvalues[None, :]
    X_hat = scores @ loadings.T
    scale = np.sqrt(2.0 * pop.p * (1.0 - pop.p))
    p_adjusted = np.clip(pop.p[None, :] + 0.5 * scale[None, :] * X_hat, 0.0, 1.0)
    expected = pop.G + 2.0 * p_adjusted

    np.testing.assert_allclose(result["Gpar"], expected)
    assert result["n_components_used"] == 2
    assert 0.0 <= result["fraction_af_clipped"] <= 1.0


def test_impute_gpar_compute_error_uses_shared_matrix_metrics():
    pop = psg.Population(
        N=6,
        M=5,
        p_init=0.3,
        seed=21,
        params=psg.PopulationParams(keep_past_generations=1),
    )
    pop.simulate_generations(generations=1, related_offspring=True, trait_updates=False)

    result = pop.impute_Gpar(method="AF_pop", compute_error=True)
    expected = summarize_matrix_error(pop.get_Gpar(), result["Gpar"])

    for key, value in expected.items():
        if np.isnan(value):
            assert np.isnan(result["error_metrics"][key])
        else:
            np.testing.assert_allclose(result["error_metrics"][key], value)


def test_generate_offspring_rejects_unknown_offspring_distribution():
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=22)

    with pytest.raises(ValueError, match="n_offspring_dist"):
        pop.generate_offspring(n_offspring_dist="unsupported")


def test_population_params_store_R_without_pop_R_attribute():
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=220)

    assert isinstance(pop.params, psg.PopulationParams)
    assert not hasattr(pop, "R")
    assert pop.params.R_type == "indep"
    np.testing.assert_allclose(pop.params.R, np.full(pop.M, 0.5))
    assert pop.params.keep_past_generations == 1
    assert pop.params.track_pedigree is False
    assert pop.params.track_haplotypes is False
    assert pop.params.metric_retention == "store_every"
    assert pop.params.metric_last_k is None
    assert pop.params.trait_updates is True

    pop.set_params(R=0.25)
    assert pop.params.R_type == "custom"
    np.testing.assert_allclose(pop.params.R, np.full(pop.M, 0.25))

    new_R = np.linspace(0.1, 0.4, pop.M)
    pop.set_params(R=new_R)
    np.testing.assert_allclose(pop.params.R, new_R)


def test_population_set_params_R_type_regenerates_R_from_genome_structure():
    pop = psg.Population(N=6, M=5, p_init=0.3, seed=222)

    pop.set_params(R_type="uniform")
    assert pop.params.R_type == "uniform"
    np.testing.assert_allclose(
        pop.params.R,
        psg.genome.generate_recombination_rates(pop.M, R_type="uniform"),
    )

    pop.set_params(R_type="indep")
    np.testing.assert_allclose(pop.params.R, np.full(pop.M, 0.5))


def test_population_init_rejects_legacy_storage_arguments():
    with pytest.raises(TypeError):
        psg.Population(N=6, M=4, p_init=0.3, keep_past_generations=1)

def test_simulate_generations_uses_params_with_temporary_overrides(monkeypatch):
    pop = psg.Population(
        N=6,
        M=4,
        p_init=0.3,
        seed=221,
        params=psg.PopulationParams(keep_past_generations=1, trait_updates=True),
    )
    pop.set_params(
        related_offspring=False,
        s=0.2,
        mu=0.3,
        n_offspring_dist="constant",
    )

    calls = []

    def fake_next_generation(s, mu):
        calls.append(("next", s, mu))
        return pop.H.copy()

    monkeypatch.setattr(pop, "next_generation", fake_next_generation)
    pop.simulate_generations(generations=1, trait_updates=False)

    assert calls == [("next", 0.2, 0.3)]

    def fake_generate_offspring(s, mu, R, n_offspring_dist, AM_r, AM_trait, AM_type):
        calls.append(("offspring", s, mu, R.copy(), n_offspring_dist, AM_r, AM_trait, AM_type))
        relations = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in pop.relations.items()
        }
        return (pop.H.copy(), relations, None)

    monkeypatch.setattr(pop, "generate_offspring", fake_generate_offspring)
    pop.simulate_generations(
        generations=1,
        related_offspring=True,
        mu=0.0,
        n_offspring_dist="poisson",
        trait_updates=False,
    )

    assert calls[1][0] == "offspring"
    assert calls[1][2] == 0.0
    assert calls[1][4] == "poisson"
    assert pop.params.related_offspring is False
    assert pop.params.mu == 0.3
    assert pop.params.n_offspring_dist == "constant"


def test_simulate_generations_defaults_to_trait_updates_from_params(monkeypatch):
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=223)
    calls = []

    def fake_next_generation(s, mu):
        return pop.H.copy()

    def fake_update_traits(traits=None):
        calls.append(traits)

    monkeypatch.setattr(pop, "next_generation", fake_next_generation)
    monkeypatch.setattr(pop, "update_traits", fake_update_traits)
    pop.set_params(related_offspring=False, trait_updates=True)

    pop.simulate_generations(generations=1)

    assert calls == [None]


def test_constant_offspring_distribution_balances_noninteger_expected_counts():
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=23)

    counts = pop._sample_offspring_pair_counts(
        P_mate=np.full(pop.N // 2, 1.0 / (pop.N // 2)),
        N_offspring=5,
        n_offspring_dist="constant",
    )

    np.testing.assert_array_equal(np.sort(counts), np.array([1, 2, 2], dtype=np.int32))
    assert counts.sum() == 5


def test_generate_offspring_constant_distribution_gives_two_children_per_pair_when_balanced():
    pop = psg.Population(N=8, M=4, p_init=0.3, seed=24)

    _, relations, _ = pop.generate_offspring(n_offspring_dist="constant")

    family_sizes = np.bincount(relations["full_sibs"], minlength=pop.N // 2)
    np.testing.assert_array_equal(family_sizes, np.full(pop.N // 2, 2, dtype=int))


def test_simulate_generations_passes_constant_offspring_distribution():
    pop = psg.Population(
        N=6,
        M=4,
        p_init=0.3,
        seed=25,
        params=psg.PopulationParams(keep_past_generations=1),
    )

    pop.simulate_generations(
        generations=1,
        related_offspring=True,
        trait_updates=False,
        n_offspring_dist="constant",
    )

    family_sizes = np.bincount(pop.relations["full_sibs"], minlength=pop.N // 2)
    np.testing.assert_array_equal(family_sizes, np.full(pop.N // 2, 2, dtype=int))

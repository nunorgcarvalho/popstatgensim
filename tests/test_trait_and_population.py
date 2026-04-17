import numpy as np
import pytest

import popstatgensim as psg
from popstatgensim.traits import (
    FixedEffect,
    GeneticEffect,
    RandomEffect,
    simulate_pgs_standardized_weights,
)


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
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=32, keep_past_generations=1)
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
    pop = psg.Population(N=6, M=5, p_init=0.3, seed=33, keep_past_generations=1)
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
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=334, keep_past_generations=1)
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
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=335, keep_past_generations=1)
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

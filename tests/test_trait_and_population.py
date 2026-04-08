import numpy as np
import pytest

import popstatgensim as psg
from popstatgensim.traits import FixedEffect, GeneticEffect, RandomEffect


def test_population_trait_can_reference_existing_population_trait():
    pop = psg.Population(N=6, M=4, p_init=0.3, seed=0)
    base_values = np.arange(pop.N, dtype=float)
    pop.add_trait_from_fixed_values(name="base", y=base_values)

    effects = {
        "base_copy": FixedEffect(
            name="base_copy",
            beta=2.0,
            input_name="base",
            is_trait=True,
        )
    }
    pop.add_trait(name="y", effects=effects)

    np.testing.assert_allclose(pop.traits["y"].y, 2.0 * base_values)


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

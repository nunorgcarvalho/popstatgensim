import numpy as np

import popstatgensim as psg
from popstatgensim.traits import FixedEffect, GeneticEffect


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

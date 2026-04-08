# Refactor Proposal for `popstatgensim`

## Summary

This document proposes a structural refactor of the `popstatgensim` repository so the package is easier to maintain, easier to document, and better aligned with an eventual public release.

The main goals are:

1. Split large multi-purpose source files into smaller modules with clearer responsibilities.
2. Make core mathematical and simulation logic easier to read and cite in future documentation.
3. Separate pure computation from plotting, exporting, and high-level orchestration.
4. Reduce unnecessary coupling between major parts of the package.
5. Preserve current functionality as much as possible, with most changes focused on code location and module boundaries rather than algorithmic behavior.

This proposal is intentionally optimized for the cleanest long-term package structure rather than backwards compatibility with the current module layout.

## Guiding Principles

### 1. Organize by domain, not by generic "functions" files

The current package has several files such as `popgen_functions.py`, `statgen_functions.py`, and `core_functions.py` that each contain multiple different conceptual areas. That structure made sense while the package was small, but it now makes it harder to:

1. find the authoritative implementation of a given idea,
2. document parts of the package cleanly,
3. understand dependency direction,
4. avoid circular coupling between classes and helper functions.

The refactor should move toward domain-based modules and subpackages, where each file has a clear conceptual purpose.

### 2. Keep pure logic separate from wrappers and orchestration

A recurring pattern in the current codebase is that a high-level class method wraps a lower-level function. That is a good pattern and should be kept. However, the pure functions should live in modules devoted to the relevant math or simulation concept, while high-level classes like `Population`, `Trait`, and `SuperPopulation` should mainly orchestrate those functions and manage state.

### 3. Separate computation from presentation and I/O

Plotting functions and export functions should not live in the same modules as the core computations they visualize or serialize. That separation will make it easier to:

1. document the scientific logic independently from plotting behavior,
2. avoid large modules mixing unrelated concerns,
3. test numerical logic without involving matplotlib or file-writing code.

### 4. Favor explicit, one-way dependency flow

Lower-level modules should not depend on higher-level orchestrator objects. In particular:

1. utility and math modules should not depend on `Population`,
2. trait-generation helpers should not depend on the whole simulation layer,
3. plotting and export layers should sit near the edge of the package and depend on the computational core, not the other way around.

This does not mean `Trait` can never reference its owning `Population`; it means that the deepest computational pieces should work primarily on arrays and explicit inputs rather than requiring broad access to simulation objects.

### 5. Make the code legible enough to reference in docs

You mentioned wanting future documentation pages to reference code directly or through mathematical representations. To support that, the refactor should prioritize:

1. smaller files,
2. narrow file responsibilities,
3. mathematically coherent modules,
4. fewer "miscellaneous" helpers mixed into unrelated logic.

## Proposed Top-Level Package Structure

The long-term target is to replace the current flat module layout inside `src/popstatgensim/` with domain-oriented subpackages such as:

```text
src/popstatgensim/
  __init__.py
  genetics/
    __init__.py
    genotypes.py
    pca.py
    ld.py
    frequencies.py
  genome/
    __init__.py
    structure.py
  pedigree/
    __init__.py
    relations.py
    ibd.py
    pedigree.py
    relative_types.py
  traits/
    __init__.py
    effects.py
    trait.py
    effect_sampling.py
    random_effects.py
    fixed_effects.py
  simulation/
    __init__.py
    population.py
    superpopulation.py
  estimation/
    __init__.py
    reml.py
    he.py
    _common.py
    _reml_accel.c
  plotting/
    __init__.py
    genetics.py
    estimation.py
    common.py
  io/
    __init__.py
    gcta.py
  utils/
    __init__.py
    stats.py
    misc.py
```

This exact naming can still be adjusted slightly during implementation, but the broader structure should remain domain-based rather than function-bucket-based.

## Dependency Philosophy

The codebase should follow a mostly one-way dependency flow:

1. `utils` is the bottom layer.
   It contains generic helpers that do not depend on package-specific simulation classes.

2. `genetics`, `genome`, `pedigree`, `traits`, and `estimation` are domain layers.
   These may depend on `utils`, but should not depend on `simulation` unless there is a very strong reason.

3. `simulation` is the orchestration layer.
   It may depend on `genetics`, `genome`, `pedigree`, and `traits`, because `Population` and `SuperPopulation` combine those pieces into user-facing workflows.

4. `plotting` and `io` are edge layers.
   They depend on the computational layers, but the computational layers should not depend on them.

5. `__init__.py` should be a curated public API surface.
   It should not simply wildcard-import everything from every module.

This structure makes it much easier to reason about what is foundational versus what is a wrapper, adapter, or convenience layer.

## File Scope Rules

The following rules should govern how current and future code is grouped:

1. One major public class per file whenever practical.

2. Pure functions should be grouped by mathematical or simulation topic.
   For example, PCA-related functions should live together, but PCA code should not share a file with family-relationship helpers just because both are "population genetics."

3. Plotting functions should live in plotting modules, not next to the main computation by default.

4. Export and file-format logic should live in `io/`.

5. Data-only lookup tables or constants can stay in small dedicated files.

6. Private helper functions should usually live next to the public function or class they support, unless they are reused across multiple modules.

7. A module should be small enough that a future documentation page could plausibly cite it as the canonical implementation of one concept.

## Specific Proposed Moves

### 1. `core_functions.py`

This file is currently a mix of statistics helpers, convenience helpers, and plotting helpers.

Proposed moves:

1. `corr` -> `utils/stats.py`
2. `report_CI` -> `utils/stats.py`
3. `to_bits` -> `utils/misc.py`
4. `get_pop_kwargs` -> `simulation` helper logic or `utils/misc.py`
5. `_get_default_colors` -> `plotting/common.py`
6. `plot_over_time` -> `plotting/common.py`

Rationale:

1. The statistical helpers do not belong in a package-wide miscellaneous file.
2. The plotting helpers should not live with non-plotting utilities.
3. `get_pop_kwargs` is only meaningful because of simulation orchestration and should live near the code that uses it.

### 2. `export_functions.py`

This file is already compact and coherent, but it should still move into a dedicated I/O namespace.

Proposed moves:

1. `export_GRM_GCTA` -> `io/gcta.py`
2. `export_trait` -> `io/gcta.py`

Rationale:

1. These functions implement serialization to a specific external format.
2. Keeping them in `io/` makes future exporters easier to add cleanly.

### 3. `popgen_functions.py`

This file currently contains several distinct conceptual domains and should be split substantially.

#### 3.1 Genotype handling and basic matrix transforms

Move to `genetics/genotypes.py`:

1. `make_G`
2. `compute_freqs`
3. `center_G`
4. `standardize_G`
5. `compute_GRM`

These are foundational genotype-matrix operations and should be easy to locate as a coherent set.

#### 3.2 PCA

Move to `genetics/pca.py`:

1. `PCAResult`
2. `_validate_pca_axes`
3. `_orient_pca_scores`
4. `_format_pc_axis_label` if retained as an internal helper close to PCA logic
5. `compute_PCA`

Move plotting pieces to `plotting/genetics.py`:

1. `plot_PCA`

Rationale:

1. PCA computation is a self-contained conceptual area.
2. The `PCAResult` data object belongs with the PCA computation rather than a giant mixed file.
3. The plotting function should be separated from the core decomposition logic.

#### 3.3 Site-frequency and allele-frequency summaries

Move to `genetics/frequencies.py`:

1. `draw_p_FST`
2. `get_FST`
3. `get_fixation_t`
4. `summarize_ps`

Move `plot_site_frequency_spectrum` to `plotting/genetics.py`.

If desired, site-frequency plotting could remain documented alongside frequency analysis, but implementation-wise it should still live in the plotting layer.

#### 3.4 LD and neighborhood matrices

Move to `genetics/ld.py`:

1. `make_neighbor_matrix`
2. `compute_corr_matrix`
3. `compute_LD_matrix`

These functions all describe one coherent workflow and should stay together.

#### 3.5 Genome structure and haplotype-generation helpers

Move to `genome/structure.py`:

1. `draw_binom_haplos`
2. `generate_LD_blocks`
3. `generate_chromosomes`
4. `draw_p_init`

Rationale:

1. These functions define the structure of the genome and the joint distributional assumptions used when simulating haplotypes.
2. They are conceptually separate from downstream genotype statistics and therefore fit better in a `genome` domain than in general `genetics`.

#### 3.6 Compact family-relation helpers

Move to `pedigree/relations.py`:

1. `initialize_relations`
2. `get_relation_matrix`

These are not really genotype-analysis functions; they are pedigree/relationship utilities.

#### 3.7 IBD and relatedness utilities

Move to `pedigree/ibd.py`:

1. `get_true_IBD1`
2. `get_true_IBD_tensor`
3. `get_true_IBD_arr`
4. `get_coeff_kinship`
5. `get_coeff_inbreeding`
6. `get_coeff_relatedness`
7. `compute_K_IBD`
8. `greedy_unrelated_subset`
9. `IBDSegment`
10. `IBD_tensor_to_segments`
11. `extract_crossover_points`
12. `extract_all_crossover_points`

These functions are conceptually tied and should be documented as an IBD/relatedness toolkit.

### 4. `statgen_functions.py`

This file contains at least four distinct areas and should be split accordingly.

#### 4.1 Effect-size and trait-component sampling

Move to `traits/effect_sampling.py`:

1. `generate_causal_effects`
2. `generate_genetic_effects`
3. `compute_genetic_value`
4. `generate_noise_value`
5. `get_G_std_for_effects`
6. `get_standardized_effects`

These belong together as effect-generation and scaling logic.

#### 4.2 Fixed-effect utilities

Move to `traits/fixed_effects.py`:

1. `scale_binary_FE`

This may remain a small file, which is fine if it clarifies the boundary.

#### 4.3 Random-effect and kernel helpers

Move to `traits/random_effects.py`:

1. `psd_sqrt`
2. `nearest_correlation_matrix`
3. `build_design_matrix_from_groups`
4. `_standardize_correlation_matrix`
5. `_center_and_scale_random_effect`
6. `is_identity_matrix`
7. `get_group_assignments_from_design`
8. `apply_identity_cluster_kernel_sqrt`
9. `get_identity_cluster_kernel_trace`
10. `_calibrate_random_fixed_loading_from_propagated`
11. `_get_kappa`
12. `_validate_random_effect_inputs`
13. `_calibrate_random_fixed_loading`
14. `get_random_effects`

These functions form a clear internal subsystem and should live together.

#### 4.4 Heritability estimation wrappers and plotting

Move to `plotting/estimation.py`:

1. `plot_HE_regression`

The bottom-of-file aliases:

1. `run_HEreg`
2. `run_HE_regression`
3. `run_REML`

should no longer be re-exported from `statgen_functions.py`. Instead, estimation APIs should live under `estimation/` and be re-exported deliberately from `estimation/__init__.py` and, if desired, the package root.

### 5. `reml.py`

This file is more coherent than some of the others, but it is large enough that splitting it still makes sense.

Proposed structure:

1. `estimation/_common.py`
   Shared input validation, covariance construction, and shared linear-algebra helpers.

2. `estimation/he.py`
   Haseman-Elston-specific internals and the user-facing `run_HEreg`.

3. `estimation/reml.py`
   AI-REML, EM-REML, exact/stochastic paths, and the user-facing `run_REML`.

4. Keep `_reml_accel.c` internal to the estimation layer.

Rationale:

1. HE and REML are related, but still distinct enough to deserve their own top-level modules.
2. Shared internal helpers should be separated from the entrypoints they support.
3. The compiled extension is an implementation detail of the estimation subsystem.

### 6. `popsim.py`

This is the most important file to split because it currently carries too many responsibilities.

#### 6.1 Simulation objects

Move to `simulation/population.py`:

1. `Population`

Move to `simulation/superpopulation.py`:

1. `SuperPopulation`

These are high-level user-facing orchestrator classes and should not share a file with all effect and pedigree classes.

#### 6.2 Trait and effect classes

Move to `traits/effects.py`:

1. `Effect`
2. `GeneticEffect`
3. `FixedEffect`
4. `CorrelatedRandomEffect`
5. `NoiseEffect`

Move to `traits/trait.py`:

1. `Trait`

Rationale:

1. These types all belong to the trait-generation domain rather than the population-simulation domain.
2. Splitting `Trait` from the individual effect classes is justified because both areas are now substantial.

#### 6.3 Pedigree data structures

Move to `pedigree/pedigree.py`:

1. `RelObj`
2. `Pedigree`

This is a clear pedigree subsystem and should not live at the bottom of `popsim.py`.

#### 6.4 Important note about `Trait` and `Population`

The proposal is not that `Trait` must become completely ignorant of `Population`. Instead, the goal is:

1. `Trait` may still keep a light reference to its owning population when that is useful.
2. `Population` may still resolve trait-to-trait dependencies and supply arrays or named inputs.
3. The deepest helper functions used by `Trait` should operate on explicit data rather than requiring broad access to `Population` internals.

So if a trait component depends on another trait's values, that functionality can remain. The difference is that the dependency should be made more explicit and localized, rather than relying on a large monolithic module to implicitly connect everything.

### 7. `relative_types.py`

Move to `pedigree/relative_types.py`.

This file is already small and conceptually clear, so it should stay data-only.

## Public API Philosophy

Not every importable symbol needs to be part of the documented, supported public API.

The package should distinguish between:

1. supported public interfaces,
2. advanced-but-undocumented module-level functions,
3. internal helpers.

### Recommended public API

The root package should likely expose a curated set of major public symbols such as:

1. `Population`
2. `SuperPopulation`
3. `Trait`
4. `GeneticEffect`
5. `FixedEffect`
6. `CorrelatedRandomEffect`
7. `NoiseEffect`
8. `PCAResult`
9. `run_REML`
10. `run_HEreg`

Additional domain namespaces should also be intentionally exposed, for example:

1. `popstatgensim.genetics`
2. `popstatgensim.traits`
3. `popstatgensim.estimation`
4. `popstatgensim.io`
5. `popstatgensim.plotting`

### Internal helpers

Many helper functions will still be importable by knowledgeable users because Python allows that, but they should not all be treated as stable promises. The package should signal internal/private status through:

1. leading-underscore names,
2. omission from the curated package root,
3. omission from user-facing documentation.

## Documentation Structure Philosophy

This proposal recommends creating `docs/architecture/` as a maintainer-facing part of the docs tree.

The intended distinction is:

1. `docs/architecture/`
   Design notes, refactor plans, dependency rules, class/module responsibility guides, and codebase structure documentation.

2. Future user-facing docs folders such as `docs/tutorials/`, `docs/examples/`, `docs/api/`, `docs/vignettes/`, or similar
   User guidance, scientific explanations, API docs, worked examples, and educational material.

The point of an architecture folder is to preserve project-level reasoning that helps future refactors and future Codex sessions stay aligned with the intended structure of the package.

## Testing Philosophy

The repository currently has notebooks under `tests/`, but not a conventional automated test suite.

The refactor should establish `tests/` as the place for automated regression and unit tests, likely using `pytest`.

### What automated tests would do

A typical test file would:

1. create a small deterministic input,
2. run one function or workflow,
3. assert that the output shape, values, error behavior, or invariants match expectations.

### What kinds of tests should be added

1. Unit tests for pure genetics utilities.
   Example: genotype standardization, GRM computation, PCA shape and variance outputs.

2. Unit tests for trait/effect helpers.
   Example: effect-size scaling, random-effect construction, fixed-effect scaling.

3. Unit tests for pedigree and IBD utilities.
   Example: relation classification, IBD tensor behavior, kinship-related outputs.

4. Numerical regression tests for estimation.
   Example: `run_HEreg` and `run_REML` on small seeded problems with known or previously recorded outputs.

5. Integration tests for high-level workflows.
   Example: constructing a `Population`, adding traits, simulating generations, building a `SuperPopulation`, and checking that the main workflow still behaves as expected.

### How tests support refactoring

Yes, tests should be used specifically to preserve behavior before and after the structural changes.

The ideal strategy is:

1. identify representative current behaviors,
2. capture them in deterministic tests,
3. refactor the code structure,
4. confirm the same behaviors still hold.

Because many parts of this package are stochastic, tests should often use fixed seeds and tolerance-based numerical checks rather than exact comparisons of every floating-point value.

### What should happen to the notebooks

The current notebooks should eventually be moved out of `tests/` and into a location such as:

1. `examples/`
2. `notebooks/`

They are valuable as exploratory demonstrations, but they should not be the primary automated validation mechanism.

## Git and Commit Strategy

This refactor should be implemented in the dedicated worktree branch only.

A good implementation strategy would involve intermediate commits grouped by coherent changes, for example:

1. establish new package directories and move low-risk utility modules,
2. split genetics modules,
3. split trait/effect modules,
4. split simulation and pedigree modules,
5. split estimation modules,
6. update imports and package exports,
7. add automated tests and move notebooks.

Intermediate commits would make it easier to:

1. review the refactor in pieces,
2. revert or revise one subsystem if needed,
3. keep a clean history of how the restructuring was performed.

## Recommended Order of Implementation

To minimize breakage during the actual refactor, the implementation should likely proceed in this order:

1. Create the new subpackage directories and `__init__.py` files.
2. Move low-risk pure-function modules first:
   `utils`, `io`, `plotting`, and small genetics helpers.
3. Split `popgen_functions.py` into domain files.
4. Split `statgen_functions.py` into domain files.
5. Split `popsim.py` into `simulation/`, `traits/`, and `pedigree/`.
6. Split `reml.py` into `estimation/` submodules.
7. Replace wildcard exports with a curated public API.
8. Add or expand automated tests to lock in behavior.
9. Move notebooks from `tests/` to a more appropriate location.

## Final Recommendation

The package should move away from a small-number-of-large-files design and toward a layered, domain-oriented structure where:

1. core scientific logic is easy to find and document,
2. classes live in files that match their conceptual roles,
3. plotting and exporting are separated from numerical logic,
4. dependencies flow from low-level math upward into simulation orchestration,
5. future growth can happen by adding coherent modules rather than continuing to enlarge monolithic files.

This should make the codebase more legible both for you and for future user-facing documentation, while still preserving the current scientific and simulation functionality as closely as possible.

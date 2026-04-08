# Refactoring Changes Log

This file tracks the structural refactor as it is implemented.

## Phase 0: Setup

Date: 2026-04-08

Completed:

1. Created `docs/architecture/`.
2. Wrote the main proposal in `docs/architecture/refactor_proposal.md`.
3. Updated the proposal to use the `genome` domain instead of `recombination`.
4. Created this running change log.
5. Created `docs/architecture/refactor_efficiency_proposals.md` for non-implemented performance ideas noticed during refactoring.

Planned next:

1. Create the new package subdirectories.
2. Split low-level utilities, plotting, I/O, and domain helpers out of the monolithic modules.
3. Continue logging major structural changes here as they happen.

## Phase 1: New package skeleton and low-level module split

Completed:

1. Created the new subpackages:
   `utils`, `io`, `plotting`, `genome`, `pedigree`, `traits`, `simulation`, and `estimation`.
2. Split low-level helper code out of the old monolithic files into new implementation modules:
   `utils/stats.py`, `utils/misc.py`, `plotting/common.py`, `io/gcta.py`,
   `genome/genotypes.py`, `genome/pca.py`, `genome/ld.py`, `genome/frequencies.py`,
   `genome/structure.py`, `pedigree/relations.py`, `pedigree/ibd.py`,
   `traits/effect_sampling.py`, `traits/fixed_effects.py`, `traits/random_effects.py`,
   `plotting/genome.py`, and `plotting/estimation.py`.
3. Collapsed `core_functions.py`, `export_functions.py`, `popgen_functions.py`, and `statgen_functions.py` into compatibility wrappers that now re-export the new implementation modules.

## Phase 2: Class/module split and package API

Completed:

1. Split `popsim.py` into:
   `simulation/population.py`, `simulation/superpopulation.py`,
   `traits/effects.py`, `traits/trait.py`, and `pedigree/pedigree.py`.
2. Moved relationship metadata into `pedigree/relative_types.py`.
3. Replaced the old `popsim.py` and `relative_types.py` files with thin compatibility wrappers.
4. Added subpackage-level `__init__.py` exports so the new structure is importable as namespaces.
5. Replaced the root package `__init__.py` wildcard exports with a curated API that exposes:
   major public classes, estimation entrypoints, export helpers, and the main subpackages.
6. Added the `estimation/` namespace and routed its public entrypoints through
   `estimation/he.py`, `estimation/reml.py`, and `estimation/_common.py`.

Validation performed:

1. Ran `python3 -m py_compile` across the root package and all subpackage modules.
2. Ran import smoke tests using the project venv and confirmed the root package and major subpackages import successfully.

Planned next:

1. Add automated regression-oriented tests for the refactored package structure.
2. Move the current notebooks out of `tests/` into a more appropriate examples/notebooks location.
3. Write a maintainer-facing repository structure document under `docs/architecture/`.
4. Continue making staged git commits at coherent milestones.

## Phase 3: Verification and repository cleanup

Completed:

1. Added pytest-based automated checks under `tests/` covering:
   root API imports, subpackage exposure, genome helpers, trait/population workflows, and estimation entrypoints.
2. Added `tests/conftest.py` to make the `src/` layout importable under pytest and to stabilize matplotlib test behavior.
3. Moved exploratory notebooks from `tests/` to `notebooks/`.
4. Wrote `docs/architecture/repository_structure.md` as a future maintainer guide for the refactored repo layout.

Validation performed:

1. Ran `/n/groups/price/nuno/.venv_py13/bin/python -m pytest /n/groups/price/nuno/psgs-codex/tests -q`
2. Confirmed all tests pass after moving notebooks out of `tests/`.

Planned next:

1. Create the next git checkpoint for tests, docs, and repo cleanup.
2. Review whether any additional structural cleanup is needed before stopping.

## Phase 4: Remove legacy compatibility modules

Completed:

1. Deleted the old flat compatibility modules from the package root:
   `core_functions.py`, `export_functions.py`, `popgen_functions.py`,
   `popsim.py`, `relative_types.py`, `statgen_functions.py`, and the old root `reml.py`.
2. Updated internal imports so package code now points directly to the new domain modules.
3. Moved the REML implementation into `src/popstatgensim/estimation/reml.py`.
4. Moved the REML C accelerator into `src/popstatgensim/estimation/_reml_accel.c`.
5. Updated `setup.py` so the compiled extension now builds as `popstatgensim.estimation._reml_accel`.
6. Added tests asserting that the removed legacy flat modules are no longer importable.

Validation performed:

1. Ran `python3 -m py_compile` across the root package and all subpackages.
2. Ran import smoke tests for the root package and the `simulation` / `estimation` namespaces.

Planned next:

1. Run the full pytest suite after the legacy-module removal.
2. Create a final cleanup commit for the no-compatibility package layout.

## Phase 5: Merge genome analysis into the `genome` namespace

Completed:

1. Moved the genotype-analysis modules into `src/popstatgensim/genome/` so genome structure, genotype transforms, PCA, LD, and allele-frequency summaries now live under one namespace.
2. Moved `draw_p_FST()` into `genome/structure.py` alongside `draw_p_init()`.
3. Moved `compute_freqs()` into `genome/frequencies.py`.
4. Renamed the plotting helper module from `plotting/genetics.py` to `plotting/genome.py`.
5. Removed the `src/popstatgensim/genetics/` package and updated the root API to expose only `genome`.
6. Updated tests and maintainer docs to reflect the single-domain genome layout.

Validation performed:

1. Run `python3 -m py_compile` across the package after the namespace merge.
2. Run the pytest suite after replacing the old `genetics` imports.

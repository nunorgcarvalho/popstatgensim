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
2. Split low-level utilities, plotting, I/O, genetics, and genome helpers out of the monolithic modules.
3. Continue logging major structural changes here as they happen.

## Phase 1: New package skeleton and low-level module split

Completed:

1. Created the new subpackages:
   `utils`, `io`, `plotting`, `genetics`, `genome`, `pedigree`, `traits`, `simulation`, and `estimation`.
2. Split low-level helper code out of the old monolithic files into new implementation modules:
   `utils/stats.py`, `utils/misc.py`, `plotting/common.py`, `io/gcta.py`,
   `genetics/genotypes.py`, `genetics/pca.py`, `genetics/ld.py`, `genetics/frequencies.py`,
   `genome/structure.py`, `pedigree/relations.py`, `pedigree/ibd.py`,
   `traits/effect_sampling.py`, `traits/fixed_effects.py`, `traits/random_effects.py`,
   and `plotting/estimation.py`.
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
   root API imports, subpackage exposure, genetics/genome helpers, trait/population workflows, and estimation entrypoints.
2. Added `tests/conftest.py` to make the `src/` layout importable under pytest and to stabilize matplotlib test behavior.
3. Moved exploratory notebooks from `tests/` to `notebooks/`.
4. Wrote `docs/architecture/repository_structure.md` as a future maintainer guide for the refactored repo layout.

Validation performed:

1. Ran `/n/groups/price/nuno/.venv_py13/bin/python -m pytest /n/groups/price/nuno/psgs-codex/tests -q`
2. Confirmed all tests pass after moving notebooks out of `tests/`.

Planned next:

1. Create the next git checkpoint for tests, docs, and repo cleanup.
2. Review whether any additional structural cleanup is needed before stopping.

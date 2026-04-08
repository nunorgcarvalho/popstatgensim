# Repository Structure Guide

## Purpose

This document describes the intended structure of the `popstatgensim` repository for future maintenance and extension.

The repository is organized so that:

1. scientific logic lives in small domain-oriented modules,
2. plotting and file export stay separate from core computation,
3. high-level simulation classes orchestrate lower-level helpers,
4. public imports are curated at the package root and subpackage boundaries.

## Top-Level Layout

```text
popstatgensim/
  docs/
    architecture/
  notebooks/
  src/
    popstatgensim/
  tests/
```

### `docs/architecture/`

Maintainer-facing design documents live here. This includes:

1. structure and dependency guidance,
2. refactor proposals and change logs,
3. deferred efficiency ideas,
4. other internal design notes that help future maintenance stay consistent.

### `notebooks/`

Exploratory and example notebooks live here. These are useful for demonstrations, manual exploration, and scientific checks, but they are not the primary automated verification layer.

### `tests/`

Automated regression and smoke tests live here. New functionality should generally add or update pytest coverage in this directory.

## Package Layout

Inside `src/popstatgensim/`, the codebase is organized by domain:

### `genetics/`

Genotype-matrix operations and downstream genetics analyses.

Typical contents:

1. genotype transformations,
2. GRM construction,
3. PCA computation,
4. LD helpers,
5. allele-frequency summaries.

### `genome/`

Genome structure and haplotype-generation assumptions.

Typical contents:

1. initial allele-frequency generation,
2. haplotype simulation,
3. chromosome/recombination-map construction,
4. genome-structure helpers used before downstream genetics analysis.

### `pedigree/`

Relationship, pedigree, and IBD logic.

Typical contents:

1. compact relation encodings,
2. pedigree path construction,
3. relationship classification,
4. true-IBD utilities,
5. pedigree-specific constants and lookup tables.

### `traits/`

Trait and effect definitions.

Typical contents:

1. effect classes,
2. the `Trait` class,
3. effect-size sampling,
4. fixed-effect helpers,
5. random-effect kernel and sampling helpers.

### `simulation/`

High-level user-facing simulation classes.

Typical contents:

1. `Population`,
2. `SuperPopulation`,
3. stateful orchestration over genetics, pedigree, and trait subsystems.

### `estimation/`

Heritability and variance-component estimation entrypoints.

Typical contents:

1. REML entrypoints,
2. HE entrypoints,
3. shared estimation helpers exposed through the estimation namespace.

### `plotting/`

Visualization helpers only.

Typical contents:

1. common plotting helpers,
2. genetics plots,
3. estimation plots.

Plotting code should not become the authoritative home of scientific computations.

### `io/`

Export and serialization logic.

Typical contents:

1. GCTA-compatible file writers,
2. future data-export formats.

### `utils/`

Small generic helpers used across multiple domains.

Typical contents:

1. statistical formatting helpers,
2. small general-purpose utilities.

## Root Package Files

The package root contains the curated public API in `__init__.py`.

Implementation code should live in the domain subpackages, not in additional flat modules at the package root. New functionality should be added by extending the relevant domain package rather than by introducing one-off root-level helper files.

## Dependency Direction

Preferred dependency flow:

1. `utils` is the bottom layer.
2. `genetics`, `genome`, `pedigree`, `traits`, and `estimation` build on `utils`.
3. `simulation` depends on domain layers and coordinates user-facing workflows.
4. `plotting` and `io` depend on outputs from computational layers.
5. Lower-level computation should not depend on plotting or export code.

## File Placement Rules

When adding or moving code, use these rules:

1. One major public class per file whenever practical.
2. Group pure functions by domain concept, not by broad labels like "miscellaneous functions."
3. Keep plotting separate from computation.
4. Keep export/file-format code separate from scientific logic.
5. Keep data-only constants or lookup tables in small dedicated files.
6. Keep private helpers near the public function/class they support unless they are genuinely shared.

## Public API Philosophy

The package root should expose the major public classes and entrypoints, while the subpackages expose the more detailed domain APIs.

In practice:

1. end users should be able to work from `popstatgensim` and the major domain namespaces,
2. maintainers should prefer explicit imports from subpackages when editing internal code,
3. underscore-prefixed helpers should be treated as internal implementation details.

## Testing Expectations

The automated test suite should protect behavior during future refactors.

Recommended test layers:

1. unit tests for pure helper functions,
2. workflow tests for `Population`, `Trait`, and `SuperPopulation`,
3. numerical regression tests for estimation entrypoints,
4. smoke tests for the curated public API and namespace imports.

Whenever possible, new structural changes should be accompanied by tests that verify current behavior rather than relying only on notebooks or manual inspection.

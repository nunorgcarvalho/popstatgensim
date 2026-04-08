# Refactor Efficiency Proposals

This file records possible efficiency improvements noticed during the structural refactor.

These are intentionally not being implemented as part of the refactor itself. The goal is to keep the current refactor focused on package structure, file boundaries, imports, and maintainability rather than changing numerical behavior or performance characteristics.

## Notes

1. `pedigree/ibd.py:compute_K_IBD()`
   The current implementation computes relatedness for every pair of individuals with nested Python loops. If true-IBD workflows become common on larger populations, this could likely be accelerated by vectorizing some haplotype comparisons or by caching pairwise summaries.

2. `genetics/ld.py:make_neighbor_matrix()`
   The current implementation appends row/column indices in Python loops for every variant and every neighbor within the LD window. For larger marker sets, a more vectorized window-construction approach or blockwise sparse assembly could reduce Python overhead.

3. `genetics/ld.py:compute_corr_matrix()`
   Correlations for neighboring variants are computed entry-by-entry in Python. If LD analyses become a bottleneck, this could likely be sped up with blockwise linear algebra on local slices of `X` rather than scalar sparse construction.

# Benchmarking `run_REML()`

## Methods

Benchmarks used the simulation recipe from `IRI/scripts/simulations/sandbox/sandbox_4_benchmark.ipynb`: `M=2000`, `N` varied, one simulated generation with related offspring, trait variance components `V_A=0.4`, `V_A_par=0.1`, `V_Eps=0.5`, and `M_causal=500`.
For each `N`, the population, phenotype, and GRMs were simulated once and cached. Each benchmark then loaded the cached arrays and measured only the `run_REML()` call in a fresh Python subprocess. Peak RSS was sampled during the fit after the arrays were already resident in memory, so `peak_rss_fit_gb` captures the actual process footprint during REML rather than the simulation step.
The main benchmark factors were population size (`N`), number of fitted relationship matrices (1-matrix GREML versus 3-matrix RDR-SNP), REML method (`AI_stochastic`, `AI`, `FS`), and `safety_checks` (`True` versus `False`). Additional sensitivity studies varied `n_probes`, `dtype`, and the compiled stochastic accelerator for `AI_stochastic`.

## Results

A total of 70 benchmark rows were recorded, of which 70 completed successfully and 0 ended in error or were skipped after a smaller run in the same family hit a practical resource limit.

### Core Summary

| suite | phase | model | requested_method | safety_checks | dtype | n_probes | use_accelerator | N | elapsed_s | peak_rss_fit_gb | peak_incremental_rss_fit_gb | iterations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | core_scaling | GREML | AI_stochastic | True | float32 | 50 | True | 7000 | 93.236 | 2.520 | 2.031 | 5.000 |
| main | core_scaling | GREML | AI_stochastic | True | float32 | 50 | True | 10000 | 250.161 | 4.997 | 4.130 | 5.000 |
| main | core_scaling | GREML | AI_stochastic | True | float32 | 50 | True | 13000 | 369.553 | 8.350 | 6.967 | 5.000 |
| main | core_scaling | GREML | AI_stochastic | True | float32 | 50 | True | 16000 | 693.520 | 12.569 | 10.541 | 5.000 |
| main | core_scaling | GREML | AI_stochastic | True | float32 | 50 | True | 20000 | 1212.847 | 19.551 | 16.448 | 5.000 |
| main | core_scaling | RDR-SNP | AI_stochastic | True | float32 | 50 | True | 7000 | 79.872 | 4.347 | 3.128 | 5.000 |
| main | core_scaling | RDR-SNP | AI_stochastic | True | float32 | 50 | True | 10000 | 201.340 | 8.731 | 6.372 | 5.000 |
| main | core_scaling | RDR-SNP | AI_stochastic | True | float32 | 50 | True | 13000 | 464.965 | 14.654 | 10.753 | 6.000 |
| main | core_scaling | RDR-SNP | AI_stochastic | True | float32 | 50 | True | 16000 | 713.563 | 22.112 | 16.266 | 5.000 |
| main | core_scaling | RDR-SNP | AI_stochastic | True | float32 | 50 | True | 20000 | 1305.090 | 34.455 | 25.391 | 5.000 |
| main | dtype_check | GREML | AI_stochastic | True | float64 | 50 | True | 7000 | 95.030 | 2.341 | 1.853 | 5.000 |
| main | dtype_check | RDR-SNP | AI_stochastic | True | float64 | 50 | True | 7000 | 169.285 | 3.803 | 2.584 | 5.000 |
| main | n_probes_scaling | GREML | AI_stochastic | True | float32 | 10 | True | 7000 | 63.404 | 2.515 | 2.027 | 5.000 |
| main | n_probes_scaling | GREML | AI_stochastic | True | float32 | 200 | True | 7000 | 69.457 | 2.545 | 2.056 | 5.000 |
| main | n_probes_scaling | RDR-SNP | AI_stochastic | True | float32 | 10 | True | 7000 | 75.047 | 4.341 | 3.122 | 5.000 |
| main | n_probes_scaling | RDR-SNP | AI_stochastic | True | float32 | 200 | True | 7000 | 86.018 | 4.371 | 3.152 | 5.000 |
| main | safety_scaling | GREML | AI_stochastic | False | float32 | 50 | True | 10000 | 82.693 | 5.001 | 4.132 | 5.000 |
| main | safety_scaling | GREML | AI_stochastic | False | float32 | 50 | True | 16000 | 307.507 | 12.572 | 10.542 | 5.000 |
| main | safety_scaling | RDR-SNP | AI_stochastic | False | float32 | 50 | True | 10000 | 102.448 | 8.731 | 6.372 | 5.000 |
| main | safety_scaling | RDR-SNP | AI_stochastic | False | float32 | 50 | True | 16000 | 364.245 | 22.112 | 16.266 | 5.000 |

### Extrapolated N=50,000 Performance

| model | safety_checks | time_scale | time_exponent | rss_intercept_gb | rss_per_n2_gb | predicted_time_at_50000_s | predicted_peak_rss_at_50000_gb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GREML | False | 3.142e-09 | 2.612 | 0.136 | 4.859e-08 | 5906.465 | 121.608 |
| GREML | True | 2.340e-09 | 2.734 | 0.140 | 4.854e-08 | 16478.275 | 121.491 |
| RDR-SNP | False | 8.352e-09 | 2.543 | 0.143 | 8.583e-08 | 7427.842 | 214.708 |
| RDR-SNP | True | 6.325e-09 | 2.635 | 0.147 | 8.579e-08 | 15263.957 | 214.610 |

![core_scaling](results/figures/core_scaling.png)

![method_comparison](results/figures/method_comparison.png)

![n_probes_tradeoff](results/figures/n_probes_tradeoff.png)

![accelerator_compare](results/figures/accelerator_compare.png)

![scaling_extrapolation](results/figures/scaling_extrapolation.png)

## Discussion

The dominant cost drivers should be interpreted in the context of the current implementation: `run_REML()` materializes dense `N x N` covariance matrices and the exact methods explicitly form inverses and projection matrices. That makes the number of fitted relationship matrices a direct memory multiplier and makes exact methods much less scalable than `AI_stochastic`.
Extrapolation from the successful `AI_stochastic` runs suggests the following approximate process footprints at `N=50,000`:
- GREML, safety_checks=False: 1.64 hours and 121.6 GB peak RSS.
- GREML, safety_checks=True: 4.58 hours and 121.5 GB peak RSS.
- RDR-SNP, safety_checks=False: 2.06 hours and 214.7 GB peak RSS.
- RDR-SNP, safety_checks=True: 4.24 hours and 214.6 GB peak RSS.
The fitted extrapolations are empirical rather than guaranteed. They are most trustworthy near the observed `N` range and less trustworthy once fallback behavior, non-convergence, or OS-level memory pressure starts to matter.
If these benchmarks will be used for Slurm resource requests, a conservative rule is to request somewhat more than the observed peak RSS because the current measurement excludes the one-time simulation step but still depends on BLAS/LAPACK workspace and allocator behavior on the target node.

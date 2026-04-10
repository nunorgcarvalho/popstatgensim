# REML Benchmark Workspace

This folder contains scripts and outputs for benchmarking `popstatgensim.run_REML()`
using the simulation setup from `IRI/scripts/simulations/sandbox/sandbox_4_benchmark.ipynb`.

The workflow is:

1. Simulate and cache one dataset per `N`.
2. Benchmark only `run_REML()` on the cached matrices/phenotype.
3. Compare optional performance toggles such as the compiled stochastic accelerator.
4. Analyze the benchmark CSV into tables, figures, and a report.

Recommended commands:

```bash
/n/groups/price/nuno/.venv_py13/bin/python notebooks/benchmark_REML_04-10/run_benchmarks.py --suite pilot
/n/groups/price/nuno/.venv_py13/bin/python notebooks/benchmark_REML_04-10/run_benchmarks.py --suite main
/n/groups/price/nuno/.venv_py13/bin/python notebooks/benchmark_REML_04-10/analyze_benchmarks.py
```

Important output files:

- `results/benchmark_runs.csv`: one row per benchmark run
- `results/figures/*.png`: generated figures
- `results/tables/*.csv`: derived summary tables
- `report.md`: Markdown report in Methods -> Results -> Discussion style

Ignored large artifacts:

- `cache/`: cached datasets (`.npz`, `.json`)
- `results/logs/`: per-run stdout/stderr logs

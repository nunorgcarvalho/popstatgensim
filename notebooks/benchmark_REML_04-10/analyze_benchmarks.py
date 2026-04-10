from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BENCHMARK_ROOT = Path(__file__).resolve().parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from benchmark_reml_utils import FIGURE_DIR, REPORT_PATH, RESULTS_CSV, TABLE_DIR, ensure_directories


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    coef = np.polyfit(np.log(x), np.log(y), 1)
    exponent = float(coef[0])
    scale = float(np.exp(coef[1]))
    return scale, exponent


def fit_quadratic_memory(n_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    x = n_values.astype(float) ** 2
    coef = np.polyfit(x, y_values.astype(float), 1)
    slope = float(coef[0])
    intercept = float(coef[1])
    return intercept, slope


def predict_time(scale: float, exponent: float, n_value: float | np.ndarray) -> float | np.ndarray:
    return scale * (n_value ** exponent)


def predict_peak_rss(intercept: float, slope: float, n_value: float | np.ndarray) -> float | np.ndarray:
    return intercept + slope * (n_value ** 2)


def save_table(df: pd.DataFrame, name: str) -> None:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_No rows_"
    formatted = df.copy()

    def _format_float(x: float) -> str:
        if pd.isna(x):
            return ""
        x = float(x)
        if x != 0.0 and abs(x) < 1e-3:
            return f"{x:.3e}"
        return format(x, floatfmt)

    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(_format_float)
        else:
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(col) for col in formatted.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def make_core_scaling_figure(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    metrics = [("elapsed_s", "Elapsed Time (s)"), ("peak_rss_fit_gb", "Peak RSS During Fit (GB)")]
    colors = {
        ("GREML", True): "#1f77b4",
        ("GREML", False): "#7fb6e6",
        ("RDR-SNP", True): "#d62728",
        ("RDR-SNP", False): "#f2a2a2",
    }
    labels = {
        ("GREML", True): "GREML, safety=True",
        ("GREML", False): "GREML, safety=False",
        ("RDR-SNP", True): "RDR-SNP, safety=True",
        ("RDR-SNP", False): "RDR-SNP, safety=False",
    }
    core = df[
        (df["requested_method"] == "AI_stochastic")
        & (df["phase"].isin(["core_scaling", "safety_scaling"]))
        & (df["status"] == "ok")
    ].copy()
    if core.empty:
        return FIGURE_DIR / "core_scaling.png"

    summary = (
        core.groupby(["model", "safety_checks", "N"], as_index=False)[
            ["elapsed_s", "peak_rss_fit_gb"]
        ]
        .median()
        .sort_values(["model", "safety_checks", "N"])
    )
    for ax, (metric, ylabel) in zip(axes, metrics):
        for (model, safety_checks), group in summary.groupby(["model", "safety_checks"]):
            group = group.sort_values("N")
            ax.plot(
                group["N"],
                group[metric],
                marker="o",
                linewidth=2,
                color=colors[(model, bool(safety_checks))],
                label=labels[(model, bool(safety_checks))],
            )
        ax.set_xlabel("N")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_title("AI_stochastic with safety_checks=True/False", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    path = FIGURE_DIR / "core_scaling.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_method_comparison_figure(df: pd.DataFrame) -> Path:
    subset = df[
        (df["status"] == "ok")
        & (df["requested_method"].isin(["AI_stochastic", "AI", "FS"]))
        & (df["N"].isin(sorted(df["N"].unique())[:3]))
        & (df["safety_checks"] == True)
    ].copy()
    if subset.empty:
        return FIGURE_DIR / "method_comparison.png"

    grouped = (
        subset.groupby(["N", "model", "requested_method"], as_index=False)[
            ["elapsed_s", "peak_rss_fit_gb"]
        ]
        .median()
        .sort_values(["N", "model", "requested_method"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    methods = ["AI_stochastic", "AI", "FS"]
    width = 0.25
    for panel, metric in enumerate(["elapsed_s", "peak_rss_fit_gb"]):
        ax = axes[panel]
        xlabels = []
        positions = np.arange(len(grouped.groupby(["N", "model"])))
        for i, method in enumerate(methods):
            values = []
            for (N, model), group in grouped.groupby(["N", "model"]):
                row = group[group["requested_method"] == method]
                values.append(float(row.iloc[0][metric]) if not row.empty else np.nan)
                if i == 0:
                    xlabels.append(f"N={N}\n{model}")
            ax.bar(positions + (i - 1) * width, values, width=width, label=method)
        ax.set_xticks(positions)
        ax.set_xticklabels(xlabels)
        ax.set_ylabel("Elapsed Time (s)" if metric == "elapsed_s" else "Peak RSS During Fit (GB)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("Accelerator comparison, safety_checks=True only", fontsize=11)
        ax.set_title("Method comparison, safety_checks=True only", fontsize=11)
    axes[0].legend(frameon=False)
    path = FIGURE_DIR / "method_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_n_probes_figure(df: pd.DataFrame) -> Path:
    subset = df[
        (df["status"] == "ok")
        & (df["requested_method"] == "AI_stochastic")
        & (df["phase"].isin(["n_probes_compare", "n_probes_scaling"]))
        & (df["dtype"] == "float32")
        & (df["safety_checks"] == True)
    ].copy()
    if subset.empty:
        return FIGURE_DIR / "n_probes_tradeoff.png"

    target_n = int(subset["N"].median())
    subset = subset[subset["N"] == target_n]
    grouped = (
        subset.groupby(["model", "n_probes"], as_index=False)[["elapsed_s", "peak_rss_fit_gb"]]
        .median()
        .sort_values(["model", "n_probes"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric, ylabel in [
        (axes[0], "elapsed_s", "Elapsed Time (s)"),
        (axes[1], "peak_rss_fit_gb", "Peak RSS During Fit (GB)"),
    ]:
        for model, group in grouped.groupby("model"):
            ax.plot(
                group["n_probes"],
                group[metric],
                marker="o",
                linewidth=2,
                label=f"{model}, safety_checks=True",
            )
        ax.set_xlabel("n_probes")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_title("AI_stochastic n_probes sensitivity, safety_checks=True only", fontsize=11)
    axes[0].legend(frameon=False)
    path = FIGURE_DIR / "n_probes_tradeoff.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_accelerator_figure(df: pd.DataFrame) -> Path:
    subset = df[
        (df["status"] == "ok")
        & (df["requested_method"] == "AI_stochastic")
        & (df["phase"] == "accelerator_compare")
    ].copy()
    if subset.empty:
        return FIGURE_DIR / "accelerator_compare.png"

    grouped = (
        subset.groupby(["N", "model", "use_accelerator"], as_index=False)[
            ["elapsed_s", "peak_rss_fit_gb"]
        ]
        .median()
        .sort_values(["N", "model", "use_accelerator"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    width = 0.35
    group_keys = list(grouped.groupby(["N", "model"]).groups.keys())
    xlabels = [f"N={N}\n{model}" for N, model in group_keys]
    positions = np.arange(len(group_keys))
    for panel, metric in enumerate(["elapsed_s", "peak_rss_fit_gb"]):
        ax = axes[panel]
        for i, use_accel in enumerate([True, False]):
            values = []
            for (N, model), group in grouped.groupby(["N", "model"]):
                row = group[group["use_accelerator"] == use_accel]
                values.append(float(row.iloc[0][metric]) if not row.empty else np.nan)
            label = "compiled accelerator" if use_accel else "pure Python fallback"
            ax.bar(positions + (i - 0.5) * width, values, width=width, label=label)
        ax.set_xticks(positions)
        ax.set_xticklabels(xlabels)
        ax.set_ylabel("Elapsed Time (s)" if metric == "elapsed_s" else "Peak RSS During Fit (GB)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)
    path = FIGURE_DIR / "accelerator_compare.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_extrapolation_figure(df: pd.DataFrame) -> Path:
    subset = df[
        (df["status"] == "ok")
        & (df["requested_method"] == "AI_stochastic")
        & (df["dtype"] == "float32")
        & (df["n_probes"] == 50)
        & (df["use_accelerator"] == True)
    ].copy()
    if subset.empty:
        return FIGURE_DIR / "scaling_extrapolation.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    target_n = 50000.0
    color_map = {"GREML": "#1f77b4", "RDR-SNP": "#d62728"}
    linestyle_map = {True: "-", False: "--"}
    for (model, safety_checks), group in subset.groupby(["model", "safety_checks"]):
        group = (
            group.groupby("N", as_index=False)[["elapsed_s", "peak_rss_fit_gb"]]
            .median()
            .sort_values("N")
        )
        if len(group) < 2:
            continue
        Ns = group["N"].to_numpy(dtype=float)
        elapsed = group["elapsed_s"].to_numpy(dtype=float)
        peak_rss = group["peak_rss_fit_gb"].to_numpy(dtype=float)

        scale, exponent = fit_power_law(Ns, elapsed)
        intercept, slope = fit_quadratic_memory(Ns, peak_rss)
        N_grid = np.linspace(Ns.min(), max(Ns.max() * 1.1, target_n), 120)
        color = color_map[model]
        linestyle = linestyle_map[bool(safety_checks)]
        label = f"{model}, safety_checks={bool(safety_checks)}"

        axes[0].scatter(Ns, elapsed, color=color, alpha=0.7, s=24)
        axes[0].plot(
            N_grid,
            predict_time(scale, exponent, N_grid),
            linewidth=2,
            color=color,
            linestyle=linestyle,
            label=label,
        )
        axes[0].scatter(
            [target_n],
            [float(predict_time(scale, exponent, target_n))],
            color=color,
            marker="x",
            s=55,
        )

        axes[1].scatter(Ns, peak_rss, color=color, alpha=0.7, s=24)
        axes[1].plot(
            N_grid,
            predict_peak_rss(intercept, slope, N_grid),
            linewidth=2,
            color=color,
            linestyle=linestyle,
            label=label,
        )
        axes[1].scatter(
            [target_n],
            [float(predict_peak_rss(intercept, slope, target_n))],
            color=color,
            marker="x",
            s=55,
        )

    axes[0].set_xlabel("N")
    axes[0].set_ylabel("Elapsed Time (s)")
    axes[0].grid(alpha=0.3)
    axes[0].axvline(target_n, linestyle=":", color="black", linewidth=1.5)
    axes[0].set_title("AI_stochastic extrapolation to N=50,000, safety_checks=True/False", fontsize=11)
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("Peak RSS During Fit (GB)")
    axes[1].grid(alpha=0.3)
    axes[1].axvline(target_n, linestyle=":", color="black", linewidth=1.5)
    axes[1].set_title("Peak RSS extrapolation to N=50,000, safety_checks=True/False", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    path = FIGURE_DIR / "scaling_extrapolation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_summary_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ok = df[df["status"] == "ok"].copy()
    core = (
        ok.groupby(
            [
                "suite",
                "phase",
                "model",
                "requested_method",
                "safety_checks",
                "dtype",
                "n_probes",
                "use_accelerator",
                "N",
            ],
            as_index=False,
        )[
            ["elapsed_s", "peak_rss_fit_gb", "peak_incremental_rss_fit_gb", "iterations"]
        ]
        .median()
        .sort_values(["suite", "phase", "model", "requested_method", "N"])
    )
    methods = (
        ok.groupby(["model", "requested_method", "safety_checks"], as_index=False)[
            ["elapsed_s", "peak_rss_fit_gb", "iterations"]
        ]
        .agg(["median", "max"])
        .reset_index()
    )
    methods.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in methods.columns
    ]

    extrap_rows = []
    subset = ok[
        (ok["requested_method"] == "AI_stochastic")
        & (ok["dtype"] == "float32")
        & (ok["n_probes"] == 50)
        & (ok["use_accelerator"] == True)
    ].copy()
    target_n = 50000.0
    for (model, safety_checks), group in subset.groupby(["model", "safety_checks"]):
        by_n = (
            group.groupby("N", as_index=False)[["elapsed_s", "peak_rss_fit_gb"]]
            .median()
            .sort_values("N")
        )
        if len(by_n) < 2:
            continue
        Ns = by_n["N"].to_numpy(dtype=float)
        elapsed = by_n["elapsed_s"].to_numpy(dtype=float)
        peak_rss = by_n["peak_rss_fit_gb"].to_numpy(dtype=float)
        scale, exponent = fit_power_law(Ns, elapsed)
        intercept, slope = fit_quadratic_memory(Ns, peak_rss)
        extrap_rows.append(
            {
                "model": model,
                "safety_checks": bool(safety_checks),
                "time_scale": scale,
                "time_exponent": exponent,
                "rss_intercept_gb": intercept,
                "rss_per_n2_gb": slope,
                "predicted_time_at_50000_s": float(predict_time(scale, exponent, target_n)),
                "predicted_peak_rss_at_50000_gb": float(predict_peak_rss(intercept, slope, target_n)),
            }
        )
    extrap = pd.DataFrame(extrap_rows)
    return core, methods, extrap


def render_markdown_report(
    df: pd.DataFrame,
    core_table: pd.DataFrame,
    method_table: pd.DataFrame,
    extrap_table: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    ok = df[df["status"] == "ok"].copy()
    total_runs = len(df)
    ok_runs = len(ok)
    failures = len(df[df["status"] != "ok"])

    lines: list[str] = []
    lines.append("# Benchmarking `run_REML()`")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append(
        "Benchmarks used the simulation recipe from "
        "`IRI/scripts/simulations/sandbox/sandbox_4_benchmark.ipynb`: "
        "`M=2000`, `N` varied, one simulated generation with related offspring, "
        "trait variance components `V_A=0.4`, `V_A_par=0.1`, `V_Eps=0.5`, and "
        "`M_causal=500`."
    )
    lines.append(
        "For each `N`, the population, phenotype, and GRMs were simulated once and "
        "cached. Each benchmark then loaded the cached arrays and measured only the "
        "`run_REML()` call in a fresh Python subprocess. Peak RSS was sampled during "
        "the fit after the arrays were already resident in memory, so `peak_rss_fit_gb` "
        "captures the actual process footprint during REML rather than the simulation step."
    )
    lines.append(
        "The main benchmark factors were population size (`N`), number of fitted "
        "relationship matrices (1-matrix GREML versus 3-matrix RDR-SNP), REML method "
        "(`AI_stochastic`, `AI`, `FS`), and `safety_checks` (`True` versus `False`). "
        "Additional sensitivity studies varied `n_probes`, `dtype`, and the compiled "
        "stochastic accelerator for `AI_stochastic`."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        f"A total of {total_runs} benchmark rows were recorded, of which {ok_runs} completed "
        f"successfully and {failures} ended in error or were skipped after a smaller run "
        "in the same family hit a practical resource limit."
    )
    lines.append("")
    if not core_table.empty:
        lines.append("### Core Summary")
        lines.append("")
        lines.append(dataframe_to_markdown(core_table.head(20)))
        lines.append("")
    if not extrap_table.empty:
        lines.append("### Extrapolated N=50,000 Performance")
        lines.append("")
        lines.append(dataframe_to_markdown(extrap_table, floatfmt=".3f"))
        lines.append("")
    for path in figure_paths:
        rel = path.relative_to(BENCHMARK_ROOT)
        lines.append(f"![{path.stem}]({rel.as_posix()})")
        lines.append("")
    lines.append("## Discussion")
    lines.append("")
    lines.append(
        "The dominant cost drivers should be interpreted in the context of the current "
        "implementation: `run_REML()` materializes dense `N x N` covariance matrices and "
        "the exact methods explicitly form inverses and projection matrices. That makes "
        "the number of fitted relationship matrices a direct memory multiplier and makes "
        "exact methods much less scalable than `AI_stochastic`."
    )
    if not extrap_table.empty:
        lines.append(
            "Extrapolation from the successful `AI_stochastic` runs suggests the following "
            "approximate process footprints at `N=50,000`:"
        )
        for _, row in extrap_table.sort_values(["model", "safety_checks"]).iterrows():
            lines.append(
                f"- {row['model']}, safety_checks={row['safety_checks']}: "
                f"{row['predicted_time_at_50000_s'] / 3600.0:.2f} hours and "
                f"{row['predicted_peak_rss_at_50000_gb']:.1f} GB peak RSS."
            )
    lines.append(
        "The fitted extrapolations are empirical rather than guaranteed. They are most "
        "trustworthy near the observed `N` range and less trustworthy once fallback behavior, "
        "non-convergence, or OS-level memory pressure starts to matter."
    )
    lines.append(
        "If these benchmarks will be used for Slurm resource requests, a conservative rule is "
        "to request somewhat more than the observed peak RSS because the current measurement "
        "excludes the one-time simulation step but still depends on BLAS/LAPACK workspace and "
        "allocator behavior on the target node."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ensure_directories()
    if not RESULTS_CSV.exists():
        raise SystemExit(f"Benchmark CSV not found: {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV)
    core_table, method_table, extrap_table = build_summary_tables(df)
    save_table(core_table, "core_summary.csv")
    save_table(method_table, "method_summary.csv")
    save_table(extrap_table, "extrapolation_summary.csv")

    figure_paths = [
        make_core_scaling_figure(df),
        make_method_comparison_figure(df),
        make_n_probes_figure(df),
        make_accelerator_figure(df),
        make_extrapolation_figure(df),
    ]

    report_text = render_markdown_report(df, core_table, method_table, extrap_table, figure_paths)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()

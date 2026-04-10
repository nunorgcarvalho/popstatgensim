from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

BENCHMARK_ROOT = Path(__file__).resolve().parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from benchmark_reml_utils import (
    LOG_DIR,
    RESULTS_CSV,
    REPO_ROOT,
    build_benchmark_plan,
    ensure_directories,
    json_dumps_sorted,
    simulate_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["pilot", "main", "all"],
        default="pilot",
        help="Which predefined benchmark suite to run.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="/n/groups/price/nuno/.venv_py13/bin/python",
        help="Python executable to use for subprocess benchmark runs.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=RESULTS_CSV,
        help="Path to the cumulative benchmark CSV.",
    )
    parser.add_argument(
        "--force-datasets",
        action="store_true",
        help="Rebuild cached datasets even if they already exist.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Run configurations even if a result row with the same run_id already exists.",
    )
    parser.add_argument(
        "--skip-after-seconds",
        type=float,
        default=1800.0,
        help="Skip larger N in the same plan family after a successful run exceeds this runtime.",
    )
    parser.add_argument(
        "--skip-after-peak-rss-gb",
        type=float,
        default=58.0,
        help="Skip larger N in the same plan family after a successful run exceeds this peak RSS.",
    )
    return parser.parse_args()


def load_existing_results(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_results(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["suite", "phase", "model", "requested_method", "N"]).to_csv(
        path, index=False
    )


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    src_dir = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_dir if not env.get("PYTHONPATH") else f"{src_dir}:{env['PYTHONPATH']}"
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", "/tmp/matplotlib-benchmark-reml")
    return env


def run_one_benchmark(run_config: dict, dataset_path: Path, python_bin: str) -> dict:
    worker_path = BENCHMARK_ROOT / "worker_run_reml.py"
    cmd = [
        python_bin,
        str(worker_path),
        "--dataset",
        str(dataset_path),
        "--run-config-json",
        json_dumps_sorted(run_config),
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=child_env(),
        cwd=str(REPO_ROOT),
    )

    stdout_path = LOG_DIR / f"{run_config['run_id']}.stdout.log"
    stderr_path = LOG_DIR / f"{run_config['run_id']}.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    row = dict(run_config)
    row["subprocess_returncode"] = int(completed.returncode)
    row["stdout_log"] = str(stdout_path.relative_to(BENCHMARK_ROOT))
    row["stderr_log"] = str(stderr_path.relative_to(BENCHMARK_ROOT))

    if completed.returncode != 0:
        row.update(
            {
                "status": "subprocess_error",
                "failure_type": "subprocess_error",
                "failure_message": completed.stderr.strip()[:5000],
                "elapsed_s": float("nan"),
                "cpu_s": float("nan"),
                "rss_after_load_gb": float("nan"),
                "rss_after_fit_gb": float("nan"),
                "peak_rss_fit_gb": float("nan"),
                "peak_incremental_rss_fit_gb": float("nan"),
                "warnings_count": 0,
                "warnings_text": "",
                "fit_method": "",
                "fit_requested_method_echo": "",
                "used_fallback": False,
                "iterations": -1,
                "converged": False,
            }
        )
        return row

    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        row.update(
            {
                "status": "parse_error",
                "failure_type": "parse_error",
                "failure_message": "Worker produced no JSON output.",
            }
        )
        return row

    parsed = json.loads(stdout_lines[-1])
    row.update(parsed)
    return row


def should_skip_due_to_family_limit(
    run_config: dict,
    family_limits: dict[str, tuple[int, str]],
) -> bool:
    family = run_config["plan_family"]
    if family not in family_limits:
        return False
    failing_n, _ = family_limits[family]
    return int(run_config["N"]) > int(failing_n)


def main() -> None:
    args = parse_args()
    ensure_directories()

    plan = build_benchmark_plan(args.suite)
    existing = load_existing_results(args.results_csv)
    existing_run_ids = set()
    if not existing.empty and "run_id" in existing.columns and not args.rerun_existing:
        existing_run_ids = set(existing["run_id"].astype(str))

    family_limits: dict[str, tuple[int, str]] = {}
    if not existing.empty and "plan_family" in existing.columns:
        for _, row in existing.iterrows():
            if row.get("status") == "ok":
                if (
                    float(row.get("elapsed_s", 0.0)) > args.skip_after_seconds
                    or float(row.get("peak_rss_fit_gb", 0.0)) > args.skip_after_peak_rss_gb
                ):
                    family_limits[str(row["plan_family"])] = (
                        int(row["N"]),
                        "existing_limit",
                    )
            elif row.get("status") != "ok":
                family_limits[str(row["plan_family"])] = (int(row["N"]), "existing_failure")

    rows_out = existing.to_dict(orient="records") if not existing.empty else []

    for run_config in plan:
        if run_config["run_id"] in existing_run_ids:
            print(f"[skip existing] {run_config['run_id']}")
            continue

        if should_skip_due_to_family_limit(run_config, family_limits):
            reason = family_limits[run_config["plan_family"]][1]
            skipped = dict(run_config)
            skipped.update(
                {
                    "status": "skipped_family_limit",
                    "failure_type": reason,
                    "failure_message": (
                        f"Skipped because a smaller N in the same family hit {reason}."
                    ),
                    "subprocess_returncode": 0,
                }
            )
            rows_out.append(skipped)
            print(f"[skip family] {run_config['run_id']}")
            save_results(args.results_csv, pd.DataFrame(rows_out))
            continue

        dataset_path, dataset_meta_path = simulate_dataset(
            run_config["N"],
            run_config["rep"],
            force=args.force_datasets,
        )
        print(
            f"[run] {run_config['run_id']} "
            f"(dataset={dataset_path.name}, meta={dataset_meta_path.name})"
        )
        row = run_one_benchmark(run_config, dataset_path, args.python_bin)
        row["dataset_path"] = str(dataset_path.relative_to(BENCHMARK_ROOT))
        row["dataset_meta_path"] = str(dataset_meta_path.relative_to(BENCHMARK_ROOT))
        rows_out.append(row)
        save_results(args.results_csv, pd.DataFrame(rows_out))

        if row.get("status") != "ok":
            family_limits[run_config["plan_family"]] = (
                int(run_config["N"]),
                str(row.get("failure_type", "failure")),
            )
            continue

        if (
            float(row.get("elapsed_s", 0.0)) > args.skip_after_seconds
            or float(row.get("peak_rss_fit_gb", 0.0)) > args.skip_after_peak_rss_gb
        ):
            family_limits[run_config["plan_family"]] = (
                int(run_config["N"]),
                "resource_limit",
            )

    print(f"Wrote benchmark results to {args.results_csv}")


if __name__ == "__main__":
    main()

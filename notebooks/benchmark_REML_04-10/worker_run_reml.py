from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import psutil

BENCHMARK_ROOT = Path(__file__).resolve().parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from benchmark_reml_utils import dtype_from_name, format_gb

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import popstatgensim as psgs
from popstatgensim.estimation import reml as reml_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-config-json", type=str, required=True)
    return parser.parse_args()


def _sample_peak_rss(stop_event: threading.Event, container: dict, interval_s: float = 0.01) -> None:
    process = psutil.Process(os.getpid())
    peak = process.memory_info().rss
    while not stop_event.is_set():
        peak = max(peak, process.memory_info().rss)
        time.sleep(interval_s)
    peak = max(peak, process.memory_info().rss)
    container["peak_rss_bytes"] = int(peak)


def main() -> None:
    args = parse_args()
    run_config = json.loads(args.run_config_json)

    with np.load(args.dataset, allow_pickle=False) as data:
        y = np.asarray(data["y"])
        R_oo = np.asarray(data["R_oo"])
        if run_config["model"] == "GREML":
            Rs = [R_oo]
            loaded_bytes = y.nbytes + R_oo.nbytes
        else:
            R_pp = np.asarray(data["R_pp"])
            R_op = np.asarray(data["R_op"])
            Rs = [R_oo, R_pp, R_op]
            loaded_bytes = y.nbytes + R_oo.nbytes + R_pp.nbytes + R_op.nbytes
    dtype = dtype_from_name(run_config["dtype"])
    accelerator_available = reml_module._stochastic_ops_accel is not None
    if run_config["requested_method"] == "AI_stochastic" and not bool(run_config["use_accelerator"]):
        reml_module._stochastic_ops_accel = None

    process = psutil.Process(os.getpid())
    rss_after_load_bytes = int(process.memory_info().rss)

    peak_container = {"peak_rss_bytes": rss_after_load_bytes}
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_sample_peak_rss,
        args=(stop_event, peak_container),
        daemon=True,
    )

    result = {
        "status": "ok",
        "failure_type": "",
        "failure_message": "",
    }

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        monitor_thread.start()
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        try:
            out = psgs.run_REML(
                y=y,
                Rs=Rs,
                method=run_config["requested_method"],
                constrain=bool(run_config["constrain"]),
                safety_checks=bool(run_config["safety_checks"]),
                tol=float(run_config["tol"]),
                max_iter=int(run_config["max_iter"]),
                std_y=bool(run_config["std_y"]),
                verbose=0,
                n_probes=int(run_config["n_probes"]),
                seed=int(run_config["seed_solver"]),
                dtype=dtype,
            )
        except Exception as exc:  # noqa: BLE001
            out = None
            result["status"] = "error"
            result["failure_type"] = exc.__class__.__name__
            result["failure_message"] = str(exc)
        finally:
            elapsed_s = time.perf_counter() - t0
            cpu_s = time.process_time() - cpu0
            stop_event.set()
            monitor_thread.join(timeout=2.0)

    rss_after_fit_bytes = int(process.memory_info().rss)
    peak_rss_bytes = int(max(peak_container["peak_rss_bytes"], rss_after_fit_bytes))

    result.update(
        {
            "elapsed_s": elapsed_s,
            "cpu_s": cpu_s,
            "rss_after_load_gb": format_gb(rss_after_load_bytes),
            "rss_after_fit_gb": format_gb(rss_after_fit_bytes),
            "peak_rss_fit_gb": format_gb(peak_rss_bytes),
            "peak_incremental_rss_fit_gb": format_gb(peak_rss_bytes - rss_after_load_bytes),
            "warnings_count": len(caught_warnings),
            "warnings_text": " | ".join(str(w.message) for w in caught_warnings[:5]),
            "arrays_total_loaded_gb": format_gb(loaded_bytes),
            "dataset_y_dtype": str(y.dtype),
            "dataset_matrix_dtype": str(R_oo.dtype),
            "accelerator_requested": bool(run_config["use_accelerator"]),
            "accelerator_available_at_import": bool(accelerator_available),
            "accelerator_active_in_run": bool(reml_module._stochastic_ops_accel is not None),
        }
    )

    if out is not None:
        algorithm = out["algorithm"]
        result.update(
            {
                "fit_method": algorithm.get("method", ""),
                "fit_requested_method_echo": algorithm.get("requested_method", ""),
                "used_fallback": bool(
                    algorithm.get("requested_method", run_config["requested_method"])
                    != algorithm.get("method", run_config["requested_method"])
                ),
                "iterations": int(algorithm.get("iterations", -1)),
                "converged": bool(algorithm.get("converged", False)),
                "log_likelihood": float(out.get("log_likelihood", np.nan)),
                "n_var_components_reported": int(len(out["var_comps"]["est"])),
                "var_y_before_FE": float(out["var_y"]["before_FE"]),
                "var_y_after_FE": float(out["var_y"]["after_FE"]),
                "sum_components": float(out["var_y"]["sum_comp"]),
            }
        )
    else:
        result.update(
            {
                "fit_method": "",
                "fit_requested_method_echo": "",
                "used_fallback": False,
                "iterations": -1,
                "converged": False,
                "log_likelihood": np.nan,
                "n_var_components_reported": 0,
                "var_y_before_FE": np.nan,
                "var_y_after_FE": np.nan,
                "sum_components": np.nan,
            }
        )

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

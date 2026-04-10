from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import popstatgensim as psgs

BENCHMARK_ROOT = Path(__file__).resolve().parent
CACHE_DIR = BENCHMARK_ROOT / "cache"
RESULTS_DIR = BENCHMARK_ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
REPORT_PATH = BENCHMARK_ROOT / "report.md"
RESULTS_CSV = RESULTS_DIR / "benchmark_runs.csv"

NOTEBOOK_SIM_DEFAULTS = {
    "M": 2000,
    "R_type": "indep",
    "keep_past_generations": 1,
    "generations": 1,
    "related_offspring": True,
    "trait_name": "y1",
    "V_A": 0.4,
    "V_A_par": 0.1,
    "r_DIGEs": 1.0,
    "M_causal": 500,
}
NOTEBOOK_SIM_DEFAULTS["V_Eps"] = (
    1.0 - NOTEBOOK_SIM_DEFAULTS["V_A"] - NOTEBOOK_SIM_DEFAULTS["V_A_par"]
)

FIT_DEFAULTS = {
    "constrain": False,
    "tol": 1e-4,
    "max_iter": 30,
    "std_y": False,
    "seed_solver": 42,
    "n_probes": 50,
    "dtype": "float32",
}

MODEL_TO_COMPONENTS = {
    "GREML": ["R_oo"],
    "RDR-SNP": ["R_oo", "R_pp", "R_op"],
}


def ensure_directories() -> None:
    for path in [CACHE_DIR, RESULTS_DIR, LOG_DIR, FIGURE_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def format_gb(num_bytes: float) -> float:
    return float(num_bytes) / (1024.0 ** 3)


def dataset_stem(n: int, rep: int) -> str:
    return f"dataset_N{n}_rep{rep:02d}"


def dataset_npz_path(n: int, rep: int) -> Path:
    return CACHE_DIR / f"{dataset_stem(n, rep)}.npz"


def dataset_meta_path(n: int, rep: int) -> Path:
    return CACHE_DIR / f"{dataset_stem(n, rep)}.json"


def default_population_seed(n: int, rep: int) -> int:
    return 20260410 + 1000 * rep + 17 * n


def default_trait_seed(n: int, rep: int) -> int:
    return default_population_seed(n, rep) + 1


def simulate_dataset(
    n: int,
    rep: int = 0,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    ensure_directories()
    npz_path = dataset_npz_path(n, rep)
    meta_path = dataset_meta_path(n, rep)
    if npz_path.exists() and meta_path.exists() and not force:
        return npz_path, meta_path

    pop_seed = default_population_seed(n, rep)
    trait_seed = default_trait_seed(n, rep)

    t0 = time.perf_counter()
    np.random.seed(pop_seed)
    pop = psgs.Population(
        N=n,
        M=NOTEBOOK_SIM_DEFAULTS["M"],
        R_type=NOTEBOOK_SIM_DEFAULTS["R_type"],
        keep_past_generations=NOTEBOOK_SIM_DEFAULTS["keep_past_generations"],
    )
    pop.simulate_generations(
        generations=NOTEBOOK_SIM_DEFAULTS["generations"],
        related_offspring=NOTEBOOK_SIM_DEFAULTS["related_offspring"],
    )
    R_oo, R_pp, R_op = pop.get_RDR_SNP_GRMs()

    np.random.seed(trait_seed)
    genetic_effects = psgs.traits.generate_genetic_effects(
        var_A=NOTEBOOK_SIM_DEFAULTS["V_A"],
        var_A_par=NOTEBOOK_SIM_DEFAULTS["V_A_par"],
        r=NOTEBOOK_SIM_DEFAULTS["r_DIGEs"],
        M=NOTEBOOK_SIM_DEFAULTS["M"],
        M_causal=NOTEBOOK_SIM_DEFAULTS["M_causal"],
        force_var=True,
    )
    pop.add_trait(
        NOTEBOOK_SIM_DEFAULTS["trait_name"],
        effects={"A": genetic_effects["A"], "A_par": genetic_effects["A_par"]},
        var_Eps=NOTEBOOK_SIM_DEFAULTS["V_Eps"],
    )
    y = np.asarray(pop.traits[NOTEBOOK_SIM_DEFAULTS["trait_name"]].y)

    np.savez(
        npz_path,
        y=y,
        R_oo=np.asarray(R_oo),
        R_pp=np.asarray(R_pp),
        R_op=np.asarray(R_op),
    )
    elapsed = time.perf_counter() - t0

    metadata = {
        "n": int(n),
        "rep": int(rep),
        "population_seed": int(pop_seed),
        "trait_seed": int(trait_seed),
        "simulation_elapsed_s": elapsed,
        "arrays": {
            "y_shape": list(y.shape),
            "R_oo_shape": list(np.asarray(R_oo).shape),
            "R_pp_shape": list(np.asarray(R_pp).shape),
            "R_op_shape": list(np.asarray(R_op).shape),
            "y_dtype": str(np.asarray(y).dtype),
            "R_oo_dtype": str(np.asarray(R_oo).dtype),
            "R_pp_dtype": str(np.asarray(R_pp).dtype),
            "R_op_dtype": str(np.asarray(R_op).dtype),
            "total_bytes": int(
                np.asarray(y).nbytes
                + np.asarray(R_oo).nbytes
                + np.asarray(R_pp).nbytes
                + np.asarray(R_op).nbytes
            ),
        },
        "notebook_defaults": NOTEBOOK_SIM_DEFAULTS,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return npz_path, meta_path


def load_dataset_metadata(n: int, rep: int) -> dict:
    return json.loads(dataset_meta_path(n, rep).read_text(encoding="utf-8"))


def component_count(model: str) -> int:
    return len(MODEL_TO_COMPONENTS[model])


def model_sort_key(model: str) -> int:
    return {"GREML": 1, "RDR-SNP": 3}.get(model, 99)


def dtype_from_name(name: str) -> np.dtype:
    mapping = {
        "float32": np.float32,
        "float64": np.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype name: {name}")
    return mapping[name]


def run_id_from_config(config: dict) -> str:
    return (
        f"{config['suite']}"
        f"__N{config['N']}"
        f"__rep{config['rep']:02d}"
        f"__{config['model']}"
        f"__{config['requested_method']}"
        f"__safe{int(config['safety_checks'])}"
        f"__np{config['n_probes']}"
        f"__{config['dtype']}"
        f"__accel{int(config['use_accelerator'])}"
    )


def _append_configs(
    rows: list[dict],
    *,
    suite: str,
    phase: str,
    Ns: list[int],
    models: list[str],
    methods: list[str],
    safety_values: list[bool],
    n_probes_values: list[int],
    dtype_values: list[str],
    use_accelerator_values: list[bool],
    rep: int = 0,
) -> None:
    for N in Ns:
        for model in models:
            for requested_method in methods:
                for safety_checks in safety_values:
                    for n_probes in n_probes_values:
                        for dtype_name in dtype_values:
                            for use_accelerator in use_accelerator_values:
                                row = {
                                    "suite": suite,
                                    "phase": phase,
                                    "N": int(N),
                                    "rep": int(rep),
                                    "model": model,
                                    "n_relationship_matrices": component_count(model),
                                    "requested_method": requested_method,
                                    "safety_checks": bool(safety_checks),
                                    "n_probes": int(n_probes),
                                    "dtype": dtype_name,
                                    "use_accelerator": bool(use_accelerator),
                                    "constrain": FIT_DEFAULTS["constrain"],
                                    "tol": FIT_DEFAULTS["tol"],
                                    "max_iter": FIT_DEFAULTS["max_iter"],
                                    "std_y": FIT_DEFAULTS["std_y"],
                                    "seed_solver": FIT_DEFAULTS["seed_solver"],
                                    "M": NOTEBOOK_SIM_DEFAULTS["M"],
                                    "M_causal": NOTEBOOK_SIM_DEFAULTS["M_causal"],
                                    "V_A": NOTEBOOK_SIM_DEFAULTS["V_A"],
                                    "V_A_par": NOTEBOOK_SIM_DEFAULTS["V_A_par"],
                                    "V_Eps": NOTEBOOK_SIM_DEFAULTS["V_Eps"],
                                }
                                row["run_id"] = run_id_from_config(row)
                                rows.append(row)


def build_benchmark_plan(suite: str) -> list[dict]:
    rows: list[dict] = []

    if suite in {"pilot", "all"}:
        _append_configs(
            rows,
            suite="pilot",
            phase="core_scaling",
            Ns=[1000, 3000, 5000],
            models=["GREML", "RDR-SNP"],
            methods=["AI_stochastic"],
            safety_values=[True, False],
            n_probes_values=[50],
            dtype_values=["float32"],
            use_accelerator_values=[True],
        )
        _append_configs(
            rows,
            suite="pilot",
            phase="exact_method_compare",
            Ns=[1000, 3000],
            models=["GREML", "RDR-SNP"],
            methods=["AI", "FS"],
            safety_values=[True, False],
            n_probes_values=[50],
            dtype_values=["float32"],
            use_accelerator_values=[True],
        )
        _append_configs(
            rows,
            suite="pilot",
            phase="n_probes_compare",
            Ns=[3000],
            models=["GREML", "RDR-SNP"],
            methods=["AI_stochastic"],
            safety_values=[True],
            n_probes_values=[10, 25, 50, 100, 200],
            dtype_values=["float32"],
            use_accelerator_values=[True],
        )
        _append_configs(
            rows,
            suite="pilot",
            phase="dtype_compare",
            Ns=[3000],
            models=["GREML", "RDR-SNP"],
            methods=["AI_stochastic"],
            safety_values=[True],
            n_probes_values=[50],
            dtype_values=["float32", "float64"],
            use_accelerator_values=[True],
        )
        _append_configs(
            rows,
            suite="pilot",
            phase="accelerator_compare",
            Ns=[3000, 5000],
            models=["GREML", "RDR-SNP"],
            methods=["AI_stochastic"],
            safety_values=[True],
            n_probes_values=[50],
            dtype_values=["float32"],
            use_accelerator_values=[True, False],
        )

    if suite in {"main", "all"}:
        _append_configs(
            rows,
            suite="main",
            phase="core_scaling",
            Ns=[7000, 10000, 13000, 16000, 20000],
            models=["GREML", "RDR-SNP"],
            methods=["AI_stochastic"],
            safety_values=[True],
            n_probes_values=[50],
            dtype_values=["float32"],
            use_accelerator_values=[True],
        )
        _append_configs(
            rows,
            suite="main",
            phase="safety_scaling",
            Ns=[10000, 16000],
            models=["GREML", "RDR-SNP"],
            methods=["AI_stochastic"],
            safety_values=[False],
            n_probes_values=[50],
            dtype_values=["float32"],
            use_accelerator_values=[True],
        )
        _append_configs(
            rows,
            suite="main",
            phase="n_probes_scaling",
            Ns=[7000],
            models=["GREML", "RDR-SNP"],
            methods=["AI_stochastic"],
            safety_values=[True],
            n_probes_values=[10, 200],
            dtype_values=["float32"],
            use_accelerator_values=[True],
        )
        _append_configs(
            rows,
            suite="main",
            phase="dtype_check",
            Ns=[7000],
            models=["GREML", "RDR-SNP"],
            methods=["AI_stochastic"],
            safety_values=[True],
            n_probes_values=[50],
            dtype_values=["float64"],
            use_accelerator_values=[True],
        )

    exact_methods = {"AI", "FS"}
    for row in rows:
        row["plan_family"] = (
            f"{row['phase']}|{row['model']}|{row['requested_method']}|"
            f"safe={int(row['safety_checks'])}|np={row['n_probes']}|"
            f"dtype={row['dtype']}|accel={int(row['use_accelerator'])}"
        )
        row["phase_order"] = {
            "core_scaling": 1,
            "exact_method_compare": 2,
            "n_probes_compare": 3,
            "dtype_compare": 4,
            "accelerator_compare": 5,
            "safety_scaling": 6,
            "exact_method_scaling": 7,
            "n_probes_scaling": 8,
            "dtype_check": 9,
        }.get(row["phase"], 99)
        row["method_order"] = {"AI_stochastic": 1, "AI": 2, "FS": 3}.get(
            row["requested_method"], 99
        )
        row["expected_cost_rank"] = (
            row["phase_order"],
            model_sort_key(row["model"]),
            row["method_order"],
            row["N"],
            row["n_probes"],
            row["dtype"],
            int(not row["use_accelerator"]),
        )
        if row["requested_method"] in exact_methods:
            row["dtype"] = "float32"
            row["n_probes"] = 50
            row["use_accelerator"] = True
            row["run_id"] = run_id_from_config(row)

    rows.sort(key=lambda r: r["expected_cost_rank"])
    return rows


def json_dumps_sorted(data: dict) -> str:
    return json.dumps(data, sort_keys=True)

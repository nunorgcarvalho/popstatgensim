import numpy as np
import pytest

import popstatgensim as psg
from popstatgensim.estimation import reml as reml_mod
from popstatgensim import run_EO_AM, run_HEreg, run_REML
from popstatgensim.traits import build_design_matrix_from_groups


def test_estimation_entrypoints_work_from_refactored_api():
    x = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
        ],
        dtype=float,
    )
    grm = (x @ x.T) / x.shape[1]
    y = np.array([0.2, 1.0, -0.1, -1.1], dtype=float)

    out_he = run_HEreg(y=y, Rs=grm, verbose=0)
    out_reml = run_REML(y=y, Rs=grm, verbose=0, method="FS")

    assert set(out_he.keys()) == {"var_comps", "var_y", "fixed_effects", "algorithm", "log_likelihood"}
    assert set(out_reml.keys()) == {"var_comps", "var_y", "fixed_effects", "algorithm", "log_likelihood"}
    assert len(out_he["var_comps"]["est"]) == 2
    assert len(out_reml["var_comps"]["est"]) == 2
    assert np.isfinite(out_he["var_comps"]["est"]).all()
    assert np.isfinite(out_reml["var_comps"]["est"]).all()


def test_reml_warmstart_handles_design_matrix_random_effects():
    groups = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    z_groups = build_design_matrix_from_groups(groups)
    grm = np.array(
        [
            [1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.2, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.2, 1.0],
        ],
        dtype=float,
    )
    y = np.array([1.0, 0.8, -0.2, -0.1, 0.4, 0.6], dtype=float)

    out_reml = run_REML(
        y=y,
        Rs=[grm, None],
        Zs=[None, z_groups],
        verbose=0,
    )

    assert set(out_reml.keys()) == {"var_comps", "var_y", "fixed_effects", "algorithm", "log_likelihood"}
    assert len(out_reml["var_comps"]["est"]) == 3
    assert np.isfinite(out_reml["var_comps"]["est"]).all()


def test_run_reml_lazy_safety_restarts_strict_mode_after_unchecked_trigger(monkeypatch):
    y = np.array([0.2, 1.0, -0.1, -1.1], dtype=float)
    grm = np.eye(4, dtype=float)
    calls = []

    def fake_ai_stochastic(**kwargs):
        calls.append(
            (
                kwargs["safety_checks"],
                kwargs["safety_checks_enabled"],
                kwargs["monitor"] is not None,
            )
        )
        if not kwargs["safety_checks"]:
            raise reml_mod._SafetyTrigger("AI_stochastic", "mock instability", iteration=1)
        return {
            "var_comps": {"est": np.array([0.1, 0.9]), "se": np.zeros(2), "vcov": np.eye(2)},
            "var_y": {
                "before_FE": 1.0,
                "after_FE": 1.0,
                "sum_comp": 1.0,
                "sum_comp_se": 0.0,
            },
            "fixed_effects": {"est": None, "se": None, "vcov": None},
            "algorithm": {
                "method": "AI_stochastic",
                "iterations": 4,
                "converged": True,
                "safety_checks": True,
                "strict_safety_checks": True,
            },
            "log_likelihood": -1.0,
        }

    monkeypatch.setattr(reml_mod, "_run_ai_stochastic", fake_ai_stochastic)

    out = run_REML(y=y, Rs=grm, verbose=0, method="AI_stochastic", safety_checks=True)

    assert calls == [(False, True, True), (True, True, False)]
    assert out["algorithm"]["strict_safety_checks"] is True
    assert out["algorithm"]["safety_trigger"] == "mock instability"


def test_run_reml_lazy_safety_falls_back_after_strict_blowup(monkeypatch):
    y = np.array([0.2, 1.0, -0.1, -1.1], dtype=float)
    grm = np.eye(4, dtype=float)
    calls = []

    def fake_ai_stochastic(**kwargs):
        calls.append(("AI_stochastic", kwargs["safety_checks"]))
        if not kwargs["safety_checks"]:
            return {
                "var_comps": {"est": np.array([10.0, 0.1]), "se": np.zeros(2), "vcov": np.eye(2)},
                "var_y": {
                    "before_FE": 1.0,
                    "after_FE": 1.0,
                    "sum_comp": 10.1,
                    "sum_comp_se": 0.0,
                },
                "fixed_effects": {"est": None, "se": None, "vcov": None},
                "algorithm": {
                    "method": "AI_stochastic",
                    "iterations": 30,
                    "converged": False,
                    "safety_checks": True,
                    "strict_safety_checks": False,
                },
                "log_likelihood": -1.0,
            }
        return {
            "var_comps": {"est": np.array([10.0, 0.1]), "se": np.zeros(2), "vcov": np.eye(2)},
            "var_y": {
                "before_FE": 1.0,
                "after_FE": 1.0,
                "sum_comp": 10.1,
                "sum_comp_se": 0.0,
            },
            "fixed_effects": {"est": None, "se": None, "vcov": None},
            "algorithm": {
                "method": "AI_stochastic",
                "iterations": 30,
                "converged": False,
                "safety_checks": True,
                "strict_safety_checks": True,
            },
            "log_likelihood": -1.0,
        }

    def fake_ai_exact(**kwargs):
        calls.append(("AI", kwargs["safety_checks"]))
        return {
            "var_comps": {"est": np.array([0.2, 0.8]), "se": np.zeros(2), "vcov": np.eye(2)},
            "var_y": {
                "before_FE": 1.0,
                "after_FE": 1.0,
                "sum_comp": 1.0,
                "sum_comp_se": 0.0,
            },
            "fixed_effects": {"est": None, "se": None, "vcov": None},
            "algorithm": {
                "method": "AI",
                "iterations": 6,
                "converged": True,
                "safety_checks": True,
                "strict_safety_checks": False,
            },
            "log_likelihood": -1.0,
        }

    monkeypatch.setattr(reml_mod, "_run_ai_stochastic", fake_ai_stochastic)
    monkeypatch.setattr(reml_mod, "_run_ai_exact", fake_ai_exact)

    out = run_REML(y=y, Rs=grm, verbose=0, method="AI_stochastic", safety_checks=True)

    assert calls == [("AI_stochastic", False), ("AI_stochastic", True), ("AI", False)]
    assert out["algorithm"]["method"] == "AI"
    assert out["algorithm"]["requested_method"] == "AI_stochastic"


def test_run_eo_am_matches_manual_even_odd_correlation():
    x = np.array(
        [
            [1.0, 2.0, 0.5, -1.0, 2.0, 1.5],
            [0.0, -1.0, 1.5, 2.0, -0.5, 0.0],
            [2.0, 1.0, -0.5, 1.0, 0.5, -1.0],
            [-1.0, 0.5, 2.0, 0.0, 1.0, 2.5],
        ],
        dtype=float,
    )
    weights = np.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1], dtype=float)
    chrom_idx = [0, 2, 4]

    out = run_EO_AM(X=x, pgs_weights=weights, chrom_idx=chrom_idx)

    odd_pgs = x[:, [0, 1, 4, 5]] @ weights[[0, 1, 4, 5]]
    even_pgs = x[:, [2, 3]] @ weights[[2, 3]]
    expected = np.corrcoef(odd_pgs, even_pgs)[0, 1]
    assert np.isclose(out, expected)


def test_run_eo_am_requires_at_least_two_chromosomes():
    x = np.arange(12, dtype=float).reshape(4, 3)
    weights = np.array([0.1, 0.2, 0.3], dtype=float)

    with pytest.raises(ValueError, match="At least two chromosomes"):
        run_EO_AM(X=x, pgs_weights=weights, chrom_idx=[0])


def test_population_get_chrom_idx_uses_r_half_and_includes_zero():
    pop = psg.Population(N=4, M=6, p_init=0.3, seed=3)
    pop.set_params(R=np.array([0.1, 0.5, 0.2, 0.5, 0.1, 0.1], dtype=float))

    np.testing.assert_array_equal(pop.get_chrom_idx(), np.array([0, 1, 3], dtype=int))

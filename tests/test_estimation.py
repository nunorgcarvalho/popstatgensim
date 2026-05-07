import numpy as np
import pytest

import popstatgensim as psg
from popstatgensim.estimation import (
    GWASresult,
    get_exp_PGS_R2,
    get_PGS_N_for_R2,
    reml as reml_mod,
)
from popstatgensim import run_EO_AM, run_HEreg, run_REML
from popstatgensim.traits import build_design_matrix_from_groups
from popstatgensim.utils import fit_linear_regression


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


def test_fit_linear_regression_reports_core_ols_outputs():
    y = np.array([1.0, 2.5, 1.5, 4.0, 3.5], dtype=float)
    x = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [4.0, 1.0],
        ],
        dtype=float,
    )

    out = fit_linear_regression(y=y, X=x, add_intercept=True)

    design = np.column_stack((np.ones(y.shape[0]), x))
    xtx_inv = np.linalg.pinv(design.T @ design)
    coef = xtx_inv @ (design.T @ y)
    resid = y - design @ coef
    rss = float(resid @ resid)
    residual_var = rss / (y.shape[0] - design.shape[1])
    vcov = residual_var * xtx_inv
    se = np.sqrt(np.diag(vcov))

    np.testing.assert_allclose(out.coef, coef)
    np.testing.assert_allclose(out.coef_se, se)
    np.testing.assert_allclose(out.coef_vcov, vcov)
    assert out.n_samples == 5
    assert out.n_predictors == 2
    assert out.n_parameters == 3
    assert out.dof_resid == 2
    assert np.isclose(out.y_mean, y.mean())
    assert np.isclose(out.y_var, y.var())
    assert np.isclose(out.residual_var, residual_var)
    assert np.isclose(out.rss, rss)


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


def test_run_eo_am_matches_manual_even_odd_regression():
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
    odd_pgs = (odd_pgs - odd_pgs.mean()) / odd_pgs.std()
    even_pgs = (even_pgs - even_pgs.mean()) / even_pgs.std()

    design = np.column_stack([np.ones(x.shape[0]), odd_pgs])
    beta = np.linalg.pinv(design.T @ design) @ (design.T @ even_pgs)
    resid = even_pgs - design @ beta
    sigma2 = float(resid @ resid) / (x.shape[0] - design.shape[1])
    vcov = sigma2 * np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.diag(vcov))

    assert np.isclose(out["theta_est"], beta[1])
    assert np.isclose(out["theta_se"], se[1])
    assert np.isclose(out["intercept_est"], beta[0])
    assert np.isclose(out["intercept_se"], se[0])
    assert out["n_covariates"] == 0
    assert out["covariate_names"] == []
    assert out["covar_est"].shape == (0,)
    assert out["covar_se"].shape == (0,)
    assert out["predictor_side"] == "odd"
    assert out["outcome_side"] == "even"
    assert out["even_against_odd"] is True


def test_run_eo_am_flips_orientation_when_requested():
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

    out = run_EO_AM(
        X=x,
        pgs_weights=weights,
        chrom_idx=chrom_idx,
        even_against_odd=False,
    )

    odd_pgs = x[:, [0, 1, 4, 5]] @ weights[[0, 1, 4, 5]]
    even_pgs = x[:, [2, 3]] @ weights[[2, 3]]
    odd_pgs = (odd_pgs - odd_pgs.mean()) / odd_pgs.std()
    even_pgs = (even_pgs - even_pgs.mean()) / even_pgs.std()
    design = np.column_stack([np.ones(x.shape[0]), even_pgs])
    beta = np.linalg.pinv(design.T @ design) @ (design.T @ odd_pgs)

    assert np.isclose(out["theta_est"], beta[1])
    assert out["predictor_side"] == "even"
    assert out["outcome_side"] == "odd"
    assert out["even_against_odd"] is False


def test_run_eo_am_adjusts_predictor_side_pcs():
    x = np.array(
        [
            [0.0, 1.0, 2.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0, 2.0, 1.0],
            [0.0, 2.0, 1.0, 0.0, 1.0, 2.0],
            [1.0, 1.0, 2.0, 2.0, 0.0, 1.0],
            [2.0, 0.0, 1.0, 1.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    weights = np.array([0.1, 0.3, -0.2, 0.4, 0.2, -0.1], dtype=float)
    chrom_idx = [0, 2, 4]

    out = run_EO_AM(
        X=x,
        pgs_weights=weights,
        chrom_idx=chrom_idx,
        adjust_PCs=2,
        standardized_geno=False,
    )

    assert out["n_covariates"] == 2
    assert out["covariate_names"] == ["PC1", "PC2"]
    assert out["covar_est"].shape == (2,)
    assert out["covar_se"].shape == (2,)
    assert np.isfinite(out["covar_est"]).all()
    assert np.isfinite(out["covar_se"]).all()


def test_run_eo_am_adjusts_standardized_predictor_side_pcs_when_flipped():
    x = np.array(
        [
            [0.0, 1.0, 2.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0, 2.0, 1.0],
            [0.0, 2.0, 1.0, 0.0, 1.0, 2.0],
            [1.0, 1.0, 2.0, 2.0, 0.0, 1.0],
            [2.0, 0.0, 1.0, 1.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    x_std = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
    weights = np.array([0.1, 0.3, -0.2, 0.4, 0.2, -0.1], dtype=float)
    chrom_idx = [0, 2, 4]

    out = run_EO_AM(
        X=x_std,
        pgs_weights=weights,
        chrom_idx=chrom_idx,
        adjust_PCs=1,
        even_against_odd=False,
        standardized_geno=True,
    )

    assert out["n_covariates"] == 1
    assert out["covariate_names"] == ["PC1"]
    assert out["predictor_side"] == "even"
    assert out["outcome_side"] == "odd"


def test_run_eo_am_requires_at_least_two_chromosomes():
    x = np.arange(12, dtype=float).reshape(4, 3)
    weights = np.array([0.1, 0.2, 0.3], dtype=float)

    with pytest.raises(ValueError, match="At least two chromosomes"):
        run_EO_AM(X=x, pgs_weights=weights, chrom_idx=[0])


def test_population_get_chrom_idx_uses_r_half_and_includes_zero():
    pop = psg.Population(N=4, M=6, p_init=0.3, seed=3)
    pop.set_params(R=np.array([0.1, 0.5, 0.2, 0.5, 0.1, 0.1], dtype=float))

    np.testing.assert_array_equal(pop.get_chrom_idx(), np.array([0, 1, 3], dtype=int))


def test_get_exp_pgs_r2_matches_closed_form():
    observed = get_exp_PGS_R2(h2=0.4, N=1000, M=200)
    expected = (0.4 ** 2) / (0.4 + (200 / 1000))
    assert np.isclose(observed, expected)


def test_get_pgs_n_for_r2_inverts_expected_r2_formula():
    h2 = 0.4
    M = 200
    N = 1000
    r2 = get_exp_PGS_R2(h2=h2, N=N, M=M)
    observed = get_PGS_N_for_R2(h2=h2, R2=r2, M=M)
    assert np.isclose(observed, N)


def test_gwasresult_wrapper_get_exp_pgs_r2_uses_stored_n_and_m():
    result = GWASresult(
        trait_name="y",
        N=1000,
        M=200,
        n_covariates=0,
        standardize_y=True,
        standardize_geno=True,
        detailed_output=False,
        within_family=None,
        beta_est=np.zeros(200, dtype=float),
        beta_se=np.ones(200, dtype=float),
    )

    observed = result.get_exp_PGS_R2(h2=0.4)
    expected = get_exp_PGS_R2(h2=0.4, N=1000, M=200)
    assert np.isclose(observed, expected)

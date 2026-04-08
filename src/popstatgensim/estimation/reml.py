"""
Fast REML and HE regression utilities for variance-component estimation.

This module houses the variance-component estimation engine used by the
`popstatgensim.estimation` namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np
from scipy import linalg
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.lapack import get_lapack_funcs

_TINY = 1e-14

try:
    from ._reml_accel import stochastic_ops as _stochastic_ops_accel
except ImportError:
    try:
        from _greml_accel import stochastic_ops as _stochastic_ops_accel
    except ImportError:
        _stochastic_ops_accel = None

_WARNED_NO_ACCEL = False


@dataclass
class _PreparedInputs:
    """Validated and preprocessed model inputs shared across estimators."""
    y: np.ndarray
    X_user: np.ndarray | None
    X_model: np.ndarray | None
    Rs: list[np.ndarray]
    Zs: list[np.ndarray | None]
    Vs: list[np.ndarray]
    phenotype_variance: float
    n_random: int


def _as_optional_matrix_list(
    matrices: Sequence[np.ndarray | None] | np.ndarray | None,
    name: str,
) -> list[np.ndarray | None] | None:
    """Normalize a single matrix or a sequence into a Python list, preserving None entries."""
    if matrices is None:
        return None
    if isinstance(matrices, np.ndarray):
        if matrices.ndim != 2:
            raise ValueError(f"`{name}` must be a 2D array or a list of 2D arrays.")
        return [np.asarray(matrices, dtype=float)]

    matrices_list = list(matrices)
    if len(matrices_list) == 0:
        raise ValueError(f"`{name}` must contain at least one covariance matrix.")
    out = []
    for i, M in enumerate(matrices_list):
        if M is None:
            out.append(None)
            continue
        M = np.asarray(M, dtype=float)
        if M.ndim != 2:
            raise ValueError(f"`{name}[{i}]` must be a 2D array.")
        out.append(M)
    return out


def _resolve_random_effect_inputs(
    Rs: Sequence[np.ndarray | None] | np.ndarray | None = None,
    Zs: Sequence[np.ndarray | None] | np.ndarray | None = None,
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    """Resolve random-effect covariance/design inputs."""
    Rs_list = _as_optional_matrix_list(Rs, "Rs")
    Zs_list = _as_optional_matrix_list(Zs, "Zs")

    if Rs_list is None and Zs_list is None:
        raise ValueError("Must provide at least one of `Rs` or `Zs`.")
    if Rs_list is None:
        Rs_list = [None] * len(Zs_list)
    if Zs_list is None:
        Zs_list = [None] * len(Rs_list)
    if len(Rs_list) != len(Zs_list):
        raise ValueError("`Zs` must have the same length as `Rs`.")
    return Rs_list, Zs_list


def _prepare_inputs(
    y: np.ndarray,
    Rs: Sequence[np.ndarray | None] | np.ndarray | None = None,
    Zs: Sequence[np.ndarray | None] | np.ndarray | None = None,
    X: np.ndarray | None = None,
    std_y: bool = False,
) -> _PreparedInputs:
    """Validate user inputs and convert random-effect inputs into covariance matrices."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("`y` must be a 1D array.")
    if not np.isfinite(y).all():
        raise ValueError("`y` contains non-finite values.")

    n = y.shape[0]
    Rs_list, Zs_list = _resolve_random_effect_inputs(Rs=Rs, Zs=Zs)

    if X is not None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] != n:
            raise ValueError(f"`X` must have shape ({n}, q).")
        if not np.isfinite(X).all():
            raise ValueError("`X` contains non-finite values.")

    y = y - y.mean()
    if std_y:
        y_sd = y.std()
        if y_sd <= 0:
            raise ValueError("`y` must have non-zero variance.")
        y = y / y_sd
    phenotype_variance = float(y.var())
    if phenotype_variance <= 0:
        raise ValueError("`y` must have non-zero variance after preprocessing.")

    # Convert each random effect from (Z, R) form into its N x N covariance.
    Vs = []
    Rs_resolved = []
    for i, (R, Z) in enumerate(zip(Rs_list, Zs_list)):
        if Z is None:
            if R is None:
                raise ValueError(
                    f"`Rs[{i}]` must be provided when `Zs[{i}]` is None."
                )
            if R.shape != (n, n):
                raise ValueError(
                    f"`Rs[{i}]` must have shape ({n}, {n}) when `Zs[{i}]` is None."
                )
            R_i = np.asarray(R, dtype=float)
            V_i = R_i
        else:
            Z = np.asarray(Z, dtype=float)
            if Z.ndim != 2 or Z.shape[0] != n:
                raise ValueError(f"`Zs[{i}]` must have shape ({n}, q_i).")
            if R is None:
                R_i = np.eye(Z.shape[1], dtype=float)
            else:
                R_i = np.asarray(R, dtype=float)
            if R_i.shape != (Z.shape[1], Z.shape[1]):
                raise ValueError(
                    f"`Rs[{i}]` must have shape ({Z.shape[1]}, {Z.shape[1]}) "
                    f"to match `Zs[{i}]`."
                )
            V_i = Z @ R_i @ Z.T
        if not np.isfinite(V_i).all():
            raise ValueError(f"The covariance matrix for component {i + 1} is non-finite.")
        Rs_resolved.append(R_i)
        Zs_list[i] = Z
        Vs.append(0.5 * (V_i + V_i.T))

    # Match greml_stochastic(): when covariates are provided, include an intercept
    # automatically in the model matrix used for fitting.
    X_model = None
    if X is not None:
        X_model = np.column_stack([np.ones(n, dtype=float), X])

    return _PreparedInputs(
        y=y,
        X_user=X,
        X_model=X_model,
        Rs=Rs_resolved,
        Zs=Zs_list,
        Vs=Vs,
        phenotype_variance=phenotype_variance,
        n_random=len(Vs),
    )


def _build_v(Vs: list[np.ndarray], theta: np.ndarray) -> np.ndarray:
    """Assemble the total covariance V = sum_k theta_k V_k + theta_e I."""
    V = theta[-1] * np.eye(Vs[0].shape[0], dtype=float)
    for i, V_i in enumerate(Vs):
        V += theta[i] * V_i
    return V


def _factor_theta(
    Vs: list[np.ndarray],
    theta: np.ndarray,
    *,
    dtype: np.dtype | type = float,
    overwrite_a: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, bool]]:
    """Build and factor a covariance matrix for a candidate theta vector."""
    V = theta[-1] * np.eye(Vs[0].shape[0], dtype=dtype)
    for i, V_i in enumerate(Vs):
        V += dtype(theta[i]) * V_i
    Lc = cho_factor(V, lower=True, overwrite_a=overwrite_a, check_finite=False)
    return V, Lc


def _ensure_factorable_theta(
    theta: np.ndarray,
    Vs: list[np.ndarray],
    *,
    dtype: np.dtype | type = float,
    max_tries: int = 12,
) -> np.ndarray:
    """Increase the residual component until the working covariance is factorable."""
    theta = np.asarray(theta, dtype=float).copy()
    theta[-1] = max(theta[-1], _TINY)
    for _ in range(max_tries):
        try:
            _factor_theta(Vs, theta, dtype=dtype, overwrite_a=False)
            return theta
        except linalg.LinAlgError:
            theta[-1] = max(theta[-1] * 2.0, 1e-6)
    raise linalg.LinAlgError("Unable to construct a positive-definite starting covariance.")


def _compute_beta_and_py(
    Lc: tuple[np.ndarray, bool],
    y: np.ndarray,
    X: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Compute fixed-effect estimates and the projected score vector Py."""
    Vinvy = cho_solve(Lc, y, check_finite=False)
    if X is None:
        return None, None, None, Vinvy
    VinvX = cho_solve(Lc, X, check_finite=False)
    XVX = X.T @ VinvX
    XVX_inv = np.linalg.pinv(XVX)
    beta = XVX_inv @ (X.T @ Vinvy)
    Py = Vinvy - VinvX @ beta
    return beta, VinvX, XVX_inv, Py


def _compute_loglik(
    y: np.ndarray,
    X: np.ndarray | None,
    beta: np.ndarray | None,
    Lc: tuple[np.ndarray, bool],
) -> float:
    """Compute the Gaussian log-likelihood at the final variance-component estimates."""
    if X is None:
        resid = y
    else:
        resid = y - X @ beta
    Vinv_resid = cho_solve(Lc, resid, check_finite=False)
    chol = Lc[0]
    logdetV = 2.0 * np.sum(np.log(np.diag(chol)))
    n = y.shape[0]
    return float(
        -0.5 * n * np.log(2.0 * np.pi)
        - 0.5 * logdetV
        - 0.5 * resid.T @ Vinv_resid
    )


def _print_iter_message(vars_i: np.ndarray, offsets: np.ndarray, iteration: int, verbose: int) -> None:
    """Mirror the older REML iteration printouts for continuity."""
    m = len(vars_i)
    if verbose == 1:
        msg = f"#{iteration + 1}: " + ", ".join(
            [f"Comp {i + 1} = {vars_i[i]:.6f}" for i in range(m - 1)]
        )
        print(msg)
    elif verbose == 2:
        msg = f"#{iteration + 1}: " + ", ".join(
            [
                f"Comp {i + 1} = {vars_i[i]:.6f} "
                f"({'+' if offsets[i] > 0 else ''}{offsets[i]:.6f})"
                for i in range(m)
            ]
        )
        print(msg)
    elif verbose >= 3:
        for i in range(m):
            print(
                f"Iteration {iteration + 1}, Random Effect Component {i + 1}: "
                f"Updated variance = {vars_i[i]:.6f}, Offset = {offsets[i]:.6f}"
            )


def _he_warmstart(
    y: np.ndarray,
    Vs: list[np.ndarray],
    residual_variance: float,
    X: np.ndarray | None = None,
) -> np.ndarray:
    """Generic HE-based warm start used when a fast dense-GRM shortcut is unavailable."""
    theta, _, _, _, _ = _he_regression_core(
        y=y,
        Vs=Vs,
        residual_variance=residual_variance,
        X=X,
        constrain=True,
    )
    theta = np.asarray(theta, dtype=float)
    theta = np.maximum(theta, max(residual_variance, 1.0) * 1e-3)
    total = theta.sum()
    if total <= 0:
        theta = np.full_like(theta, residual_variance / len(theta))
    else:
        theta *= residual_variance / total
    return theta


def _fast_grm_he_warmstart(
    Vs: list[np.ndarray],
    y: np.ndarray,
    phenotype_variance: float,
) -> np.ndarray:
    """Fast HE warm start for the common case of dense, precomputed GRMs."""
    k = len(Vs)
    diag_Vs = [np.diag(V_i) for V_i in Vs]
    Vy = [V_i @ y for V_i in Vs]
    y2 = y ** 2
    M = np.empty((k, k), dtype=float)
    b = np.empty(k, dtype=float)
    for a in range(k):
        b[a] = float(np.dot(y, Vy[a]) - np.dot(diag_Vs[a], y2))
        for bb in range(k):
            M[a, bb] = float(
                np.einsum("ij,ij->", Vs[a], Vs[bb]) - np.dot(diag_Vs[a], diag_Vs[bb])
            )

    h0 = np.linalg.lstsq(M, b, rcond=None)[0]
    h0 = np.clip(h0, 0.02, 0.90)
    theta = np.empty(k + 1, dtype=float)
    theta[:k] = h0
    theta[k] = max(1.0 - h0.sum(), 0.05)
    theta *= phenotype_variance / theta.sum()
    return theta


def _can_use_fast_grm_warmstart(
    Rs: Sequence[np.ndarray],
    Zs: Sequence[np.ndarray | None],
    Vs: list[np.ndarray],
) -> bool:
    """Return True when the user supplied plain N x N covariance matrices."""
    if any(Z is not None for Z in Zs):
        return False
    if len(Rs) != len(Vs):
        return False
    for R, V in zip(Rs, Vs):
        if R.shape != V.shape:
            return False
        if np.asarray(R).shape[0] != np.asarray(R).shape[1]:
            return False
    return True


def _potri_inverse(Lc: tuple[np.ndarray, bool]) -> np.ndarray:
    """Expand a Cholesky factorization into an explicit inverse when needed."""
    potri = get_lapack_funcs("potri", (Lc[0],))
    Vinv_tri, info = potri(Lc[0].copy(order="F"), lower=int(Lc[1]), overwrite_c=1)
    if info != 0:
        raise linalg.LinAlgError(f"potri failed with info={info}")
    Vinv = np.tril(Vinv_tri)
    Vinv += np.tril(Vinv, k=-1).T
    return Vinv


def _get_projection_matrix(Vinv: np.ndarray, X: np.ndarray | None) -> np.ndarray:
    """Construct the REML projection matrix P."""
    if X is None:
        return Vinv
    XVX_inv = np.linalg.pinv(X.T @ Vinv @ X)
    return Vinv - Vinv @ X @ XVX_inv @ X.T @ Vinv


def _solve_update_direction(curvature: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Solve the local REML update system with a small diagonal stabilizer."""
    try:
        return np.linalg.solve(curvature + 1e-10 * np.eye(len(score)), score)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(curvature) @ score


def _legacy_newton_step(
    curvature: np.ndarray,
    score: np.ndarray,
    theta: np.ndarray,
    constrain: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Original Newton-style update used before REML safety checks were added."""
    try:
        delta = np.linalg.solve(curvature + 1e-10 * np.eye(len(theta)), score)
    except np.linalg.LinAlgError:
        return theta, np.zeros_like(theta)
    if not constrain:
        return theta + delta, delta
    alpha = 1.0
    for _ in range(25):
        candidate = theta + alpha * delta
        if np.all(candidate > _TINY):
            return np.maximum(candidate, _TINY), alpha * delta
        alpha *= 0.5
    candidate = np.maximum(theta, _TINY)
    return candidate, np.zeros_like(theta)


def _line_search_update(
    theta: np.ndarray,
    delta: np.ndarray,
    Vs: list[np.ndarray],
    *,
    constrain: bool,
    dtype: np.dtype | type = float,
    max_halvings: int = 25,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Shrink an update until the candidate covariance remains factorable."""
    if np.max(np.abs(delta)) == 0:
        return theta.copy(), np.zeros_like(theta), True

    alpha = 1.0
    for _ in range(max_halvings):
        candidate = theta + alpha * delta
        if constrain and np.any(candidate <= _TINY):
            alpha *= 0.5
            continue
        try:
            _factor_theta(Vs, candidate, dtype=dtype, overwrite_a=False)
            if constrain:
                candidate = np.maximum(candidate, _TINY)
            return candidate, alpha * delta, True
        except linalg.LinAlgError:
            alpha *= 0.5
    if constrain:
        candidate = np.maximum(theta, _TINY)
    else:
        candidate = theta.copy()
    return candidate, np.zeros_like(theta), False


def _stochastic_component_ops(
    Vs_work: list[np.ndarray],
    Py: np.ndarray,
    VinvZ: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute KPy and Hutchinson trace terms, using the C accelerator when available."""
    if _stochastic_ops_accel is not None:
        try:
            KPy, traces, _ = _stochastic_ops_accel(
                Vs_work,
                np.ascontiguousarray(Py),
                np.ascontiguousarray(VinvZ),
                np.ascontiguousarray(Z),
                None,
            )
            return KPy, traces
        except (TypeError, ValueError):
            pass

    n_comp = len(Vs_work)
    n = Py.shape[0]
    s = Z.shape[1]
    KPy = np.empty((n, n_comp), dtype=Py.dtype)
    traces = np.empty(n_comp, dtype=np.float64)
    for k, V_k in enumerate(Vs_work):
        KPy[:, k] = V_k @ Py
        traces[k] = float((Z * (V_k @ VinvZ)).sum(dtype=np.float64)) / s
    return KPy, traces


def _warn_if_no_accel() -> None:
    """Warn once per session when stochastic REML is using the slower fallback path."""
    global _WARNED_NO_ACCEL
    if _stochastic_ops_accel is None and not _WARNED_NO_ACCEL:
        warnings.warn(
            "Falling back to the pure-Python stochastic REML path because "
            "`_reml_accel` is unavailable. This can be much slower than the "
            "compiled accelerator.",
            RuntimeWarning,
            stacklevel=3,
        )
        _WARNED_NO_ACCEL = True


def _compute_var_y_after_fe(
    y: np.ndarray,
    X_model: np.ndarray | None,
    beta: np.ndarray | None,
    beta_vcov: np.ndarray | None,
) -> tuple[float, float]:
    """Compute residualized phenotype variance and its delta-method SE from beta."""
    if X_model is None or beta is None:
        return float(y.var()), 0.0

    n = y.shape[0]
    resid = y - X_model @ beta
    M = np.eye(n) - np.ones((n, n), dtype=float) / n
    var_after = float((resid.T @ M @ resid) / n)

    if beta_vcov is None:
        return var_after, np.nan

    grad = -(2.0 / n) * (X_model.T @ (M @ resid))
    var_after_se = float(np.sqrt(max(grad @ beta_vcov @ grad, 0.0)))
    return var_after, var_after_se


def _compute_fe_ols_summary(
    y: np.ndarray,
    X_model: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, float]:
    """Compute OLS fixed-effect estimates and the residual variance after FE removal."""
    if X_model is None:
        return None, None, None, float(y.var())

    beta_full = np.linalg.pinv(X_model) @ y
    residuals = y - X_model @ beta_full
    sigma2 = float(residuals @ residuals) / max(X_model.shape[0] - X_model.shape[1], 1)
    beta_vcov_full = sigma2 * np.linalg.pinv(X_model.T @ X_model)
    beta_se_full = np.sqrt(np.maximum(np.diag(beta_vcov_full), 0.0))
    return beta_full, beta_se_full, beta_vcov_full, float(residuals.var())


def _finalise_output(
    y: np.ndarray,
    X_user: np.ndarray | None,
    X_model: np.ndarray | None,
    Vs: list[np.ndarray],
    theta: np.ndarray,
    vcov: np.ndarray,
    phenotype_variance: float,
    beta_model: np.ndarray | None,
    log_likelihood: float | None,
    iterations: int,
    converged: bool,
    method: str,
    safety_checks: bool,
) -> dict:
    """Package common outputs and derived summary quantities into the public result dict."""
    theta = np.asarray(theta, dtype=float)
    vcov = np.asarray(vcov, dtype=float)
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    var_y_sum_comp = float(theta.sum())
    var_y_sum_comp_se = float(np.sqrt(max(np.ones(len(theta)) @ vcov @ np.ones(len(theta)), 0.0)))

    if X_model is None:
        fixed_effects = None
        fixed_effects_se = None
        fixed_effects_vcov = None
        beta_vcov_for_var = None
    else:
        V = _build_v(Vs, theta)
        try:
            Lc = cho_factor(V, lower=True, check_finite=False)
            VinvX = cho_solve(Lc, X_model, check_finite=False)
        except linalg.LinAlgError:
            VinvX = np.linalg.pinv(V) @ X_model
        beta_vcov_full = np.linalg.pinv(X_model.T @ VinvX)
        fixed_effects = beta_model
        fixed_effects_vcov = beta_vcov_full
        fixed_effects_se = np.sqrt(np.maximum(np.diag(fixed_effects_vcov), 0.0))
        beta_vcov_for_var = beta_vcov_full

    var_y_after_fe, _ = _compute_var_y_after_fe(
        y=y,
        X_model=X_model,
        beta=beta_model,
        beta_vcov=beta_vcov_for_var,
    )

    return {
        "var_comps": {
            "est": theta,
            "se": se,
            "vcov": vcov,
        },
        "var_y": {
            "before_FE": phenotype_variance,
            "after_FE": var_y_after_fe,
            "sum_comp": var_y_sum_comp,
            "sum_comp_se": var_y_sum_comp_se,
        },
        "fixed_effects": {
            "est": fixed_effects,
            "se": fixed_effects_se,
            "vcov": fixed_effects_vcov,
        },
        "algorithm": {
            "method": method,
            "iterations": iterations,
            "converged": converged,
            "safety_checks": safety_checks,
        },
        "log_likelihood": log_likelihood,
    }


def _run_ai_stochastic(
    y: np.ndarray,
    Vs: list[np.ndarray],
    X_user: np.ndarray | None,
    X_model: np.ndarray | None,
    init: np.ndarray,
    tol: float,
    max_iter: int,
    constrain: bool,
    verbose: int,
    seed: int,
    n_probes: int,
    dtype: np.dtype,
    phenotype_variance: float,
    safety_checks: bool,
) -> dict:
    """Run AI-REML using Hutchinson trace estimation."""
    n = y.shape[0]
    m = len(Vs)
    rng = np.random.default_rng(seed)
    theta = init.astype(np.float64, copy=True)
    if safety_checks:
        theta = _ensure_factorable_theta(theta, Vs, dtype=float)
    Vs_work = [V_i.astype(dtype, copy=False) for V_i in Vs]
    y_work = y.astype(dtype, copy=False)
    X_work = None if X_model is None else X_model.astype(dtype, copy=False)

    _warn_if_no_accel()

    # Build antithetic Rademacher probes once and reuse them across iterations.
    probe_blocks = []
    n_pairs = n_probes // 2
    if n_pairs:
        Zh = (rng.integers(0, 2, size=(n, n_pairs)) * 2 - 1).astype(dtype)
        probe_blocks.extend([Zh, -Zh])
    if n_probes % 2:
        probe_blocks.append((rng.integers(0, 2, size=(n, 1)) * 2 - 1).astype(dtype))
    Z = np.concatenate(probe_blocks, axis=1)
    s = Z.shape[1]

    AI_last = np.eye(m + 1, dtype=float)
    beta_last = None
    ll_last = None
    converged = False
    iterations = max_iter

    for iteration in range(max_iter):
        if constrain:
            theta = np.maximum(theta, _TINY)

        # Factor the current covariance in the chosen working precision.
        if safety_checks:
            try:
                _, Lc = _factor_theta(Vs_work, theta, dtype=dtype, overwrite_a=True)
            except linalg.LinAlgError:
                theta = _ensure_factorable_theta(theta, Vs, dtype=float)
                _, Lc = _factor_theta(Vs_work, theta, dtype=dtype, overwrite_a=True)
        else:
            V = theta[-1] * np.eye(n, dtype=dtype)
            for i, V_i in enumerate(Vs_work):
                V += dtype(theta[i]) * V_i

            try:
                Lc = cho_factor(V, lower=True, overwrite_a=True, check_finite=False)
            except linalg.LinAlgError:
                theta *= 1.05
                continue

        # Solve for Py and the stochastic probe transforms under the current V.
        beta_last, VinvX, XVX_inv, Py = _compute_beta_and_py(Lc, y_work, X_work)
        VinvZ = cho_solve(Lc, Z, check_finite=False)

        # Compute KPy and the stochastic score traces for each component.
        comp_py, traces = _stochastic_component_ops(Vs_work, Py, VinvZ, Z)
        KPy = np.empty((n, m + 1), dtype=dtype)
        KPy[:, :m] = comp_py
        KPy[:, m] = Py

        # Build the exact AI matrix from quadratic forms, just as in fast_greml.
        VinvKPy = cho_solve(Lc, KPy, check_finite=False).astype(np.float64)
        if X_model is None:
            PKPy = VinvKPy
            trace_corr = np.zeros(m + 1, dtype=float)
        else:
            VinvX64 = VinvX.astype(np.float64, copy=False)
            KPy64 = KPy.astype(np.float64, copy=False)
            PKPy = VinvKPy - VinvX64 @ (XVX_inv @ (VinvX64.T @ KPy64))
            trace_corr = np.empty(m + 1, dtype=float)
            for i, V_i in enumerate(Vs_work):
                trace_corr[i] = np.trace(XVX_inv @ (VinvX64.T @ (V_i.astype(np.float64) @ VinvX64)))
            trace_corr[m] = np.trace(XVX_inv @ (VinvX64.T @ VinvX64))

        KPy64 = KPy.astype(np.float64, copy=False)
        AI = 0.5 * (KPy64.T @ PKPy)
        AI_last = AI

        quad = np.einsum("n,nk->k", Py.astype(np.float64, copy=False), KPy64)
        trace_vec = np.empty(m + 1, dtype=float)
        trace_vec[:m] = traces
        trace_vec[m] = float((Z * VinvZ).sum(dtype=np.float64)) / s
        trace_vec -= trace_corr

        # Combine exact quadratic terms with stochastic trace terms for the score.
        grad = 0.5 * (quad - trace_vec)
        if safety_checks:
            delta = _solve_update_direction(AI, grad)
            theta, offsets, accepted = _line_search_update(
                theta,
                delta,
                Vs,
                constrain=constrain,
                dtype=float,
            )
            if not accepted:
                raise linalg.LinAlgError(
                    "AI_stochastic could not find a positive-definite REML update."
                )
        else:
            theta, offsets = _legacy_newton_step(AI, grad, theta, constrain=constrain)
        _print_iter_message(theta, offsets, iteration, verbose)

        if np.max(np.abs(offsets)) < tol and iteration >= 3:
            converged = True
            iterations = iteration + 1
            if verbose > 0:
                print(f"Converged after {iterations} iterations.")
            break
        if iteration == max_iter - 1 and verbose > 0:
            print(f"Reached maximum iterations ({max_iter}) without convergence.")

    # Log-likelihood is only computed once, at the final estimate.
    if safety_checks:
        theta = _ensure_factorable_theta(theta, Vs, dtype=float)
        _, Lc_final = _factor_theta(Vs, theta, dtype=float, overwrite_a=False)
    else:
        V_final = _build_v(Vs, theta)
        Lc_final = cho_factor(V_final, lower=True, check_finite=False)
    beta_last, _, _, _ = _compute_beta_and_py(Lc_final, y, X_model)
    ll_last = _compute_loglik(y=y, X=X_model, beta=beta_last, Lc=Lc_final)
    vcov = np.linalg.pinv(AI_last + 1e-12 * np.eye(m + 1))
    return _finalise_output(
        y=y,
        X_user=X_user,
        X_model=X_model,
        Vs=Vs,
        theta=theta,
        vcov=vcov,
        phenotype_variance=phenotype_variance,
        beta_model=beta_last,
        log_likelihood=ll_last,
        iterations=iterations,
        converged=converged,
        method="AI_stochastic",
        safety_checks=safety_checks,
    )


def _run_ai_exact(
    y: np.ndarray,
    Vs: list[np.ndarray],
    X_user: np.ndarray | None,
    X_model: np.ndarray | None,
    init: np.ndarray,
    tol: float,
    max_iter: int,
    constrain: bool,
    verbose: int,
    phenotype_variance: float,
    safety_checks: bool,
) -> dict:
    """Run AI-REML with exact traces and an explicit projection matrix."""
    m = len(Vs)
    theta = init.astype(float, copy=True)
    if safety_checks:
        theta = _ensure_factorable_theta(theta, Vs, dtype=float)
    AI_last = np.eye(m + 1, dtype=float)
    beta_last = None
    ll_last = None
    converged = False
    iterations = max_iter

    for iteration in range(max_iter):
        if constrain:
            theta = np.maximum(theta, _TINY)

        # Exact AI uses an explicit inverse/projection, so it is slower but deterministic.
        if safety_checks:
            try:
                _, Lc = _factor_theta(Vs, theta, dtype=float, overwrite_a=False)
            except linalg.LinAlgError:
                theta = _ensure_factorable_theta(theta, Vs, dtype=float)
                _, Lc = _factor_theta(Vs, theta, dtype=float, overwrite_a=False)
        else:
            V = _build_v(Vs, theta)
            try:
                Lc = cho_factor(V, lower=True, overwrite_a=False, check_finite=False)
            except linalg.LinAlgError:
                theta *= 1.05
                continue

        beta_last, VinvX, XVX_inv, Py = _compute_beta_and_py(Lc, y, X_model)
        Vinv = _potri_inverse(Lc)
        P = _get_projection_matrix(Vinv, X_model)
        KPy = np.column_stack([V_i @ Py for V_i in Vs] + [Py])
        PKPy = P @ KPy
        AI = 0.5 * (KPy.T @ PKPy)
        AI_last = AI

        quad = np.einsum("n,nk->k", Py, KPy)
        trace_vec = np.empty(m + 1, dtype=float)
        for i, V_i in enumerate(Vs):
            trace_vec[i] = np.trace(P @ V_i)
        trace_vec[m] = np.trace(P)

        grad = 0.5 * (quad - trace_vec)
        if safety_checks:
            delta = _solve_update_direction(AI, grad)
            theta, offsets, accepted = _line_search_update(
                theta,
                delta,
                Vs,
                constrain=constrain,
                dtype=float,
            )
            if not accepted:
                raise linalg.LinAlgError("AI could not find a positive-definite REML update.")
        else:
            theta, offsets = _legacy_newton_step(AI, grad, theta, constrain=constrain)
        _print_iter_message(theta, offsets, iteration, verbose)
        ll_last = _compute_loglik(y=y, X=X_model, beta=beta_last, Lc=Lc)

        if np.max(np.abs(offsets)) < tol and iteration >= 3:
            converged = True
            iterations = iteration + 1
            if verbose > 0:
                print(f"Converged after {iterations} iterations.")
            break
        if iteration == max_iter - 1 and verbose > 0:
            print(f"Reached maximum iterations ({max_iter}) without convergence.")

    vcov = np.linalg.pinv(AI_last + 1e-12 * np.eye(m + 1))
    return _finalise_output(
        y=y,
        X_user=X_user,
        X_model=X_model,
        Vs=Vs,
        theta=theta,
        vcov=vcov,
        phenotype_variance=phenotype_variance,
        beta_model=beta_last,
        log_likelihood=ll_last,
        iterations=iterations,
        converged=converged,
        method="AI",
        safety_checks=safety_checks,
    )


def _run_reml_em(
    y: np.ndarray,
    Vs: list[np.ndarray],
    X_user: np.ndarray | None,
    X_model: np.ndarray | None,
    init: np.ndarray,
    tol: float,
    max_iter: int,
    constrain: bool,
    verbose: int,
    phenotype_variance: float,
    safety_checks: bool,
) -> dict:
    """Run EM-REML with exact traces and exact projection matrices."""
    m = len(Vs)
    theta = init.astype(float, copy=True)
    if safety_checks:
        theta = _ensure_factorable_theta(theta, Vs, dtype=float)
    P_last = None
    beta_last = None
    ll_last = None
    converged = False
    iterations = max_iter

    sizes = np.array([V_i.shape[0] for V_i in Vs] + [y.shape[0]], dtype=float)

    for iteration in range(max_iter):
        if constrain:
            theta = np.maximum(theta, _TINY)

        # EM updates every component using the expected complete-data moments.
        if safety_checks:
            _, Lc = _factor_theta(Vs, theta, dtype=float, overwrite_a=False)
        else:
            V = _build_v(Vs, theta)
            Lc = cho_factor(V, lower=True, check_finite=False)
        beta_last, _, _, _ = _compute_beta_and_py(Lc, y, X_model)
        Vinv = cho_solve(Lc, np.eye(y.shape[0]), check_finite=False)
        P = _get_projection_matrix(Vinv, X_model)
        P_last = P
        offsets = np.zeros(m + 1, dtype=float)

        for i, V_i in enumerate(Vs + [np.eye(y.shape[0])]):
            offsets[i] = theta[i] ** 2 * ((y.T @ P @ V_i @ P @ y) - np.trace(P @ V_i)) / sizes[i]
        if safety_checks:
            theta, offsets, accepted = _line_search_update(
                theta,
                offsets,
                Vs,
                constrain=constrain,
                dtype=float,
            )
            if not accepted:
                raise linalg.LinAlgError("EM could not find a positive-definite REML update.")
        else:
            theta = theta + offsets
            if constrain:
                theta = np.maximum(theta, tol / 10)
        _print_iter_message(theta, offsets, iteration, verbose)
        ll_last = _compute_loglik(y=y, X=X_model, beta=beta_last, Lc=Lc)

        if np.max(np.abs(offsets)) < tol:
            converged = True
            iterations = iteration + 1
            if verbose > 0:
                print(f"Converged after {iterations} iterations.")
            break
        if iteration == max_iter - 1 and verbose > 0:
            print(f"Reached maximum iterations ({max_iter}) without convergence.")

    fisher = np.zeros((m + 1, m + 1), dtype=float)
    V_terms = Vs + [np.eye(y.shape[0])]
    for i in range(m + 1):
        for j in range(i, m + 1):
            fisher[i, j] = 0.5 * np.trace(P_last @ V_terms[i] @ P_last @ V_terms[j])
            fisher[j, i] = fisher[i, j]
    vcov = np.linalg.pinv(fisher)

    return _finalise_output(
        y=y,
        X_user=X_user,
        X_model=X_model,
        Vs=Vs,
        theta=theta,
        vcov=vcov,
        phenotype_variance=phenotype_variance,
        beta_model=beta_last,
        log_likelihood=ll_last,
        iterations=iterations,
        converged=converged,
        method="EM",
        safety_checks=safety_checks,
    )


def _run_reml_quad_exact(
    y: np.ndarray,
    Vs: list[np.ndarray],
    X_user: np.ndarray | None,
    X_model: np.ndarray | None,
    init: np.ndarray,
    method: str,
    tol: float,
    max_iter: int,
    constrain: bool,
    verbose: int,
    phenotype_variance: float,
    safety_checks: bool,
) -> dict:
    """Run the exact quadratic REML methods: NR, FS, or AI."""
    m = len(Vs)
    theta = init.astype(float, copy=True)
    if safety_checks:
        theta = _ensure_factorable_theta(theta, Vs, dtype=float)
    curvature_last = np.eye(m + 1, dtype=float)
    beta_last = None
    ll_last = None
    P_last = None
    converged = False
    iterations = max_iter
    V_terms = Vs + [np.eye(y.shape[0])]

    for iteration in range(max_iter):
        if constrain:
            theta = np.maximum(theta, _TINY)

        # The quadratic methods share the same score but differ in curvature.
        if safety_checks:
            _, Lc = _factor_theta(Vs, theta, dtype=float, overwrite_a=False)
        else:
            V = _build_v(Vs, theta)
            Lc = cho_factor(V, lower=True, check_finite=False)
        beta_last, _, _, _ = _compute_beta_and_py(Lc, y, X_model)
        Vinv = cho_solve(Lc, np.eye(y.shape[0]), check_finite=False)
        P = _get_projection_matrix(Vinv, X_model)
        P_last = P

        score = np.zeros(m + 1, dtype=float)
        fisher = np.zeros((m + 1, m + 1), dtype=float)
        ai = np.zeros((m + 1, m + 1), dtype=float)

        for i in range(m + 1):
            PV_i = P @ V_terms[i]
            score[i] = -0.5 * np.trace(PV_i) + 0.5 * (y.T @ PV_i @ P @ y)
            for j in range(i, m + 1):
                fisher[i, j] = 0.5 * np.trace(PV_i @ P @ V_terms[j])
                fisher[j, i] = fisher[i, j]
                ai[i, j] = 0.5 * y.T @ PV_i @ P @ V_terms[j] @ P @ y
                ai[j, i] = ai[i, j]

        if method == "FS":
            curvature = fisher
        elif method == "AI":
            curvature = ai
        elif method == "NR":
            curvature = -(fisher - 2.0 * ai)
        else:
            raise ValueError(f"Unsupported quadratic REML method '{method}'.")

        curvature_last = curvature
        if safety_checks:
            delta = _solve_update_direction(curvature, score)
            theta, offsets, accepted = _line_search_update(
                theta,
                delta,
                Vs,
                constrain=constrain,
                dtype=float,
            )
            if not accepted:
                raise linalg.LinAlgError(f"{method} could not find a positive-definite REML update.")
        else:
            offsets = np.linalg.pinv(curvature) @ score
            theta = theta + offsets
            if constrain:
                theta = np.maximum(theta, tol / 10)
        _print_iter_message(theta, offsets, iteration, verbose)
        ll_last = _compute_loglik(y=y, X=X_model, beta=beta_last, Lc=Lc)

        if np.max(np.abs(offsets)) < tol:
            converged = True
            iterations = iteration + 1
            if verbose > 0:
                print(f"Converged after {iterations} iterations.")
            break
        if iteration == max_iter - 1 and verbose > 0:
            print(f"Reached maximum iterations ({max_iter}) without convergence.")

    vcov = np.linalg.pinv(curvature_last)
    return _finalise_output(
        y=y,
        X_user=X_user,
        X_model=X_model,
        Vs=Vs,
        theta=theta,
        vcov=vcov,
        phenotype_variance=phenotype_variance,
        beta_model=beta_last,
        log_likelihood=ll_last,
        iterations=iterations,
        converged=converged,
        method=method,
        safety_checks=safety_checks,
    )


def run_REML(
    y: np.ndarray,
    Rs: Sequence[np.ndarray | None] | np.ndarray | None = None,
    Zs: Sequence[np.ndarray | None] | np.ndarray | None = None,
    X: np.ndarray | None = None,
    init: list[float] | np.ndarray | None = None,
    method: str = "AI_stochastic",
    tol: float = 1e-4,
    max_iter: int = 30,
    constrain: bool = False,
    std_y: bool = False,
    verbose: int = 2,
    n_probes: int = 50,
    seed: int = 42,
    dtype: np.dtype = np.float32,
    safety_checks: bool = True,
) -> dict:
    """
    Runs REML to estimate variance components while accounting for fixed effects.

    Parameters
    ----------
    y : ndarray, shape (N,)
        Outcome vector. It is always mean-centered. If ``std_y=True``, it is
        additionally standardized to variance 1 before fitting.
    Rs : ndarray, list of ndarray, or list containing ``None``, optional
        Random-effect cluster correlation/covariance matrices. If a single
        matrix is supplied, it is treated as a one-component model. If
        ``Zs[i]`` is ``None`` or ``Zs`` itself is ``None``, then ``Rs[i]``
        must be an ``N x N`` covariance matrix. If ``Zs[i]`` is provided and
        ``Rs[i]`` is ``None``, ``Rs[i]`` defaults to the identity matrix of
        size ``Zs[i].shape[1]``.
    Zs : ndarray, list of ndarray, or list containing ``None``, optional
        Random-effect design matrices. ``Zs[i]`` must have shape ``(N, q_i)``.
        When both ``Zs[i]`` and ``Rs[i]`` are provided, ``Rs[i]`` must have
        shape ``(q_i, q_i)``.
    X : ndarray, shape (N, p), optional
        Fixed-effect covariate matrix, not including an intercept column. If
        provided, REML automatically fits the fixed-effect design ``[1 | X]``
        internally and conditions on it when forming the projection matrix.
    init : array-like of length M, optional
        Initial values for the non-residual variance components only. The
        residual component is initialized automatically as phenotype variance
        minus the sum of the supplied values. If omitted, an HE-based warm
        start is used.
    method : {'AI_stochastic', 'AI', 'EM', 'NR', 'FS'}
        REML optimization method. Default is ``'AI_stochastic'``. If a
        Newton-style method leaves the positive-definite covariance region or
        fails to converge, ``run_REML()`` automatically retries a more stable
        method (``AI_stochastic -> AI -> FS`` and ``AI/NR -> FS``).
    tol : float, default 1e-5
        Convergence threshold on the maximum absolute parameter update.
    max_iter : int
        Maximum number of REML iterations. Default is 30.
    constrain : bool
        If True, constrains variance components to stay non-negative.
    std_y : bool, default False
        If True, standardize the phenotype to variance 1 after mean-centering.
    verbose : int, default 2
        Verbosity level. ``0`` is silent. Higher values print iteration-by-
        iteration component updates in the style of the original REML code.
    n_probes : int, default 50
        Number of Rademacher probe vectors used by ``method='AI_stochastic'``.
        Probe pairs are antithetic when possible.
    seed : int, default 42
        Random seed used to generate the stochastic trace probes.
    dtype : numpy dtype, default ``np.float32``
        Working precision for the stochastic AI solver. Ignored by the exact
        methods.
    safety_checks : bool, default True
        If True, keep REML updates inside the positive-definite covariance
        region and allow automatic fallback to stabler methods when the
        requested optimizer fails or does not converge. If False, use the
        older faster-but-riskier update path without those safeguards.
    Returns
    -------
    dict
        Nested dictionary with keys:
        ``var_comps`` containing ``est``, ``se``, and ``vcov``;
        ``var_y`` containing ``before_FE``, ``after_FE``, ``sum_comp``, and
        ``sum_comp_se``; ``fixed_effects`` containing ``est``, ``se``, and
        ``vcov``; ``algorithm`` containing ``method``, ``iterations``, and
        ``converged`` (plus ``safety_checks`` and ``requested_method`` if a
        fallback was used); and ``log_likelihood``. If fixed effects are
        supplied, the first entry in ``fixed_effects['est']`` is the
        intercept, followed by the user-supplied covariates.
    """
    if method not in {"AI_stochastic", "AI", "EM", "NR", "FS"}:
        raise ValueError("`method` must be one of 'AI_stochastic', 'AI', 'EM', 'NR', or 'FS'.")
    if n_probes < 1:
        raise ValueError("`n_probes` must be at least 1.")

    # Validate inputs once and convert all random effects to a common covariance form.
    prepared = _prepare_inputs(y=y, Rs=Rs, Zs=Zs, X=X, std_y=std_y)
    if init is None:
        y_init = prepared.y
        if prepared.X_model is not None:
            beta_init = np.linalg.pinv(prepared.X_model) @ prepared.y
            y_init = prepared.y - prepared.X_model @ beta_init
        residual_variance_init = float(y_init.var())
        # Use the cheaper GREML warm start when the model is already in dense-GRM form.
        if _can_use_fast_grm_warmstart(Rs=prepared.Rs, Zs=prepared.Zs, Vs=prepared.Vs):
            init_theta = _fast_grm_he_warmstart(
                Vs=prepared.Vs,
                y=y_init,
                phenotype_variance=residual_variance_init,
            )
        else:
            # Fall back to the more general HE regression warm start for arbitrary Z R Z' inputs.
            init_theta = _he_warmstart(
                y=y_init,
                Vs=prepared.Vs,
                residual_variance=residual_variance_init,
                X=None,
            )
    else:
        init = np.asarray(init, dtype=float)
        if init.ndim != 1 or init.shape[0] != prepared.n_random:
            raise ValueError(
                f"`init` must have length {prepared.n_random}, excluding the residual variance."
            )
        resid_init = prepared.phenotype_variance - float(init.sum())
        if resid_init <= 0:
            raise ValueError("Initial random-effect variances exceed the phenotype variance.")
        init_theta = np.concatenate([init, [resid_init]])

    def _dispatch(method_i: str) -> dict:
        if method_i == "AI_stochastic":
            return _run_ai_stochastic(
                y=prepared.y,
                Vs=prepared.Vs,
                X_user=prepared.X_user,
                X_model=prepared.X_model,
                init=init_theta,
                tol=tol,
                max_iter=max_iter,
                constrain=constrain,
                verbose=verbose,
                seed=seed,
                n_probes=n_probes,
                dtype=dtype,
                phenotype_variance=prepared.phenotype_variance,
                safety_checks=safety_checks,
            )
        if method_i == "AI":
            return _run_ai_exact(
                y=prepared.y,
                Vs=prepared.Vs,
                X_user=prepared.X_user,
                X_model=prepared.X_model,
                init=init_theta,
                tol=tol,
                max_iter=max_iter,
                constrain=constrain,
                verbose=verbose,
                phenotype_variance=prepared.phenotype_variance,
                safety_checks=safety_checks,
            )
        if method_i == "EM":
            return _run_reml_em(
                y=prepared.y,
                Vs=prepared.Vs,
                X_user=prepared.X_user,
                X_model=prepared.X_model,
                init=init_theta,
                tol=tol,
                max_iter=max_iter,
                constrain=constrain,
                verbose=verbose,
                phenotype_variance=prepared.phenotype_variance,
                safety_checks=safety_checks,
            )
        return _run_reml_quad_exact(
            y=prepared.y,
            Vs=prepared.Vs,
            X_user=prepared.X_user,
            X_model=prepared.X_model,
            init=init_theta,
            method=method_i,
            tol=tol,
            max_iter=max_iter,
            constrain=constrain,
            verbose=verbose,
            phenotype_variance=prepared.phenotype_variance,
            safety_checks=safety_checks,
        )

    fallback_order = {
        "AI_stochastic": ["AI", "FS"],
        "AI": ["FS"],
        "NR": ["FS"],
        "EM": [],
        "FS": [],
    }
    attempts = [method] + (fallback_order[method] if safety_checks else [])

    for i, method_i in enumerate(attempts):
        try:
            out = _dispatch(method_i)
        except linalg.LinAlgError as exc:
            if i == len(attempts) - 1:
                raise
            next_method = attempts[i + 1]
            warnings.warn(
                f"REML method '{method_i}' hit a non-positive-definite covariance; "
                f"retrying with '{next_method}'.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if out["algorithm"]["converged"] or i == len(attempts) - 1:
            if method_i != method:
                out["algorithm"]["requested_method"] = method
            return out

        next_method = attempts[i + 1]
        warnings.warn(
            f"REML method '{method_i}' did not converge within {max_iter} iterations; "
            f"retrying with '{next_method}'.",
            RuntimeWarning,
            stacklevel=2,
        )

    raise RuntimeError("run_REML exhausted all fallback methods without returning a result.")

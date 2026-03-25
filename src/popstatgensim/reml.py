"""
Fast REML and HE regression utilities for variance-component estimation.

This module combines the user-facing API style from ``statgen_functions.py``
with the faster AI-REML implementation from ``fast_greml/greml.py``.
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
    X: np.ndarray | None
    Vs: list[np.ndarray]
    phenotype_variance: float
    n_random: int


def _as_matrix_list(Bs: Sequence[np.ndarray] | np.ndarray, name: str) -> list[np.ndarray]:
    """Normalize a single covariance matrix or a sequence into a Python list."""
    if isinstance(Bs, np.ndarray):
        if Bs.ndim != 2:
            raise ValueError(f"`{name}` must be a 2D array or a list of 2D arrays.")
        return [Bs]
    matrices = [np.asarray(B, dtype=float) for B in Bs]
    if len(matrices) == 0:
        raise ValueError(f"`{name}` must contain at least one covariance matrix.")
    for i, B in enumerate(matrices):
        if B.ndim != 2:
            raise ValueError(f"`{name}[{i}]` must be a 2D array.")
    return matrices


def _prepare_inputs(
    y: np.ndarray,
    Bs: Sequence[np.ndarray] | np.ndarray,
    Zs: Sequence[np.ndarray | None] | None = None,
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
    Bs_list = _as_matrix_list(Bs, "Bs")
    if Zs is None:
        Zs_list = [None] * len(Bs_list)
    else:
        Zs_list = list(Zs)
        if len(Zs_list) != len(Bs_list):
            raise ValueError("`Zs` must have the same length as `Bs`.")

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

    # Convert each random effect from (Z, B) form into its N x N covariance.
    Vs = []
    for i, (B, Z) in enumerate(zip(Bs_list, Zs_list)):
        if Z is None:
            if B.shape != (n, n):
                raise ValueError(
                    f"`Bs[{i}]` must have shape ({n}, {n}) when `Zs[{i}]` is None."
                )
            V_i = np.asarray(B, dtype=float)
        else:
            Z = np.asarray(Z, dtype=float)
            if Z.ndim != 2 or Z.shape[0] != n:
                raise ValueError(f"`Zs[{i}]` must have shape ({n}, q_i).")
            if B.shape != (Z.shape[1], Z.shape[1]):
                raise ValueError(
                    f"`Bs[{i}]` must have shape ({Z.shape[1]}, {Z.shape[1]}) "
                    f"to match `Zs[{i}]`."
                )
            V_i = Z @ B @ Z.T
        if not np.isfinite(V_i).all():
            raise ValueError(f"The covariance matrix for component {i + 1} is non-finite.")
        Vs.append(0.5 * (V_i + V_i.T))

    return _PreparedInputs(
        y=y,
        X=X,
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


def _he_regression_core(
    y: np.ndarray,
    Vs: list[np.ndarray],
    phenotype_variance: float,
    X: np.ndarray | None = None,
    constrain: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the off-diagonal HE regression and map it onto variance components."""
    if X is not None:
        # HE regression is run on phenotypes residualized on the supplied covariates.
        X_ols = np.asarray(X, dtype=float)
        beta_ols = np.linalg.pinv(X_ols) @ y
        y = y - X_ols @ beta_ols

    n = y.shape[0]
    iu = np.triu_indices(n, k=1)
    # Build the pairwise regression problem over the upper triangle only.
    design = np.column_stack([V_i[iu] for V_i in Vs])
    response = y[iu[0]] * y[iu[1]]

    beta, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    if constrain:
        beta = np.maximum(beta, 0.0)
    resid = response - design @ beta
    df = max(design.shape[0] - design.shape[1], 1)
    sigma2 = float(resid @ resid) / df
    cov_beta = sigma2 * np.linalg.pinv(design.T @ design)

    # The residual component is the phenotype variance not explained by the
    # random effects estimated from off-diagonal covariance.
    transform = np.vstack([np.eye(len(Vs)), -np.ones((1, len(Vs)))])
    theta = np.concatenate([beta, [phenotype_variance - beta.sum()]])
    cov_theta = transform @ cov_beta @ transform.T
    if constrain:
        theta = np.maximum(theta, 0.0)
    return theta, cov_theta, beta


def _he_warmstart(
    y: np.ndarray,
    Vs: list[np.ndarray],
    phenotype_variance: float,
    X: np.ndarray | None = None,
) -> np.ndarray:
    """Generic HE-based warm start used when a fast dense-GRM shortcut is unavailable."""
    theta, _, _ = _he_regression_core(
        y=y,
        Vs=Vs,
        phenotype_variance=phenotype_variance,
        X=X,
        constrain=True,
    )
    theta = np.asarray(theta, dtype=float)
    theta = np.maximum(theta, max(phenotype_variance, 1.0) * 1e-3)
    total = theta.sum()
    if total <= 0:
        theta = np.full_like(theta, phenotype_variance / len(theta))
    else:
        theta *= phenotype_variance / total
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
    Bs: Sequence[np.ndarray] | np.ndarray,
    Zs: Sequence[np.ndarray | None] | None,
    Vs: list[np.ndarray],
) -> bool:
    """Return True when the user supplied plain N x N covariance matrices."""
    if Zs is not None and any(Z is not None for Z in Zs):
        return False
    Bs_list = _as_matrix_list(Bs, "Bs")
    if len(Bs_list) != len(Vs):
        return False
    for B, V in zip(Bs_list, Vs):
        if B.shape != V.shape:
            return False
        if np.asarray(B).shape[0] != np.asarray(B).shape[1]:
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


def _newton_step(AI: np.ndarray, grad: np.ndarray, theta: np.ndarray, constrain: bool) -> tuple[np.ndarray, np.ndarray]:
    """Take a Newton-style update, with optional step-halving for positivity."""
    try:
        delta = np.linalg.solve(AI + 1e-10 * np.eye(len(theta)), grad)
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


def _finalise_output(
    y: np.ndarray,
    X: np.ndarray | None,
    Vs: list[np.ndarray],
    theta: np.ndarray,
    vcov: np.ndarray,
    phenotype_variance: float,
    beta: np.ndarray | None,
    log_likelihood: float | None,
    iterations: int,
    converged: bool,
    method: str,
) -> dict:
    """Package common outputs and derived summary quantities into the public result dict."""
    theta = np.asarray(theta, dtype=float)
    vcov = np.asarray(vcov, dtype=float)
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))

    if X is None:
        fixed_effects_se = None
        fixed_effects_vcov = None
    else:
        V = _build_v(Vs, theta)
        try:
            Lc = cho_factor(V, lower=True, check_finite=False)
            VinvX = cho_solve(Lc, X, check_finite=False)
        except linalg.LinAlgError:
            VinvX = np.linalg.pinv(V) @ X
        fixed_effects_vcov = np.linalg.pinv(X.T @ VinvX)
        fixed_effects_se = np.sqrt(np.maximum(np.diag(fixed_effects_vcov), 0.0))

    return {
        "var_components": theta,
        "var_components_se": se,
        "var_components_vcov": vcov,
        "var_y": phenotype_variance,
        "fixed_effects": beta,
        "fixed_effects_se": fixed_effects_se,
        "fixed_effects_vcov": fixed_effects_vcov,
        "log_likelihood": log_likelihood,
        "iterations": iterations,
        "converged": converged,
        "method": method,
    }


def _run_ai_stochastic(
    y: np.ndarray,
    Vs: list[np.ndarray],
    X: np.ndarray | None,
    init: np.ndarray,
    tol: float,
    max_iter: int,
    constrain: bool,
    verbose: int,
    seed: int,
    n_probes: int,
    dtype: np.dtype,
    phenotype_variance: float,
) -> dict:
    """Run AI-REML using Hutchinson trace estimation."""
    n = y.shape[0]
    m = len(Vs)
    rng = np.random.default_rng(seed)
    theta = init.astype(np.float64, copy=True)
    Vs_work = [V_i.astype(dtype, copy=False) for V_i in Vs]
    y_work = y.astype(dtype, copy=False)
    X_work = None if X is None else X.astype(dtype, copy=False)
    X64 = None if X is None else X.astype(np.float64, copy=False)

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
        if X is None:
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
        theta, offsets = _newton_step(AI, grad, theta, constrain=constrain)
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
    V_final = _build_v(Vs, theta)
    Lc_final = cho_factor(V_final, lower=True, check_finite=False)
    beta_last, _, _, _ = _compute_beta_and_py(Lc_final, y, X)
    ll_last = _compute_loglik(y=y, X=X, beta=beta_last, Lc=Lc_final)
    vcov = np.linalg.pinv(AI_last + 1e-12 * np.eye(m + 1))
    return _finalise_output(
        y=y,
        X=X,
        Vs=Vs,
        theta=theta,
        vcov=vcov,
        phenotype_variance=phenotype_variance,
        beta=beta_last,
        log_likelihood=ll_last,
        iterations=iterations,
        converged=converged,
        method="AI_stochastic",
    )


def _run_ai_exact(
    y: np.ndarray,
    Vs: list[np.ndarray],
    X: np.ndarray | None,
    init: np.ndarray,
    tol: float,
    max_iter: int,
    constrain: bool,
    verbose: int,
    phenotype_variance: float,
) -> dict:
    """Run AI-REML with exact traces and an explicit projection matrix."""
    m = len(Vs)
    theta = init.astype(float, copy=True)
    AI_last = np.eye(m + 1, dtype=float)
    beta_last = None
    ll_last = None
    converged = False
    iterations = max_iter

    for iteration in range(max_iter):
        if constrain:
            theta = np.maximum(theta, _TINY)

        # Exact AI uses an explicit inverse/projection, so it is slower but deterministic.
        V = _build_v(Vs, theta)
        try:
            Lc = cho_factor(V, lower=True, overwrite_a=False, check_finite=False)
        except linalg.LinAlgError:
            theta *= 1.05
            continue

        beta_last, VinvX, XVX_inv, Py = _compute_beta_and_py(Lc, y, X)
        Vinv = _potri_inverse(Lc)
        P = _get_projection_matrix(Vinv, X)
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
        theta, offsets = _newton_step(AI, grad, theta, constrain=constrain)
        _print_iter_message(theta, offsets, iteration, verbose)
        ll_last = _compute_loglik(y=y, X=X, beta=beta_last, Lc=Lc)

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
        X=X,
        Vs=Vs,
        theta=theta,
        vcov=vcov,
        phenotype_variance=phenotype_variance,
        beta=beta_last,
        log_likelihood=ll_last,
        iterations=iterations,
        converged=converged,
        method="AI",
    )


def _run_reml_em(
    y: np.ndarray,
    Vs: list[np.ndarray],
    X: np.ndarray | None,
    init: np.ndarray,
    tol: float,
    max_iter: int,
    constrain: bool,
    verbose: int,
    phenotype_variance: float,
) -> dict:
    """Run EM-REML with exact traces and exact projection matrices."""
    m = len(Vs)
    theta = init.astype(float, copy=True)
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
        V = _build_v(Vs, theta)
        Lc = cho_factor(V, lower=True, check_finite=False)
        beta_last, _, _, _ = _compute_beta_and_py(Lc, y, X)
        Vinv = cho_solve(Lc, np.eye(y.shape[0]), check_finite=False)
        P = _get_projection_matrix(Vinv, X)
        P_last = P
        offsets = np.zeros(m + 1, dtype=float)

        for i, V_i in enumerate(Vs + [np.eye(y.shape[0])]):
            offsets[i] = theta[i] ** 2 * ((y.T @ P @ V_i @ P @ y) - np.trace(P @ V_i)) / sizes[i]
        theta = theta + offsets
        if constrain:
            theta = np.maximum(theta, tol / 10)
        _print_iter_message(theta, offsets, iteration, verbose)
        ll_last = _compute_loglik(y=y, X=X, beta=beta_last, Lc=Lc)

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
        X=X,
        Vs=Vs,
        theta=theta,
        vcov=vcov,
        phenotype_variance=phenotype_variance,
        beta=beta_last,
        log_likelihood=ll_last,
        iterations=iterations,
        converged=converged,
        method="EM",
    )


def _run_reml_quad_exact(
    y: np.ndarray,
    Vs: list[np.ndarray],
    X: np.ndarray | None,
    init: np.ndarray,
    method: str,
    tol: float,
    max_iter: int,
    constrain: bool,
    verbose: int,
    phenotype_variance: float,
) -> dict:
    """Run the exact quadratic REML methods: NR, FS, or AI."""
    m = len(Vs)
    theta = init.astype(float, copy=True)
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
        V = _build_v(Vs, theta)
        Lc = cho_factor(V, lower=True, check_finite=False)
        beta_last, _, _, _ = _compute_beta_and_py(Lc, y, X)
        Vinv = cho_solve(Lc, np.eye(y.shape[0]), check_finite=False)
        P = _get_projection_matrix(Vinv, X)
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
        offsets = np.linalg.pinv(curvature) @ score
        theta = theta + offsets
        if constrain:
            theta = np.maximum(theta, tol / 10)
        _print_iter_message(theta, offsets, iteration, verbose)
        ll_last = _compute_loglik(y=y, X=X, beta=beta_last, Lc=Lc)

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
        X=X,
        Vs=Vs,
        theta=theta,
        vcov=vcov,
        phenotype_variance=phenotype_variance,
        beta=beta_last,
        log_likelihood=ll_last,
        iterations=iterations,
        converged=converged,
        method=method,
    )


def run_HEreg(
    y: np.ndarray,
    Bs: Sequence[np.ndarray] | np.ndarray,
    Zs: Sequence[np.ndarray | None] | None = None,
    X: np.ndarray | None = None,
    constrain: bool = False,
    std_y: bool = False,
    verbose: int = 1,
) -> dict:
    """
    Runs Haseman-Elston regression for one or more variance components.

    Parameters
    ----------
    y : ndarray, shape (N,)
        Outcome vector. The vector is always mean-centered. If ``std_y=True``,
        it is also scaled to variance 1 before fitting.
    Bs : ndarray or list of ndarray
        Random-effect covariance matrices. If a single matrix is supplied, it
        is treated as a one-component model. If a list is supplied, each
        element is one random-effect covariance matrix.
    Zs : list of ndarray or None, optional
        Random-effect design matrices. ``Zs[i]`` must have shape ``(N, q_i)``
        and ``Bs[i]`` must then have shape ``(q_i, q_i)``. If ``Zs[i]`` is
        ``None``, the corresponding ``Bs[i]`` is treated as already being an
        ``N x N`` covariance matrix. If ``Zs`` itself is ``None``, all random
        effects are assumed to be supplied directly as ``N x N`` matrices.
    X : ndarray, shape (N, p), optional
        Fixed-effect covariate matrix. If provided, the phenotype is first
        residualized on ``X`` by ordinary least squares before running HE
        regression.
    constrain : bool, default False
        If True, negative component estimates are clipped to zero.
    std_y : bool, default False
        If True, standardize the phenotype to variance 1 after mean-centering.
    verbose : int, default 1
        If greater than 0, print a completion message.

    Returns
    -------
    dict
        Dictionary with the same core fields as ``run_REML``:
        ``var_components``, ``var_components_se``, ``var_components_vcov``,
        ``var_y``, ``fixed_effects``, ``fixed_effects_se``,
        ``fixed_effects_vcov``, ``log_likelihood``, ``iterations``,
        ``converged``, and ``method``. For HE regression,
        ``log_likelihood`` is returned as ``None`` and ``iterations`` is
        always 1.
    """
    prepared = _prepare_inputs(y=y, Bs=Bs, Zs=Zs, X=X, std_y=std_y)
    theta, vcov, beta_nonresid = _he_regression_core(
        y=prepared.y,
        Vs=prepared.Vs,
        phenotype_variance=prepared.phenotype_variance,
        X=prepared.X,
        constrain=constrain,
    )

    if prepared.X is None:
        beta = None
        fixed_effects_vcov = None
        fixed_effects_se = None
    else:
        beta = np.linalg.pinv(prepared.X) @ prepared.y
        residuals = prepared.y - prepared.X @ beta
        sigma2 = float(residuals @ residuals) / max(prepared.X.shape[0] - prepared.X.shape[1], 1)
        fixed_effects_vcov = sigma2 * np.linalg.pinv(prepared.X.T @ prepared.X)
        fixed_effects_se = np.sqrt(np.maximum(np.diag(fixed_effects_vcov), 0.0))

    out = _finalise_output(
        y=prepared.y,
        X=prepared.X,
        Vs=prepared.Vs,
        theta=theta,
        vcov=vcov,
        phenotype_variance=prepared.phenotype_variance,
        beta=beta,
        log_likelihood=None,
        iterations=1,
        converged=True,
        method="HE",
    )
    out["fixed_effects_se"] = fixed_effects_se
    out["fixed_effects_vcov"] = fixed_effects_vcov
    out["he_nonresid_estimates"] = beta_nonresid
    return out


def run_REML(
    y: np.ndarray,
    Bs: Sequence[np.ndarray] | np.ndarray,
    Zs: Sequence[np.ndarray | None] | None = None,
    X: np.ndarray | None = None,
    init: list[float] | np.ndarray | None = None,
    method: str = "AI_stochastic",
    tol: float = 1e-5,
    max_iter: int = 30,
    constrain: bool = False,
    std_y: bool = False,
    verbose: int = 2,
    n_probes: int = 50,
    seed: int = 42,
    dtype: np.dtype = np.float32,
) -> dict:
    """
    Runs REML to estimate variance components while accounting for fixed effects.

    Parameters
    ----------
    y : ndarray, shape (N,)
        Outcome vector. It is always mean-centered. If ``std_y=True``, it is
        additionally standardized to variance 1 before fitting.
    Bs : ndarray or list of ndarray
        Random-effect covariance matrices. If a single matrix is supplied, it
        is treated as a one-component model. If a list is supplied, each
        element defines one random-effect covariance matrix.
    Zs : list of ndarray or None, optional
        Random-effect design matrices. ``Zs[i]`` must have shape ``(N, q_i)``
        and ``Bs[i]`` must then have shape ``(q_i, q_i)``. If ``Zs[i]`` is
        ``None``, the corresponding ``Bs[i]`` is treated as already being an
        ``N x N`` covariance matrix. If ``Zs`` itself is ``None``, all random
        effects are assumed to be supplied directly as ``N x N`` matrices.
    X : ndarray, shape (N, p), optional
        Fixed-effect covariate matrix. If provided, REML conditions on these
        fixed effects when forming the projection matrix.
    init : array-like of length M, optional
        Initial values for the non-residual variance components only. The
        residual component is initialized automatically as phenotype variance
        minus the sum of the supplied values. If omitted, an HE-based warm
        start is used.
    method : {'AI_stochastic', 'AI', 'EM', 'NR', 'FS'}
        REML optimization method. Default is ``'AI_stochastic'``.
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

    Returns
    -------
    dict
        Dictionary with keys:
        ``var_components``, ``var_components_se``, ``var_components_vcov``,
        ``var_y``, ``fixed_effects``, ``fixed_effects_se``,
        ``fixed_effects_vcov``, ``log_likelihood``, ``iterations``,
        ``converged``, and ``method``.
    """
    if method not in {"AI_stochastic", "AI", "EM", "NR", "FS"}:
        raise ValueError("`method` must be one of 'AI_stochastic', 'AI', 'EM', 'NR', or 'FS'.")
    if n_probes < 1:
        raise ValueError("`n_probes` must be at least 1.")

    # Validate inputs once and convert all random effects to a common covariance form.
    prepared = _prepare_inputs(y=y, Bs=Bs, Zs=Zs, X=X, std_y=std_y)
    if init is None:
        y_init = prepared.y
        if prepared.X is not None:
            beta_init = np.linalg.pinv(prepared.X) @ prepared.y
            y_init = prepared.y - prepared.X @ beta_init
        # Use the cheaper GREML warm start when the model is already in dense-GRM form.
        if _can_use_fast_grm_warmstart(Bs=Bs, Zs=Zs, Vs=prepared.Vs):
            init_theta = _fast_grm_he_warmstart(
                Vs=prepared.Vs,
                y=y_init,
                phenotype_variance=prepared.phenotype_variance,
            )
        else:
            # Fall back to the more general HE regression warm start for arbitrary Z B Z' inputs.
            init_theta = _he_warmstart(
                y=prepared.y,
                Vs=prepared.Vs,
                phenotype_variance=prepared.phenotype_variance,
                X=prepared.X,
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

    if method == "AI_stochastic":
        return _run_ai_stochastic(
            y=prepared.y,
            Vs=prepared.Vs,
            X=prepared.X,
            init=init_theta,
            tol=tol,
            max_iter=max_iter,
            constrain=constrain,
            verbose=verbose,
            seed=seed,
            n_probes=n_probes,
            dtype=dtype,
            phenotype_variance=prepared.phenotype_variance,
        )
    if method == "AI":
        return _run_ai_exact(
            y=prepared.y,
            Vs=prepared.Vs,
            X=prepared.X,
            init=init_theta,
            tol=tol,
            max_iter=max_iter,
            constrain=constrain,
            verbose=verbose,
            phenotype_variance=prepared.phenotype_variance,
        )
    if method == "EM":
        return _run_reml_em(
            y=prepared.y,
            Vs=prepared.Vs,
            X=prepared.X,
            init=init_theta,
            tol=tol,
            max_iter=max_iter,
            constrain=constrain,
            verbose=verbose,
            phenotype_variance=prepared.phenotype_variance,
        )
    return _run_reml_quad_exact(
        y=prepared.y,
        Vs=prepared.Vs,
        X=prepared.X,
        init=init_theta,
        method=method,
        tol=tol,
        max_iter=max_iter,
        constrain=constrain,
        verbose=verbose,
        phenotype_variance=prepared.phenotype_variance,
    )

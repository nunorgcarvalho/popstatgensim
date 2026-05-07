"""GWAS entrypoints."""

from __future__ import annotations

import numpy as np


def _validate_gwas_inputs(
    y: np.ndarray,
    G: np.ndarray,
    covariates: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate and coerce GWAS inputs to dense float arrays."""
    y = np.asarray(y, dtype=float)
    G = np.asarray(G, dtype=float)

    if y.ndim != 1:
        raise ValueError("`y` must be a 1D array.")
    if G.ndim != 2:
        raise ValueError("`G` must be a 2D array.")
    if G.shape[0] != y.shape[0]:
        raise ValueError("`y` and `G` must have the same number of samples.")

    if covariates is None:
        return y, G, None

    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim != 2:
        raise ValueError("`covariates` must be a 2D array with shape (N, C).")
    if covariates.shape[0] != y.shape[0]:
        raise ValueError("`covariates` must have the same number of rows as `y`.")
    return y, G, covariates


def _standardize_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Mean-center and variance-standardize a vector."""
    mean = float(np.mean(values))
    sd = float(np.std(values))
    if not np.isfinite(sd) or sd == 0.0:
        raise ValueError(f"Cannot standardize `{name}` because it has zero variance.")
    return (values - mean) / sd


def run_GWAS(
    y: np.ndarray,
    G: np.ndarray,
    covariates: np.ndarray | None = None,
    standardize_y: bool = True,
    detailed_output: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Run a univariate linear-regression GWAS across all variants.
    """
    y, G, covariates = _validate_gwas_inputs(y=y, G=G, covariates=covariates)
    if standardize_y:
        y = _standardize_vector(y, name="y")

    n_samples, n_variants = G.shape
    n_covariates = 0 if covariates is None else covariates.shape[1]

    if verbose:
        print("GWAS preprocessing: building design components.")

    intercept = np.ones((n_samples, 1), dtype=float)
    X_base = intercept if covariates is None else np.column_stack((intercept, covariates))
    n_params = X_base.shape[1] + 1
    dof = n_samples - n_params
    if dof <= 0:
        raise ValueError(
            "Not enough samples to fit GWAS regressions after accounting for the intercept, "
            "covariates, and SNP effect."
        )

    intercept_est = np.empty(n_variants, dtype=float)
    intercept_se = np.empty(n_variants, dtype=float)
    beta_est = np.empty(n_variants, dtype=float)
    beta_se = np.empty(n_variants, dtype=float)
    if n_covariates > 0:
        covar_est = np.empty((n_variants, n_covariates), dtype=float)
        covar_se = np.empty((n_variants, n_covariates), dtype=float)
    else:
        covar_est = np.empty((n_variants, 0), dtype=float)
        covar_se = np.empty((n_variants, 0), dtype=float)

    progress_step = max(n_variants // 10, 1)
    if verbose:
        print("GWAS regression: fitting one linear model per variant.")

    for j in range(n_variants):
        if verbose and (j % progress_step == 0 or j == n_variants - 1):
            print(f"GWAS progress: completed {j + 1}/{n_variants} variants.")

        x_j = G[:, j:j + 1]
        X_j = np.column_stack((X_base, x_j))
        xtx_inv = np.linalg.pinv(X_j.T @ X_j)
        beta_full = xtx_inv @ (X_j.T @ y)
        resid = y - (X_j @ beta_full)
        sigma2 = float(resid @ resid) / dof
        vcov = sigma2 * xtx_inv
        se_full = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

        intercept_est[j] = beta_full[0]
        intercept_se[j] = se_full[0]
        if n_covariates > 0:
            covar_est[j, :] = beta_full[1:-1]
            covar_se[j, :] = se_full[1:-1]
        beta_est[j] = beta_full[-1]
        beta_se[j] = se_full[-1]

    return {
        "n_samples": n_samples,
        "n_variants": n_variants,
        "n_covariates": n_covariates,
        "standardize_y": bool(standardize_y),
        "detailed_output": bool(detailed_output),
        "beta_est": beta_est,
        "beta_se": beta_se,
        **(
            {
                "intercept_est": intercept_est,
                "intercept_se": intercept_se,
                "covar_est": covar_est,
                "covar_se": covar_se,
            }
            if detailed_output else
            {}
        ),
    }


__all__ = ["run_GWAS"]

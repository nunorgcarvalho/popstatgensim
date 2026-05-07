"""GWAS entrypoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pgs import get_exp_PGS_R2


@dataclass
class GWASresult:
    """
    Container for GWAS outputs.

    Attributes
    ----------
    N : int
        Number of samples used in the GWAS.
    M : int
        Number of variants tested in the GWAS.
    """

    trait_name: str | None
    N: int
    M: int
    n_covariates: int
    standardize_y: bool
    standardize_geno: bool
    detailed_output: bool
    within_family: str | None
    beta_est: np.ndarray
    beta_se: np.ndarray
    gamma_est: np.ndarray | None = None
    gamma_se: np.ndarray | None = None
    intercept_est: np.ndarray | None = None
    intercept_se: np.ndarray | None = None
    covar_est: np.ndarray | None = None
    covar_se: np.ndarray | None = None

    def get_exp_PGS_R2(self, h2: float) -> float:
        """
        Return the expected PGS R^2 using this GWAS sample size and variant count.

        Reference
        ---------
        Daetwyler et al. 2008, PLoS ONE.
        """
        return get_exp_PGS_R2(h2=h2, N=self.N, M=self.M)


def _validate_gwas_inputs(
    y: np.ndarray,
    G: np.ndarray,
    covariates: np.ndarray | None,
    G_par: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Validate and coerce GWAS inputs to dense float arrays."""
    y = np.asarray(y, dtype=float)
    G = np.asarray(G, dtype=float)

    if y.ndim != 1:
        raise ValueError("`y` must be a 1D array.")
    if G.ndim != 2:
        raise ValueError("`G` must be a 2D array.")
    if G.shape[0] != y.shape[0]:
        raise ValueError("`y` and `G` must have the same number of samples.")

    if G_par is not None:
        G_par = np.asarray(G_par, dtype=float)
        if G_par.ndim != 2:
            raise ValueError("`G_par` must be a 2D array.")
        if G_par.shape != G.shape:
            raise ValueError("`G_par` must have the same shape as `G`.")

    if covariates is None:
        return y, G, None, G_par

    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim != 2:
        raise ValueError("`covariates` must be a 2D array with shape (N, C).")
    if covariates.shape[0] != y.shape[0]:
        raise ValueError("`covariates` must have the same number of rows as `y`.")
    return y, G, covariates, G_par


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
    G_par: np.ndarray | None = None,
    standardize_y: bool = True,
    detailed_output: bool = False,
    verbose: bool = False,
    trait_name: str | None = None,
    standardize_geno: bool = True,
    within_family: str | None = None,
) -> GWASresult:
    """
    Run a univariate linear-regression GWAS across all variants.
    """
    within_family = None if within_family is None else str(within_family)
    if within_family not in {None, 'Gpar'}:
        raise ValueError("`within_family` must be None or 'Gpar'.")

    y, G, covariates, G_par = _validate_gwas_inputs(
        y=y,
        G=G,
        covariates=covariates,
        G_par=G_par,
    )
    if within_family == 'Gpar' and G_par is None:
        raise ValueError("`G_par` must be provided when `within_family='Gpar'`.")
    if standardize_y:
        y = _standardize_vector(y, name="y")

    n_samples, n_variants = G.shape
    n_covariates = 0 if covariates is None else covariates.shape[1]

    if verbose:
        print("GWAS preprocessing: building design components.")

    intercept = np.ones((n_samples, 1), dtype=float)
    X_base = intercept if covariates is None else np.column_stack((intercept, covariates))
    n_params = X_base.shape[1] + 1 + (1 if within_family == 'Gpar' else 0)
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
    if within_family == 'Gpar':
        gamma_est = np.empty(n_variants, dtype=float)
        gamma_se = np.empty(n_variants, dtype=float)
    else:
        gamma_est = None
        gamma_se = None
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
        if within_family == 'Gpar':
            gpar_j = G_par[:, j:j + 1]
            X_j = np.column_stack((X_base, gpar_j, x_j))
        else:
            X_j = np.column_stack((X_base, x_j))
        xtx_inv = np.linalg.pinv(X_j.T @ X_j)
        beta_full = xtx_inv @ (X_j.T @ y)
        resid = y - (X_j @ beta_full)
        sigma2 = float(resid @ resid) / dof
        vcov = sigma2 * xtx_inv
        se_full = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

        intercept_est[j] = beta_full[0]
        intercept_se[j] = se_full[0]
        covar_stop = 1 + n_covariates
        if n_covariates > 0:
            covar_est[j, :] = beta_full[1:covar_stop]
            covar_se[j, :] = se_full[1:covar_stop]
        if within_family == 'Gpar':
            gamma_est[j] = beta_full[covar_stop]
            gamma_se[j] = se_full[covar_stop]
        beta_est[j] = beta_full[-1]
        beta_se[j] = se_full[-1]

    return GWASresult(
        trait_name=trait_name,
        N=n_samples,
        M=n_variants,
        n_covariates=n_covariates,
        standardize_y=bool(standardize_y),
        standardize_geno=bool(standardize_geno),
        detailed_output=bool(detailed_output),
        within_family=within_family,
        beta_est=beta_est,
        beta_se=beta_se,
        gamma_est=gamma_est,
        gamma_se=gamma_se,
        intercept_est=intercept_est if detailed_output else None,
        intercept_se=intercept_se if detailed_output else None,
        covar_est=covar_est if detailed_output else None,
        covar_se=covar_se if detailed_output else None,
    )


__all__ = ["GWASresult", "run_GWAS"]

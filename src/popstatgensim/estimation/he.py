"""Haseman-Elston estimation entrypoints."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .reml import _compute_fe_ols_summary, _finalise_output, _prepare_inputs


def _he_regression_core(
    y: np.ndarray,
    Vs: list[np.ndarray],
    residual_variance: float,
    X: np.ndarray | None = None,
    constrain: bool = False,
    pair_relatedness_threshold: float | None = None,
    pair_filter_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
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
    n_pairs_total = int(response.shape[0])

    if pair_relatedness_threshold is not None:
        if pair_filter_matrix is None:
            raise ValueError(
                "`pair_filter_matrix` must be provided when `pair_relatedness_threshold` "
                "is specified."
            )
        pair_filter_matrix = np.asarray(pair_filter_matrix, dtype=float)
        if pair_filter_matrix.shape != (n, n):
            raise ValueError(f"`pair_filter_matrix` must have shape ({n}, {n}).")
        pair_mask = pair_filter_matrix[iu] <= pair_relatedness_threshold
        if not np.any(pair_mask):
            raise ValueError(
                "Pair filtering removed every off-diagonal pair from HE regression."
            )
        design = design[pair_mask, :]
        response = response[pair_mask]

    n_pairs_used = int(response.shape[0])
    if n_pairs_used <= design.shape[1]:
        raise ValueError(
            "Not enough pairwise observations remain to fit HE regression after pair "
            "filtering."
        )

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
    theta = np.concatenate([beta, [residual_variance - beta.sum()]])
    cov_theta = transform @ cov_beta @ transform.T
    if constrain:
        theta = np.maximum(theta, 0.0)
    return theta, cov_theta, beta, n_pairs_used, n_pairs_total


def run_HEreg(
    y: np.ndarray,
    Rs: Sequence[np.ndarray | None] | np.ndarray | None = None,
    Zs: Sequence[np.ndarray | None] | np.ndarray | None = None,
    X: np.ndarray | None = None,
    constrain: bool = False,
    std_y: bool = False,
    pair_relatedness_threshold: float | None = None,
    pair_filter_matrix: np.ndarray | None = None,
    verbose: int = 1,
) -> dict:
    """
    Runs Haseman-Elston regression for one or more variance components.

    Parameters
    ----------
    y : ndarray, shape (N,)
        Outcome vector. The vector is always mean-centered. If ``std_y=True``,
        it is also scaled to variance 1 before fitting.
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
        provided, an intercept is automatically added internally and the
        phenotype is first residualized on ``[1 | X]`` by ordinary least
        squares before running HE regression.
    constrain : bool, default False
        If True, negative component estimates are clipped to zero.
    std_y : bool, default False
        If True, standardize the phenotype to variance 1 after mean-centering.
    pair_relatedness_threshold : float, optional
        If provided, HE regression excludes pairwise observations whose relatedness in
        `pair_filter_matrix` exceeds this threshold. By default, when a threshold is
        supplied and `pair_filter_matrix` is omitted, the first fitted covariance
        matrix is used, which for the usual one-component model is the SNP GRM.
    pair_filter_matrix : ndarray, shape (N, N), optional
        Matrix used to decide which pairs are excluded when
        `pair_relatedness_threshold` is set. This can be a SNP GRM, true-IBD
        relatedness matrix, or any other N x N pairwise relatedness measure.
    verbose : int, default 1
        If greater than 0, print a completion message.
    Returns
    -------
    dict
        Nested dictionary with keys:
        ``var_comps`` containing ``est``, ``se``, and ``vcov``;
        ``var_y`` containing ``before_FE``, ``after_FE``, ``sum_comp``, and
        ``sum_comp_se``; ``fixed_effects`` containing ``est``, ``se``, and
        ``vcov``; ``algorithm`` containing ``method``, ``iterations``, and
        ``converged``; and ``log_likelihood``. If fixed effects are supplied,
        the first entry in ``fixed_effects['est']`` is the intercept, followed
        by the user-supplied covariates. For HE regression, ``log_likelihood``
        is returned as ``None`` and ``algorithm['iterations']`` is always 1.
    """
    prepared = _prepare_inputs(y=y, Rs=Rs, Zs=Zs, X=X, std_y=std_y)
    beta_full, beta_se_full, beta_vcov_full, residual_variance = _compute_fe_ols_summary(
        y=prepared.y,
        X_model=prepared.X_model,
    )
    if pair_relatedness_threshold is not None and pair_filter_matrix is None:
        pair_filter_matrix = prepared.Vs[0]

    theta, vcov, _, n_pairs_used, n_pairs_total = _he_regression_core(
        y=prepared.y,
        Vs=prepared.Vs,
        residual_variance=residual_variance,
        X=prepared.X_model,
        constrain=constrain,
        pair_relatedness_threshold=pair_relatedness_threshold,
        pair_filter_matrix=pair_filter_matrix,
    )

    out = _finalise_output(
        y=prepared.y,
        X_user=prepared.X_user,
        X_model=prepared.X_model,
        Vs=prepared.Vs,
        theta=theta,
        vcov=vcov,
        phenotype_variance=prepared.phenotype_variance,
        beta_model=beta_full,
        log_likelihood=None,
        iterations=1,
        converged=True,
        method="HE",
        safety_checks=False,
    )
    if prepared.X_model is not None:
        out["fixed_effects"]["est"] = beta_full
        out["fixed_effects"]["se"] = beta_se_full
        out["fixed_effects"]["vcov"] = beta_vcov_full
    # HE does not estimate the residual component through likelihood; we report
    # it on the post-fixed-effect variance scale instead.
    out["var_comps"]["est"][-1] = residual_variance - np.sum(out["var_comps"]["est"][:-1])
    out["algorithm"]["n_pairs_total"] = n_pairs_total
    out["algorithm"]["n_pairs_used"] = n_pairs_used
    out["algorithm"]["pair_relatedness_threshold"] = pair_relatedness_threshold
    return out


__all__ = ["run_HEreg"]

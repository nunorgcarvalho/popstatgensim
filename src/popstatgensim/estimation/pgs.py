"""Polygenic-score estimation helpers.

References
----------
Daetwyler et al. 2008, PLoS ONE.
"""

from __future__ import annotations


def get_exp_PGS_R2(h2: float, N: int, M: int) -> float:
    """
    Return the expected PGS R^2 under the Daetwyler et al. approximation.

    Reference
    ---------
    Daetwyler et al. 2008, PLoS ONE.
    """
    h2 = float(h2)
    N = int(N)
    M = int(M)
    return (h2 ** 2) / (h2 + (M / N))


def get_PGS_N_for_R2(h2: float, R2: float, M: int) -> float:
    """
    Return the GWAS sample size implied by the Daetwyler et al. approximation.

    Reference
    ---------
    Daetwyler et al. 2008, PLoS ONE.
    """
    h2 = float(h2)
    R2 = float(R2)
    M = int(M)
    return (R2 * M) / (h2 * (h2 - R2))


__all__ = ["get_exp_PGS_R2", "get_PGS_N_for_R2"]

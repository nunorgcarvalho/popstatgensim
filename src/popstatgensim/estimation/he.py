"""Haseman-Elston estimation entrypoints."""

from .reml import run_HEreg, run_HEreg as run_HE_regression

__all__ = ["run_HEreg", "run_HE_regression"]

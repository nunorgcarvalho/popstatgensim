"""General utility helpers for popstatgensim."""

from .misc import get_pop_kwargs, to_bits
from .matrix_metrics import (
    matrix_axiswise_correlation,
    matrix_mean_absolute_error,
    matrix_mean_squared_error,
    matrix_pearson_correlation,
    matrix_root_mean_squared_error,
    summarize_matrix_error,
)
from .stats import corr, report_CI

__all__ = [
    "corr",
    "report_CI",
    "get_pop_kwargs",
    "to_bits",
    "matrix_mean_squared_error",
    "matrix_root_mean_squared_error",
    "matrix_mean_absolute_error",
    "matrix_pearson_correlation",
    "matrix_axiswise_correlation",
    "summarize_matrix_error",
]

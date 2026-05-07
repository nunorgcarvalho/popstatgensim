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
from .stats import LinearRegressionResult, corr, fit_linear_regression, report_CI, standardize_vector

__all__ = [
    "corr",
    "LinearRegressionResult",
    "fit_linear_regression",
    "report_CI",
    "standardize_vector",
    "get_pop_kwargs",
    "to_bits",
    "matrix_mean_squared_error",
    "matrix_root_mean_squared_error",
    "matrix_mean_absolute_error",
    "matrix_pearson_correlation",
    "matrix_axiswise_correlation",
    "summarize_matrix_error",
]

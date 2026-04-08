"""Compatibility wrapper for utilities moved into subpackages."""

from .plotting.common import _get_default_colors, plot_over_time
from .utils.misc import get_pop_kwargs, to_bits
from .utils.stats import corr, report_CI

__all__ = [
    "corr",
    "get_pop_kwargs",
    "report_CI",
    "to_bits",
    "_get_default_colors",
    "plot_over_time",
]

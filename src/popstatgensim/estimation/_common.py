"""Shared internal helpers re-exported from the legacy REML module."""

from ..reml import (
    _PreparedInputs,
    _as_optional_matrix_list,
    _build_v,
    _compute_beta_and_py,
    _compute_loglik,
    _ensure_factorable_theta,
    _factor_theta,
    _prepare_inputs,
    _print_iter_message,
    _resolve_random_effect_inputs,
)

__all__ = [
    "_PreparedInputs",
    "_as_optional_matrix_list",
    "_resolve_random_effect_inputs",
    "_prepare_inputs",
    "_build_v",
    "_factor_theta",
    "_ensure_factorable_theta",
    "_compute_beta_and_py",
    "_compute_loglik",
    "_print_iter_message",
]

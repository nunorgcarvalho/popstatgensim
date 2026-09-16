"""Pedigree, relations, and IBD utilities."""

from .ibd import (
    IBDSegment,
    IBD_tensor_to_segments,
    compute_K_IBD,
    extract_all_crossover_points,
    extract_crossover_points,
    get_coeff_inbreeding,
    get_coeff_kinship,
    get_coeff_relatedness,
    get_true_IBD1,
    get_true_IBD_arr,
    get_true_IBD_tensor,
    greedy_unrelated_subset,
)
from .graph import PedigreeGraph
from .paths import (
    LinealChain,
    MatingChain,
    PathSignature,
    chain_weight,
    implied_covariance,
    predicted_corr,
    predicted_cov,
    proband_correction,
)
from .pedigree import PedKey2, Pedigree  # unpickling shim for the removed layer
from .relate import (
    KNOWN_SIGNATURES,
    VALID_CLASS_SPECS,
    RelateResult,
    enumerate_class,
    relate,
    signature_name,
)
from .relations import get_relation_matrix, initialize_relations

__all__ = [
    "PedigreeGraph",
    "PathSignature",
    "LinealChain",
    "MatingChain",
    "chain_weight",
    "implied_covariance",
    "proband_correction",
    "predicted_cov",
    "predicted_corr",
    "relate",
    "RelateResult",
    "enumerate_class",
    "signature_name",
    "KNOWN_SIGNATURES",
    "VALID_CLASS_SPECS",
    "PedKey2",
    "Pedigree",
    "initialize_relations",
    "get_relation_matrix",
    "IBDSegment",
    "get_true_IBD1",
    "get_true_IBD_tensor",
    "get_true_IBD_arr",
    "get_coeff_kinship",
    "get_coeff_inbreeding",
    "get_coeff_relatedness",
    "compute_K_IBD",
    "greedy_unrelated_subset",
    "IBD_tensor_to_segments",
    "extract_crossover_points",
    "extract_all_crossover_points",
]

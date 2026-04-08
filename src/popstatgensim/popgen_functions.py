"""Compatibility wrapper for genetics, genome, and pedigree helpers."""

from .genetics.frequencies import draw_p_FST, get_FST, get_fixation_t, summarize_ps
from .genetics.genotypes import center_G, compute_freqs, compute_GRM, make_G, standardize_G
from .genetics.ld import compute_LD_matrix, compute_corr_matrix, make_neighbor_matrix
from .genetics.pca import (
    PCAResult,
    _format_pc_axis_label,
    _orient_pca_scores,
    _validate_pca_axes,
    compute_PCA,
)
from .genome.structure import draw_binom_haplos, draw_p_init, generate_LD_blocks, generate_chromosomes
from .pedigree.ibd import (
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
from .pedigree.relations import get_relation_matrix, initialize_relations
from .plotting.genetics import plot_PCA, plot_site_frequency_spectrum

__all__ = [
    "PCAResult",
    "IBDSegment",
    "make_G",
    "compute_freqs",
    "center_G",
    "standardize_G",
    "compute_GRM",
    "plot_site_frequency_spectrum",
    "_validate_pca_axes",
    "_orient_pca_scores",
    "_format_pc_axis_label",
    "compute_PCA",
    "plot_PCA",
    "make_neighbor_matrix",
    "compute_corr_matrix",
    "compute_LD_matrix",
    "draw_binom_haplos",
    "generate_LD_blocks",
    "generate_chromosomes",
    "draw_p_init",
    "draw_p_FST",
    "get_FST",
    "get_fixation_t",
    "summarize_ps",
    "initialize_relations",
    "get_relation_matrix",
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

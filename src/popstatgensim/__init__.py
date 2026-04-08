"""Public package interface for popstatgensim."""

from . import estimation, genome, io, pedigree, plotting, simulation, traits, utils
from .estimation import run_HE_regression, run_HEreg, run_REML
from .genome import PCAResult
from .io import export_GRM_GCTA, export_trait
from .simulation import Population, SuperPopulation
from .traits import CorrelatedRandomEffect, FixedEffect, GeneticEffect, NoiseEffect, Trait

__all__ = [
    "Population",
    "SuperPopulation",
    "Trait",
    "GeneticEffect",
    "FixedEffect",
    "CorrelatedRandomEffect",
    "NoiseEffect",
    "PCAResult",
    "run_HEreg",
    "run_HE_regression",
    "run_REML",
    "export_GRM_GCTA",
    "export_trait",
    "genome",
    "pedigree",
    "traits",
    "simulation",
    "estimation",
    "plotting",
    "io",
    "utils",
]

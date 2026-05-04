"""Public package interface for popstatgensim."""

from . import estimation, genome, io, pedigree, plotting, simulation, traits, utils
from .estimation import run_EO_AM, run_HEreg, run_REML
from .genome import PCAResult
from .io import export_GRM_GCTA, export_trait, read_GRM_GCTA
from .simulation import Population, PopulationParams, SuperPopulation
from .traits import FixedEffect, GeneticEffect, NoiseEffect, RandomEffect, Trait

__all__ = [
    "Population",
    "PopulationParams",
    "SuperPopulation",
    "Trait",
    "GeneticEffect",
    "FixedEffect",
    "RandomEffect",
    "NoiseEffect",
    "PCAResult",
    "run_EO_AM",
    "run_HEreg",
    "run_REML",
    "export_GRM_GCTA",
    "read_GRM_GCTA",
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

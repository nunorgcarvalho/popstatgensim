"""Compatibility wrapper for classes moved into simulation, traits, and pedigree subpackages."""

from .pedigree.pedigree import PedKey2, PedPath, PathSig, Pedigree, RelObj
from .simulation.population import Population
from .simulation.superpopulation import SuperPopulation
from .traits.effects import CorrelatedRandomEffect, Effect, FixedEffect, GeneticEffect, NoiseEffect
from .traits.trait import Trait

__all__ = [
    "Population",
    "Effect",
    "GeneticEffect",
    "FixedEffect",
    "CorrelatedRandomEffect",
    "NoiseEffect",
    "Trait",
    "SuperPopulation",
    "PedPath",
    "PedKey2",
    "PathSig",
    "RelObj",
    "Pedigree",
]

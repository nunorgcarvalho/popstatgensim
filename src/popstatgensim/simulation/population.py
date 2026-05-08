"""Population class for forward simulation workflows."""

from __future__ import annotations

import copy
import inspect
import warnings
from dataclasses import dataclass, fields
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from ..genome import frequencies as genome_frequencies
from ..genome import genotypes as genome_genotypes
from ..genome import ld as genome_ld
from ..genome import pca as genome_pca
from ..genome import structure as genome_structure
from ..pedigree import ibd as pedigree_ibd
from ..pedigree import relations as pedigree_relations
from ..pedigree.pedigree import Pedigree
from ..plotting import common as common_plotting
from ..plotting import genome as genome_plotting
from ..traits import effect_sampling as trait_sampling
from ..traits.trait import Trait
from ..utils import matrix_metrics as utils_matrix_metrics
from ..utils import misc as misc_utils
from ..utils import stats as stats_utils


_PARAM_UNSET = object()


@dataclass
class PopulationParams:
    '''
    Parameters controlling how a Population evolves in future simulations.
    '''
    R_type: str = 'indep'
    R: Optional[np.ndarray] = None
    related_offspring: bool = True
    s: Union[float, np.ndarray] = 0.0
    mu: Union[float, np.ndarray] = 0.0
    n_offspring_dist: str = 'constant'
    AM_r: float = 0.0
    AM_trait: Union[str, np.ndarray, None] = None
    AM_type: str = 'phenotypic'
    keep_past_generations: int = 1
    track_pedigree: bool = False
    track_haplotypes: bool = False
    metric_retention: str = 'store_every'
    metric_last_k: Optional[int] = None
    trait_updates: bool = True


class Population:
    '''
    Class for a population to simulate. Contains genotype information. Contains methods to simulate change in population over time.
    '''

    ########################
    #### Initialization ####
    ########################

    def __init__(self, N: int, M: int, P: int = 2,
                 p_init: Union[float, np.ndarray] = None,
                 params: Union[PopulationParams, dict, None] = None,
                 seed: int = None):
        '''
        Initializes a population, simulating initial genotypes from specified allele frequencies.
        Parameters:
            N (int): Population size of individuals (not haplotypes).
            M (int): Total number of variants in genome.
            P (int): Ploidy of genotpes. Default is 2 (diploid).
            p_init (float or array): Initial allele frequency of variants. If only a single value is provided, it is treated as the initial allele frequency for all variants. Alternatively, can be an array of length M for variant-specfic allele frequencies. If not provided, default is uniform distribution of allele frequencies between 0.05 and 0.95.
            params (PopulationParams or dict): Optional parameter object used to
                initialize simulation, tracking, and storage settings.
            seed (int): Initial seed to use when simulating genotypes (and allele frequencies if necessary).
        '''
        # sets seed if specified
        if seed is not None:
            seed = np.random.seed(seed)
        
        # draws initial allele frequencies from uniform distribution between 0.05 and 0.95 if not specified
        if p_init is None:
            p_init = genome_structure.draw_p_init(M, method = 'uniform', params = (0.05, 0.95))
        elif type(p_init) == float or type(p_init) == int:
            # if only single value is given, all variants have same initial frequency
            p_init = np.full(M, p_init)

        # generates initial genotypes and records allele frequencies
        H = genome_structure.draw_binom_haplos(p_init, N, P)

        # passes haplotype to base constructor for further initialization
        self._initialize_H(
            H,
            params=params,
        )
    
    @classmethod
    def from_H(cls, H: np.ndarray,
               params: Union[PopulationParams, dict, None] = None):
        '''
        Initializes a population from a given haplotype array.
        Parameters:
            H (3D array): N*M*P array of haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
            params (PopulationParams or dict): Optional parameter object used to
                initialize simulation, tracking, and storage settings.
        Returns:
            Population: A new Population object initialized with the given haplotype array.
        '''
        # creates new instance of class
        pop = cls.__new__(cls)
        # initializes the object with the given haplotype array
        pop._initialize_H(
            H,
            params=params,
        )
        return pop

    def _initialize_H(self, H: np.ndarray,
                      params: Union[PopulationParams, dict, None] = None):
        '''
        Initializes a population from a given haplotype array. See the `from_H` class method for details.
        '''
        # sets basic population attributes from haplotype array
        (self.N, self.M, self.P) = H.shape
        self.params = PopulationParams()

        # initializes default/initial attributes
        self.t = 0 # generation
        self.T_breaks = [self.t] # simulation breaks
        self.traits = {}
        self.BPs = np.arange(self.M) # variant positions in base pairs (BPs)
        self.G_par = None
        self._X = None
        self._G_std = None
        self._G_par_std = None
        self._GRM_cache = {}
        self._DGRM_cache = {}
        self._X_dtype = np.float32
        self.params.R = self._generate_R_from_type(self.params.R_type)
        self.K = np.diag(np.ones(self.N)) # kinship matrix (initially identity, not functional yet)
        # how many past generations to keep in memory
        self.past = [self]
        self.update_keep_past_gens(keep_past_generations=self.params.keep_past_generations)
        if params is not None:
            self.set_params(params=params)
        Haplos = None
        if self.params.track_haplotypes:
            Haplos = np.full(H.shape, -1, dtype=np.int32)

        # further attributes
        self._update_obj(H=H, Haplos=Haplos, update_past=False)
        self.assign_sex() # assigns sex (F:0 / M:1)
        self.relations = pedigree_relations.initialize_relations(self.N)
        self.ped = Pedigree(self.N)

        # defines metrics
        self.metric = {}
        self._define_metric('p', genome_frequencies.compute_freqs, shape = [self.M], # allele frequency
                            P = self.P)
        self._initialize_metrics(G = self.G)
    
    def update_keep_past_gens(self, keep_past_generations: int):
        '''
        Small function for structuring the 'past' attribute to reflect the number of generations in the past that are stored in the current population object. If the number of past generations is reduced, the older generations are removed from the array. If the number if past generations is increased, 'past' array is padded with Nones for future compatibility.
        Parameters:
            keep_past_generations (int): Number of past generations to keep in the object.
        '''
        keep_past_generations = int(keep_past_generations)
        if keep_past_generations < 0:
            raise ValueError('`keep_past_generations` must be non-negative.')
        if self.params.track_pedigree and keep_past_generations < 1:
            raise ValueError('Must keep at least 1 past generation to track pedigree.')

        current_past_gens = len(self.past) - 1 # should also just be self.keep_past_generations
        if keep_past_generations < current_past_gens:
            self.past = self.past[0:keep_past_generations + 1]
        elif keep_past_generations > current_past_gens:
            for _ in range(current_past_gens, keep_past_generations):
                self.past.append(None) # initializes past generations' objects as None
        self.params.keep_past_generations = keep_past_generations

    @property
    def keep_past_generations(self) -> int:
        return int(self.params.keep_past_generations)

    @keep_past_generations.setter
    def keep_past_generations(self, value: int):
        self.set_params(keep_past_generations=value)

    @property
    def track_pedigree(self) -> bool:
        return bool(self.params.track_pedigree)

    @track_pedigree.setter
    def track_pedigree(self, value: bool):
        self.set_params(track_pedigree=value)

    @property
    def track_haplotypes(self) -> bool:
        return bool(self.params.track_haplotypes)

    @track_haplotypes.setter
    def track_haplotypes(self, value: bool):
        self.set_params(track_haplotypes=value)

    @property
    def metric_retention(self) -> str:
        return str(self.params.metric_retention)

    @metric_retention.setter
    def metric_retention(self, value: str):
        self.set_params(metric_retention=value)

    @property
    def metric_last_k(self) -> Optional[int]:
        return self.params.metric_last_k

    @metric_last_k.setter
    def metric_last_k(self, value: Optional[int]):
        self.set_params(metric_last_k=value)

    @property
    def trait_updates(self) -> bool:
        return bool(self.params.trait_updates)

    @trait_updates.setter
    def trait_updates(self, value: bool):
        self.set_params(trait_updates=value)

    def _normalize_metric_retention(self, metric_retention: str, metric_last_k: Optional[int]) -> tuple[str, Optional[int]]:
        metric_retention = str(metric_retention).strip().lower()
        valid = {'store_every', 'store_last_k', 'summary_only', 'disabled'}
        if metric_retention not in valid:
            raise ValueError(f"Unknown metric retention policy: {metric_retention}")
        if metric_last_k is not None:
            metric_last_k = int(metric_last_k)
            if metric_last_k <= 0:
                raise ValueError('`metric_last_k` must be a positive integer when provided.')
        if metric_retention == 'store_last_k' and (metric_last_k is None or metric_last_k <= 0):
            raise ValueError("metric_last_k must be a positive integer when metric_retention='store_last_k'.")
        return metric_retention, metric_last_k


    def _validate_R(self, R: Union[float, np.ndarray]) -> np.ndarray:
        '''
        Validates and returns a recombination-rate vector matching Population.M.
        '''
        if isinstance(R, (float, int, np.floating, np.integer)):
            R = np.full(self.M, float(R), dtype=float)
        else:
            R = np.asarray(R, dtype=float)
        if R.ndim != 1:
            raise ValueError('Population parameter `R` must be a 1D array or scalar.')
        if R.shape[0] != self.M:
            raise ValueError('Population parameter `R` must have length M.')
        if not np.all(np.isfinite(R)):
            raise ValueError('Population parameter `R` must contain finite values.')
        if np.any(R < 0.0):
            raise ValueError('Population parameter `R` must contain non-negative values.')
        return R.copy()

    def _generate_R_from_type(self, R_type: str) -> np.ndarray:
        return self._validate_R(genome_structure.generate_recombination_rates(self.M, R_type=R_type))

    def _copy_param_value(self, value):
        return value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)

    def _set_param_value(self, params: PopulationParams, name: str, value,
                         update_r_type_for_R: bool = True):
        if name == 'R_type':
            R_type = str(value).strip().lower()
            if R_type == 'custom':
                if params.R is None:
                    raise ValueError("Population parameter `R_type='custom'` requires `R` to be set.")
                params.R_type = R_type
                return
            R = self._generate_R_from_type(R_type)
            params.R_type = R_type
            params.R = R
            self._DGRM_cache = {}
        elif name == 'R':
            if value is None:
                if self.params.R is None:
                    raise ValueError('Population parameter `R` cannot be None.')
                params.R = self.params.R.copy()
                return
            params.R = self._validate_R(value)
            if update_r_type_for_R:
                params.R_type = 'custom'
            self._DGRM_cache = {}
        elif name == 'related_offspring':
            params.related_offspring = bool(value)
        elif name == 'AM_r':
            value = float(value)
            if abs(value) > 1:
                raise ValueError('Population parameter `AM_r` must be between -1 and 1.')
            params.AM_r = value
        elif name == 'n_offspring_dist':
            params.n_offspring_dist = str(value).strip().lower()
        elif name == 'AM_type':
            params.AM_type = str(value)
        elif name == 'keep_past_generations':
            keep_past_generations = int(value)
            if keep_past_generations < 0:
                raise ValueError('`keep_past_generations` must be non-negative.')
            if params.track_pedigree and keep_past_generations < 1:
                raise ValueError('Must keep at least 1 past generation to track pedigree.')
            if params is self.params:
                self.update_keep_past_gens(keep_past_generations)
            else:
                params.keep_past_generations = keep_past_generations
        elif name == 'track_pedigree':
            track_pedigree = bool(value)
            if track_pedigree and params.keep_past_generations < 1:
                raise ValueError('Must keep at least 1 past generation to track pedigree.')
            params.track_pedigree = track_pedigree
        elif name == 'track_haplotypes':
            params.track_haplotypes = bool(value)
            if params is self.params:
                if params.track_haplotypes and hasattr(self, 'H') and getattr(self, 'Haplos', None) is None:
                    self.Haplos = np.full(self.H.shape, -1, dtype=np.int32)
                elif not params.track_haplotypes:
                    self.Haplos = None
        elif name == 'metric_retention':
            metric_retention, metric_last_k = self._normalize_metric_retention(
                value,
                params.metric_last_k,
            )
            params.metric_retention = metric_retention
            params.metric_last_k = metric_last_k
        elif name == 'metric_last_k':
            metric_retention, metric_last_k = self._normalize_metric_retention(
                params.metric_retention,
                value,
            )
            params.metric_retention = metric_retention
            params.metric_last_k = metric_last_k
        elif name == 'trait_updates':
            params.trait_updates = bool(value)
        elif name in {'s', 'mu', 'AM_trait'}:
            setattr(params, name, self._copy_param_value(value))
        else:
            raise ValueError(f'Unknown population parameter: {name}.')

    def _apply_param_updates(self, target: PopulationParams, updates: dict,
                             update_r_type_for_R: bool = True):
        field_order = [
            'R_type',
            'R',
            'related_offspring',
            's',
            'mu',
            'n_offspring_dist',
            'AM_r',
            'AM_trait',
            'AM_type',
            'keep_past_generations',
            'track_pedigree',
            'track_haplotypes',
            'metric_last_k',
            'metric_retention',
            'trait_updates',
        ]
        for name in field_order:
            if name in updates:
                self._set_param_value(
                    target,
                    name,
                    updates[name],
                    update_r_type_for_R=update_r_type_for_R,
                )

    def set_params(self, params: Union[PopulationParams, dict] = None, return_params: bool = False,
                   **kwargs) -> PopulationParams:
        '''
        Permanently updates parameters controlling future population simulations.
        Parameters:
            params (PopulationParams or dict): Optional parameter object or dictionary
                whose values are applied before keyword arguments.
            return_params (bool): Whether to return the updated parameter object after applying the updates. Default is False.
            **kwargs: Population parameter values to update.
        Returns (if `return_params=True`):
            PopulationParams: The updated parameter object stored on this Population.
        '''
        if params is not None:
            if isinstance(params, PopulationParams):
                self._apply_param_updates(
                    self.params,
                    {
                        field.name: self._copy_param_value(getattr(params, field.name))
                        for field in fields(PopulationParams)
                    },
                    update_r_type_for_R=False,
                )
            elif isinstance(params, dict):
                self._apply_param_updates(self.params, dict(params))
            else:
                raise ValueError('`params` must be a PopulationParams object or dictionary.')

        valid_names = {field.name for field in fields(PopulationParams)}
        unknown = sorted(name for name in kwargs if name not in valid_names)
        if unknown:
            raise ValueError(f"Unknown population parameter: {unknown[0]}.")
        self._apply_param_updates(self.params, kwargs)
        if return_params:
            return self.params

    def _resolved_params(self, params: Union[PopulationParams, dict] = None,
                         overrides: dict = None) -> PopulationParams:
        resolved = copy.deepcopy(self.params)
        if params is not None:
            if isinstance(params, PopulationParams):
                self._apply_param_updates(
                    resolved,
                    {
                        field.name: self._copy_param_value(getattr(params, field.name))
                        for field in fields(PopulationParams)
                    },
                    update_r_type_for_R=False,
                )
            elif isinstance(params, dict):
                self._apply_param_updates(resolved, dict(params))
            else:
                raise ValueError('`params` must be a PopulationParams object or dictionary.')
        if overrides:
            self._apply_param_updates(resolved, overrides)
        return resolved

    def _params_equal(self, other_params: PopulationParams) -> bool:
        '''
        Returns whether another PopulationParams object matches this population's params.
        '''
        for field in fields(PopulationParams):
            value_self = getattr(self.params, field.name)
            value_other = getattr(other_params, field.name)
            if isinstance(value_self, np.ndarray) or isinstance(value_other, np.ndarray):
                if not np.array_equal(np.asarray(value_self), np.asarray(value_other)):
                    return False
            elif value_self != value_other:
                return False
        return True

    def set_founding_haplotypes(self):
        '''
        Generates a complementary haplotype array for each individual containing a haplotype identifier for each allele. This functions treats the current generation as founders (individuals are unrelated from each other) such that each of their chromosomes has a unique identifier for all alleles in it. Subsequent generations can then track the inheritance of these founding haplotypes. Haplotypes are given an integer in the order they appear in the haplotype array.
        '''
        self.params.track_haplotypes = True
        ids = np.arange(self.N * self.P, dtype=np.int32).reshape(self.N, self.P)
        Haplos = np.broadcast_to(ids[:, None, :], self.H.shape).copy()
        self._update_obj(Haplos=Haplos)

    @property
    def X(self) -> np.ndarray:
        '''
        Lazily computes and caches the standardized genotype matrix.
        '''
        if self._X is None:
            X = genome_genotypes.standardize_G(self.G, self.p, self.P, impute=True, std_method='observed')
            self._X = np.asarray(X, dtype=self._X_dtype)
        return self._X

    @X.setter
    def X(self, value: np.ndarray):
        if value is None:
            self._X = None
        else:
            self._X = np.asarray(value, dtype=self._X_dtype)

    def _require_haplotype_tracking(self):
        if not self.params.track_haplotypes or self.Haplos is None:
            raise ValueError(
                'Haplotype IDs are not stored for this population. '
                'Initialize with track_haplotypes=True or call set_founding_haplotypes().'
            )

    def get_relation_matrix(self, relation: str, dtype: np.dtype = np.uint8) -> np.ndarray:
        '''
        Returns a dense matrix representation of one stored relation.
        '''
        return pedigree_relations.get_relation_matrix(self.relations, relation, self.N, dtype=dtype)

    def _get_full_sibships(self) -> list[np.ndarray]:
        '''
        Returns lists of row indices for each full-sib family in the current generation.

        Individuals with missing `full_sibs` labels are treated as singleton families.
        '''
        family_ids = np.asarray(self.relations['full_sibs'], dtype=np.int32)
        if family_ids.ndim != 1 or family_ids.shape[0] != self.N:
            raise ValueError("Compact 'full_sibs' relation must have shape (N,).")

        sibships = []
        valid_mask = family_ids >= 0
        if np.any(valid_mask):
            for family_id in np.unique(family_ids[valid_mask]):
                sibships.append(np.flatnonzero(family_ids == family_id).astype(np.int32, copy=False))

        singleton_idx = np.flatnonzero(~valid_mask)
        for idx in singleton_idx:
            sibships.append(np.array([idx], dtype=np.int32))

        return sibships

    def _get_full_sibship_sizes(self) -> np.ndarray:
        '''
        Returns the size of each individual's full-sib family, including the individual.
        '''
        n_sibs = np.ones(self.N, dtype=np.int32)
        for sibship in self._get_full_sibships():
            n_sibs[sibship] = sibship.size
        return n_sibs

    def get_G_std(self) -> np.ndarray:
        '''
        Returns cached genotype standard deviations for the current generation.
        '''
        if self._G_std is None:
            self._G_std = np.asarray(trait_sampling.get_G_std_for_effects(self.G, P=self.P), dtype=np.float32)
        return self._G_std

    def get_Gpar_std(self, G_par: np.ndarray = None) -> np.ndarray:
        '''
        Returns cached parental-genotype standard deviations for the current generation.
        '''
        if G_par is None:
            if self._G_par_std is None:
                G_par = self.get_Gpar()
                self._G_par_std = np.asarray(
                    trait_sampling.get_G_std_for_effects(G_par, P=2 * self.P),
                    dtype=np.float32,
                )
            return self._G_par_std
        return np.asarray(trait_sampling.get_G_std_for_effects(G_par, P=2 * self.P), dtype=np.float32)

    def _get_standardized_genotypes(self, std_method: str = 'observed') -> np.ndarray:
        '''
        Returns a standardized genotype matrix for the current generation.

        The default ``std_method='observed'`` reuses ``Population.X``. Other
        standardization methods are computed on demand.
        '''
        std_method = str(std_method).strip().lower()
        if std_method == 'observed':
            return np.asarray(self.X, dtype=float)
        if std_method == 'binomial':
            return np.asarray(
                genome_genotypes.standardize_G(
                    self.G,
                    self.p,
                    self.P,
                    impute=True,
                    std_method=std_method,
                ),
                dtype=float,
            )
        raise ValueError("`std_method` must be either 'observed' or 'binomial'.")

    def get_GRM(self, std_method: str = 'observed') -> np.ndarray:
        '''
        Returns the SNP genomic relationship matrix (GRM) for the current population.

        The GRM is cached separately for each supported ``std_method`` so repeated
        calls do not recompute it. With the default ``std_method='observed'``, this
        wraps ``genome.compute_GRM()`` applied to ``Population.X``.
        '''
        std_method = str(std_method).strip().lower()
        if std_method not in self._GRM_cache:
            X = self._get_standardized_genotypes(std_method=std_method)
            self._GRM_cache[std_method] = np.asarray(
                genome_genotypes.compute_GRM(X),
                dtype=float,
            )
        return self._GRM_cache[std_method]

    def get_DGRM(self, method: str = 'genome_wide',
                 std_method: str = 'binomial') -> np.ndarray:
        '''
        Returns the assortative-mating disequilibrium GRM (DGRM) for the current
        population.

        Parameters:
            method (str): DGRM construction method. Supported values are:
                - ``'genome_wide'``: computes
                  ``D = C^{-1} (G^2 - (N / M) G)`` using the current analyzed
                  sample.
                - ``'between_chroms'``: reproduces the between-chromosome DGRM
                  construction in the authors' ``makeDGRM.cpp`` by accumulating
                  chromosome-specific ``zz_k`` blocks and retaining only the
                  cross-chromosome component.
            std_method (str): Genotype standardization method used to build the
                underlying GRM. This defaults to ``'binomial'`` rather than
                ``'observed'`` because assortative mating inflates observed
                per-SNP genotype variance, and the binomial scaling matches the
                genome-wide Zhang et al. DGREML implementation.
        Returns:
            DGRM (2D array): N*N disequilibrium relationship matrix.
        '''
        method = str(method).strip().lower()
        std_method = str(std_method).strip().lower()
        cache_key = (method, std_method)
        if cache_key in self._DGRM_cache:
            return self._DGRM_cache[cache_key]

        G = self.get_GRM(std_method=std_method)
        n = self.N
        M = self.G.shape[1]

        if method == 'genome_wide':
            H = G @ G - (n / M) * G
            C = np.trace(H) / n
            if np.isclose(C, 0.0):
                raise ValueError('DGRM normalization constant is zero or numerically unstable.')
            D = np.asarray(H / C, dtype=float)
        elif method == 'between_chroms':
            chrom_idx = self.get_chrom_idx()
            if chrom_idx.size < 2:
                raise ValueError(
                    "The between-chromosome DGRM requires at least two chromosomes. "
                    "Update `Population.params.R` so `get_chrom_idx()` identifies multiple chromosomes."
                )

            chrom_end = np.concatenate([chrom_idx[1:], np.array([M], dtype=int)])
            X = self._get_standardized_genotypes(std_method=std_method)
            zz_sum = G * M
            W = np.zeros_like(G, dtype=float)
            C = 0.0

            for start, stop in zip(chrom_idx, chrom_end):
                M_k = stop - start
                if M_k <= 0:
                    continue
                zz_k = X[:, start:stop] @ X[:, start:stop].T
                W += zz_k @ zz_k
                pi_k = M_k / M
                C += pi_k * pi_k

            B = (zz_sum @ zz_sum - W) / (n * M * M)
            B_bar = np.trace(B) / n
            if np.isclose(B_bar, 0.0):
                raise ValueError(
                    'Between-chromosome DGRM normalization is zero or numerically unstable.'
                )
            D = np.asarray(((1.0 - C) / B_bar) * B, dtype=float)
        else:
            raise ValueError("`method` must be either 'genome_wide' or 'between_chroms'.")

        D = 0.5 * (D + D.T)
        self._DGRM_cache[cache_key] = D
        return D

    def get_chrom_idx(self) -> np.ndarray:
        '''
        Returns ascending chromosome start indices inferred from recombination rates.

        Chromosome starts are defined as variant positions where ``Population.params.R``
        is exactly ``0.5``, with index 0 always included.
        '''
        R = np.asarray(self.params.R, dtype=float)
        if R.ndim != 1:
            raise ValueError('Population.params.R must be a 1D array.')
        if R.shape[0] != self.M:
            raise ValueError('Population.params.R must have length M.')

        chrom_idx = np.flatnonzero(R == 0.5)
        chrom_idx = np.unique(np.concatenate([np.array([0], dtype=int), chrom_idx]))
        return np.sort(chrom_idx)


    ###################################
    #### Storing object attributes ####
    ###################################

    def _update_obj(self, H: np.ndarray = None, update_past: bool = True,
                    relations: dict = None, Haplos: np.ndarray = None,
                    update_pedigree: bool = False):
        '''
        Update the population object's attributes.
        Parameters:
            H (3D array): Haplotype array.
            update_past (bool): Whether to update the memory of past generations. Default is True.
            relations (dict): Dictionary of relationship matrices to update. If None, does not update relationships. Default is None.
            Haplos (3D array): Haplotype identifier array. If None, does not update haplotype identifiers. Default is None.
            update_pedigree (bool): Whether to update pedigree information. Must also update past generations. Default is False.
        '''
        # keep this before updating anything else
        if update_past:
            self._update_past()
        
        if H is not None:
            self.H = H
            self.G = genome_genotypes.make_G(self.H)
            self.G_par = None
            self._G_std = None
            self._G_par_std = None
            self._GRM_cache = {}
            self._DGRM_cache = {}
            self.p = genome_frequencies.compute_freqs(self.G, self.P)
            self.X = None
        if Haplos is not None:
            self.Haplos = Haplos
        elif not self.track_haplotypes:
            self.Haplos = None
        if relations is not None:
            self._update_relations(relations)
        if update_pedigree and update_past:
            # this must happen after updating past generations and their relations
            self._update_pedigree()

    def _update_past(self):
        '''
        Updates the past generations' objects, shifting them by one and adding the current object as the first element.
        '''
        if self.keep_past_generations > 0:
            # makes copy of current object
            gen_t1 = copy.deepcopy(self)
            # removes unnecessary attributes (e.g. past generations)
            gen_t1.past = None
            # shifts previous generations
            self.past[2:] = self.past[1:self.keep_past_generations]
            self.past[1] = gen_t1
            self.past[0] = self
        self.t += 1
    
    def _update_relations(self, relations: dict):
        '''
        Updates the relationships in the population object.
        Parameters:
            relations (dict): Dictionary of relationship matrices to update. Keys are relationship types (e.g. 'spouses', 'parents') and values are the corresponding matrices.
        '''
        for key in relations:
            if key == 'spouses':
                if self.keep_past_generations >= 1:
                    self.past[1].relations['spouses'] = relations['spouses'] # updates prior spouse matrix for past gen
                self.relations['spouses'] = np.full(self.N, -1, dtype=np.int32)
            else:
                self.relations[key] = relations[key] # sets other relationship matrices for current generation

    def _update_pedigree(self):
        '''
        Updates the pedigree information in the population object for a new generation.
        '''
        if self.keep_past_generations < 1:
            raise Exception('Must keep at least 1 past generation to track pedigree.')
        # makes new Pedigree object from scratch, and adds parent infices and Pedigree pointer
        self.ped = Pedigree(self.N, par_idx=self.relations['parents'], par_Ped=self.past[1].ped)
        # fills out the relationship matrix
        self.ped.construct_paths()

    def store_neighbor_matrix(self, LDwindow: float = None) -> sparse.coo_matrix:
        '''
        Stores a boolean sparse matrix of variants (`neighbor_matrix`) within the specified LD window distance (`LDwindow`) based on the object's variant positions (currently only supports `BPs`).
        Parameters:
            LDwindow (float): Maximum distance between variants to be considered neighbors. In the same units as the positions used. If not provided, defaults to infinite maximum distance.
        '''
        # by default, uses object's base pair positions
        positions = self.BPs
        self.neighbor_matrix = genome_ld.make_neighbor_matrix(positions=positions, LDwindow=LDwindow)

    def store_LD_matrix(self):
        '''
        Stores the correlation matrix (`corr_matrix`) and its square, the LD matrix (`LD_matrix`). Must have pre-computed `neighbor_matrix` attribute (see `store_neighbor_matrix()`).

        Parameters:
            corr_matrix (sparse 2D matrix (CSR)): M*M correlation matrix between pairs of variants in CSR format. Can be computed with `get_corr_matrix()`.
        Returns:
            LD_matrix (sparse 2D matrix (CSR)): M*M LD matrix (square of correlation) between pairs of variants in CSR format.
        '''
        if not hasattr(self, 'neighbor_matrix'):
            raise Exception('Must have pre-computed neighbor_matrix. Use `Population.store_neighbor_matrix()`.')
        self.corr_matrix = genome_ld.compute_corr_matrix(self.X, self.neighbor_matrix)
        self.LD_matrix = genome_ld.compute_LD_matrix(self.corr_matrix)

    def get_relatedness_matrix(self, source: str = 'GRM',
                               standardize_ibd: bool = False,
                               std_method: str = 'observed') -> np.ndarray:
        '''
        Returns a pairwise relatedness matrix for the current population.
        Parameters:
            source (str): Which relatedness measure to use. Options are:
                - 'GRM' (default): SNP-based genomic relationship matrix from `Population.X`.
                - 'IBD': True relatedness from tracked haplotype IDs.
            standardize_ibd (bool): Passed to the true-IBD relatedness computation when
                `source='IBD'`. Default is False.
            std_method (str): Genotype standardization method used when
                `source='GRM'`. Passed to `Population.get_GRM()`. Default is
                `'observed'`.
        Returns:
            relatedness (2D array): N*N relatedness matrix.
        '''
        source = source.upper()
        if source == 'GRM':
            return np.asarray(self.get_GRM(std_method=std_method), dtype=float)
        if source == 'IBD':
            return np.asarray(
                pedigree_ibd.compute_K_IBD(self.Haplos, standardize=standardize_ibd),
                dtype=float,
            )
        raise ValueError("source must be either 'GRM' or 'IBD'.")

    def subset_individuals(self, i_keep: np.ndarray,
                           keep_past_generations: int = None) -> 'Population':
        '''
        Returns a new Population object containing only the specified individuals.
        Parameters:
            i_keep (1D int or bool array): Individuals to retain. Boolean masks must
                have length `Population.N`.
            keep_past_generations (int): Optional override for the number of past
                generations to preserve. If None, uses
                `Population.params.keep_past_generations`. Parent generations are
                recursively subset to the ancestors referenced by the retained
                individuals when available.
        Returns:
            Population: Subsetted population.
        '''
        i_keep = np.asarray(i_keep)
        if i_keep.ndim != 1:
            raise ValueError('`i_keep` must be a 1D index array or boolean mask.')
        if i_keep.dtype == bool:
            if i_keep.shape[0] != self.N:
                raise ValueError('Boolean `i_keep` mask must have length Population.N.')
            i_keep = np.flatnonzero(i_keep)
        else:
            i_keep = i_keep.astype(np.int64, copy=False)

        if i_keep.size == 0:
            raise ValueError('Must retain at least one individual.')
        if np.any(i_keep < 0) or np.any(i_keep >= self.N):
            raise IndexError('`i_keep` contains indices outside the population.')
        if np.unique(i_keep).size != i_keep.size:
            raise ValueError('`i_keep` cannot contain duplicate indices.')
        if keep_past_generations is None:
            keep_past_generations = self.keep_past_generations

        H_new = self.H[i_keep, :, :]
        params_new = copy.deepcopy(self.params)
        params_new.keep_past_generations = int(keep_past_generations)
        if params_new.keep_past_generations < 1:
            params_new.track_pedigree = False
        new_pop = Population.from_H(
            H_new,
            params=params_new,
        )

        new_pop.BPs = self.BPs.copy()
        new_pop.t = self.t
        new_pop.T_breaks = copy.deepcopy(self.T_breaks)
        new_pop.K = self.K[np.ix_(i_keep, i_keep)].copy()

        if self.track_haplotypes and self.Haplos is not None:
            new_pop.Haplos = self.Haplos[i_keep, :, :].copy()
        if self._X is not None:
            new_pop.X = self._X[i_keep, :].copy()
        if self._GRM_cache:
            new_pop._GRM_cache = {
                key: value[np.ix_(i_keep, i_keep)].copy()
                for key, value in self._GRM_cache.items()
            }
        if self._DGRM_cache:
            new_pop._DGRM_cache = {
                key: value[np.ix_(i_keep, i_keep)].copy()
                for key, value in self._DGRM_cache.items()
            }

        idx_map = np.full(self.N, -1, dtype=np.int32)
        idx_map[i_keep] = np.arange(i_keep.size, dtype=np.int32)

        parent_map = None
        parent_source = self.relations.get('parent_source', 'past')
        source_parents = np.asarray(self.relations['parents'], dtype=np.int32)
        if (keep_past_generations > 0 and parent_source == 'past'
                and self.past is not None and len(self.past) > 1
                and self.past[1] is not None):
            parent_keep = np.unique(source_parents[i_keep][source_parents[i_keep] >= 0])
            if parent_keep.size > 0:
                new_parent = self.past[1].subset_individuals(
                    parent_keep,
                    keep_past_generations=keep_past_generations - 1,
                )
                new_pop.past[1] = new_parent
                for gen in range(2, keep_past_generations + 1):
                    prev_gen = new_pop.past[gen - 1]
                    if prev_gen is None or prev_gen.past is None or len(prev_gen.past) < 2:
                        break
                    new_pop.past[gen] = prev_gen.past[1]

                parent_N_old = int(self.relations.get('parent_N', self.past[1].N))
                parent_map = np.full(parent_N_old, -1, dtype=np.int32)
                parent_map[parent_keep] = np.arange(parent_keep.size, dtype=np.int32)

        relations_new = pedigree_relations.initialize_relations(new_pop.N)

        spouse_ids = np.asarray(self.relations['spouses'], dtype=np.int32)[i_keep]
        spouse_new = np.full(new_pop.N, -1, dtype=np.int32)
        valid_spouse = spouse_ids >= 0
        spouse_new[valid_spouse] = idx_map[spouse_ids[valid_spouse]]
        spouse_new[spouse_new < 0] = -1
        relations_new['spouses'] = spouse_new

        family_ids = np.asarray(self.relations['full_sibs'], dtype=np.int32)[i_keep]
        full_sibs_new = np.full(new_pop.N, -1, dtype=np.int32)
        valid_family = family_ids >= 0
        if np.any(valid_family):
            _, inverse = np.unique(family_ids[valid_family], return_inverse=True)
            full_sibs_new[valid_family] = inverse.astype(np.int32)
        relations_new['full_sibs'] = full_sibs_new

        parents_new = np.full((new_pop.N, 2), -1, dtype=np.int32)
        parents_old = source_parents[i_keep, :]
        valid_parents = parents_old >= 0
        if parent_source == 'current':
            parents_new[valid_parents] = idx_map[parents_old[valid_parents]]
            parents_new[parents_new < 0] = -1
            relations_new['parent_N'] = new_pop.N
            relations_new['parent_source'] = 'current'
        elif parent_map is not None and new_pop.past[1] is not None:
            parents_new[valid_parents] = parent_map[parents_old[valid_parents]]
            parents_new[parents_new < 0] = -1
            relations_new['parent_N'] = new_pop.past[1].N
            relations_new['parent_source'] = 'past'
        relations_new['parents'] = parents_new
        new_pop.relations = relations_new

        if (self.track_pedigree and keep_past_generations > 0
                and new_pop.past[1] is not None and np.any(parents_new >= 0)):
            new_pop.ped = Pedigree(
                new_pop.N,
                par_idx=relations_new['parents'],
                par_Ped=new_pop.past[1].ped,
            )
            new_pop.ped.construct_paths()
        else:
            new_pop.ped = Pedigree(new_pop.N)

        new_pop.traits = {}
        for name, trait in self.traits.items():
            trait_new = trait.index_trait(
                i_keep,
                self.G,
                G_already_indexed=False,
                pop=new_pop,
            )
            trait_new.name = name
            new_pop.traits[name] = trait_new

        return new_pop

    def prune_sibs(self, max_n_sibs: int = None,
                   min_n_sibs: int = None,
                   seed: int = None,
                   keep_past_generations: int = None) -> 'Population':
        '''
        Returns a population pruned by full-sib family size.

        Families smaller than `min_n_sibs` are removed entirely. Families larger than
        `max_n_sibs` are down-sampled uniformly at random so that exactly
        `max_n_sibs` members remain. Family sizes here include the focal individual,
        so singleton families have size 1.
        Parameters:
            max_n_sibs (int): Maximum allowed full-sib family size after pruning.
            min_n_sibs (int): Minimum allowed full-sib family size after pruning.
            seed (int): Optional seed controlling random down-sampling within large
                families.
            keep_past_generations (int): Number of ancestral generations to preserve
                in the returned population. Defaults to the current object's value.
        Returns:
            Population: Pruned population.
        '''
        if max_n_sibs is None and min_n_sibs is None:
            raise ValueError('Must provide at least one of `max_n_sibs` or `min_n_sibs`.')

        if max_n_sibs is not None:
            max_n_sibs = int(max_n_sibs)
            if max_n_sibs < 1:
                raise ValueError('`max_n_sibs` must be at least 1 when provided.')
        if min_n_sibs is not None:
            min_n_sibs = int(min_n_sibs)
            if min_n_sibs < 1:
                raise ValueError('`min_n_sibs` must be at least 1 when provided.')
        if min_n_sibs is not None and max_n_sibs is not None and min_n_sibs > max_n_sibs:
            raise ValueError('`min_n_sibs` cannot exceed `max_n_sibs`.')

        if keep_past_generations is None:
            keep_past_generations = self.keep_past_generations

        rng = np.random.default_rng(seed)
        keep_mask = np.zeros(self.N, dtype=bool)
        for sibship in self._get_full_sibships():
            family_size = sibship.size
            if min_n_sibs is not None and family_size < min_n_sibs:
                continue
            sibship_keep = sibship
            if max_n_sibs is not None and family_size > max_n_sibs:
                sibship_keep = np.sort(
                    rng.choice(sibship, size=max_n_sibs, replace=False).astype(np.int64, copy=False)
                )
            keep_mask[sibship_keep] = True

        if not np.any(keep_mask):
            raise ValueError('No individuals remain after applying the sibling-pruning filters.')

        return self.subset_individuals(
            np.flatnonzero(keep_mask),
            keep_past_generations=keep_past_generations,
        )

    def find_unrelated_individuals(self, threshold: float,
                                   source: str = 'GRM',
                                   relatedness: np.ndarray = None,
                                   standardize_ibd: bool = False) -> np.ndarray:
        '''
        Returns a greedy maximal subset of individuals with pairwise relatedness at
        or below the specified threshold.
        Parameters:
            threshold (float): Maximum allowed off-diagonal relatedness.
            source (str): Relatedness source to use when `relatedness` is not
                provided. Options are 'GRM' (default) and 'IBD'.
            relatedness (2D array): Optional precomputed relatedness matrix.
            standardize_ibd (bool): Passed to `get_relatedness_matrix()` when
                `source='IBD'`.
        Returns:
            i_keep (1D int array): Indices of retained individuals.
        '''
        if relatedness is None:
            relatedness = self.get_relatedness_matrix(
                source=source,
                standardize_ibd=standardize_ibd,
            )
        return pedigree_ibd.greedy_unrelated_subset(relatedness, threshold)

    def prune_related_individuals(self, threshold: float,
                                  source: str = 'GRM',
                                  relatedness: np.ndarray = None,
                                  standardize_ibd: bool = False,
                                  keep_past_generations: int = None,
                                  return_indices: bool = False):
        '''
        Returns a subsetted population in which all retained pairs have relatedness at
        or below `threshold`.
        Parameters:
            threshold (float): Maximum allowed off-diagonal relatedness.
            source (str): Relatedness source used when `relatedness` is not supplied.
                Options are 'GRM' (default) and 'IBD'.
            relatedness (2D array): Optional precomputed relatedness matrix.
            standardize_ibd (bool): Passed to `get_relatedness_matrix()` when
                `source='IBD'`.
            keep_past_generations (int): Optional override for the number of past
                generations to preserve in the returned population. If None, uses
                `Population.params.keep_past_generations`.
            return_indices (bool): If True, also returns the retained indices in the
                original population.
        Returns:
            Population or tuple: The pruned population, optionally along with `i_keep`.
        '''
        i_keep = self.find_unrelated_individuals(
            threshold=threshold,
            source=source,
            relatedness=relatedness,
            standardize_ibd=standardize_ibd,
        )
        pop_new = self.subset_individuals(
            i_keep=i_keep,
            keep_past_generations=keep_past_generations,
        )
        if return_indices:
            return (pop_new, i_keep)
        return pop_new

    def get_RDR_SNP_GRMs(self, G_par: np.ndarray = None,
                         std_method: str = 'observed') -> list:
        '''
        Constructs the three SNP-based GRMs used for Related Disequilibrium Regression (RDR). This could be done more efficiently by reusing intermediate calculations, but is currently implemented in a straightforward way for clarity.\

        Parameters:
            G_par (2D array): Optional parental genotype-sum matrix. If not
                provided, this is obtained from `Population.get_Gpar()`.
            std_method (str): Genotype standardization method passed to
                `standardize_G()`. Default is `'observed'`.

        Returns:
            list: `[R_oo, R_pp, R_op]`, where:
            - `R_oo = X_o X_o^T / M`
            - `R_pp = X_par X_par^T / (2M)`
            - `R_op = (X_o X_par^T + X_par X_o^T) / (2M)`

        Notes:
            - `X_o` is the standardized offspring genotype matrix with column mean 0 and variance 1.
            - `G_par` is the sum of maternal and paternal genotypes for each individual.
            - `X_par` is the standardized version of `G_par` with column mean 0 and variance 2.
        '''
        G_o = self.G
        X_o = self._get_standardized_genotypes(std_method=std_method)

        if G_par is None:
            G_par = self.get_Gpar()
        p_par = G_par.mean(axis=0) / (2 * self.P)
        X_par = genome_genotypes.standardize_G(G_par, p_par, 2 * self.P, impute=True,
                                  std_method=std_method, target_var=2.0)

        M = G_o.shape[1]
        R_oo = self.get_GRM(std_method=std_method)
        R_pp = genome_genotypes.compute_GRM(X_par) / 2
        R_op = (X_o @ X_par.T + X_par @ X_o.T) / (2 * M)

        return [R_oo, R_pp, R_op]

    def compute_PCA(self, n_components: int = 2, **kwargs) -> genome_pca.PCAResult:
        '''
        Computes a PCA for the current population.
        Parameters:
            n_components (int): Number of leading PCs to compute.
            **kwargs: Additional arguments passed to the PCA computation helper.
        Returns:
            PCAResult: PCA result object for the current population.
        '''
        return genome_pca.compute_PCA(
            G=self.G,
            p=self.p,
            P=self.P,
            n_components=n_components,
            **kwargs,
        )

    def add_trait(self, name: str, **kwargs):
        '''
        Initializes and generates a trait from pre-defined Effect objects.
        Parameters:
            name (str): Name of trait.
            **kwargs: All other arguments are passed to the Trait constructor. See Trait.__init__ for details.
        '''
        inputs = copy.deepcopy(kwargs.pop('inputs', {}))
        effects = kwargs.get('effects')
        if effects is None:
            raise ValueError('Population.add_trait requires an effects dictionary.')

        if any(name_i == 'A' for name_i in effects) and 'G' not in inputs:
            inputs['G'] = self.G
            inputs['G_std'] = self.get_G_std()
        if any(name_i == 'A_par' for name_i in effects) and 'G_par' not in inputs:
            inputs['G_par'] = self.get_Gpar()
            inputs['G_par_std'] = self.get_Gpar_std(inputs['G_par'])

        trait = Trait(inputs=inputs, pop=self, name=name, **kwargs)
        self.traits[name] = trait
    
    def add_trait_from_fixed_values(self, name: str, y: np.ndarray,
                                    trait_type: str = 'fixed'):
        '''
        Initializes and generates trait from specified fixed trait values. No sub component or effects (e.g. genetic, random effects) are stored.
        Parameters:
            name (str): Name of trait.
            y (1D array): Trait values for each individual in the population.
            trait_type (str): Non-regenerating trait type to assign. Must be 'fixed' or
                'permanent'. Default is 'fixed'.
        '''
        trait = Trait.from_fixed_values(y, trait_type=trait_type, pop=self, name=name)
        self.traits[name] = trait

    def update_traits(self, traits: list = None):
        '''
        Updates all traits by generating based on the current genotype matrix. Random noise components are re-generated. Causal genetic effects remain fixed. Only updates traits of composite type, except for sex, which is assigned through assign_sex().

        Parameters:
            traits (list of str): List of trait names to update. If None, updates all traits in the object.
        '''
        if traits is None:
            traits = self.traits.keys()
        # loops through each trait
        for key in self.traits:
            if key not in traits:
                continue
            trait = self.traits[key]
            if trait.type == 'composite':
                trait_inputs = {'G': self.G, 'G_std': self.get_G_std()}
                if 'A_par' in trait.effects:
                    G_par = self.get_Gpar()
                    trait_inputs['G_par'] = G_par
                    trait_inputs['G_par_std'] = self.get_Gpar_std(G_par)
                trait.update_inputs(copy_inputs=False, **trait_inputs)
                trait.generate_trait()
            elif key == 'sex':
                self.assign_sex()

    def _drop_nondynamic_random_effects(self, reason: str):
        '''
        Removes trait random effects that were defined from fixed matrices or
        arrays and therefore cannot safely follow a changed population state.
        '''
        for trait in self.traits.values():
            if trait.type != 'composite':
                continue
            trait._drop_random_effects_for_population_reshape(reason=reason)

    def assign_sex(self):
        '''
        Randomly assigns individuals to being female (0) or male (1), keeping the
        counts as balanced as possible.
        '''
        sex_arr = np.zeros(self.N, dtype=np.uint8)
        # randomly assigns as close as possible to half of indices to be 1s
        sex_arr[np.random.choice(self.N, self.N // 2, replace=False)] = 1
        # adds sex as a Trait object (with name 'sex')
        self.add_trait_from_fixed_values(name='sex', y=sex_arr)

    #######################################
    #### Analysis of object attributes ####
    ####################################### 
    
    def _define_metric(self, metric_name: str, metric_func: callable, shape: list, **kwargs):
        '''
        Defines a metric for the population object. The metric is computed using the provided function and stored in the `metric` attribute of the object.

        Parameters:
            metric_name (str): Name of the metric to define.
            metric_func (callable): Function that computes the metric.
            shape (list): A list specifying the shape of the metric output at each generation. So for example, a metric like allele frequency would have [M] since at each generation, there is an allele frequency for each variant. A metric like population size would have [1] since it is a single value at each generation. Can be greater than length 1.
            **kwargs: Additional keyword arguments to pass to the metric function. These should be fixed settings of the metric function, not parameters that change over time.
        '''
        self.metric[metric_name] = {
            'func': metric_func, # the base function that computes the metric
            'active': True, # whether the metric is run per generation
            'shape': shape, # the shape of the metric output at each generation
            'valid_keys': set(inspect.signature(metric_func).parameters.keys()), # the valid keys for the metric function
            'kwargs': kwargs, # the fixed settings of the metric function
            'values': None, # the values of the metric over generations, initialized to None
            'ts': None,
            }
    
    def _prep_metrics(self, generations: int):
        '''
        For each active metric, makes temporary array to store metric values over generations of simulation.
        Parameters:
            generations (int): Number of new generations to prepare each metric for.
        '''
        for metric_name in self.metric:
            if self.metric[metric_name]['active']:
                # adds number of generations to first dimension of metric history shape
                shape = [generations] + self.metric[metric_name]['shape']
                # initializes values to NaN
                self.metric[metric_name]['temp'] = np.full(shape, np.nan)
                self.metric[metric_name]['temp_ts'] = np.full(generations, -1, dtype=np.int64)

    def _update_temp_metrics(self, t: int, generation: int = None, **kwargs):
        '''
        Updates the temporary metrics for the current generation. This is done by calling the metric function with the current generation's data and storing the result in the temporary metric array. The method `_prep_metrics` should be called before this method to prepare the temporary metric arrays.
        Parameters:
            t (int): Generation index for new batch of simulated generations. That is, instead of the population's current generation, it should be the index of the generation within the pre-specified number of new generations to simulate by `simulate_generations()`.
            **kwargs: All extra arguments that are passed to any metric function. Only the parameters needed for each metric function are passed to that function. The pre-specified arguments set in `_define_metric()` are automatically passed to each metric function.
        '''
        for metric_name in self.metric:
            if self.metric[metric_name]['active']:
                fixed_args = self.metric[metric_name]['kwargs']
                new_args = {k: v for k, v in kwargs.items() if k in self.metric[metric_name]['valid_keys']}
                args = {**new_args, **fixed_args}
                # runs metric
                metric_output = self.metric[metric_name]['func'](**args)
                # forces single-value metrics to be in array format
                if not isinstance(metric_output, np.ndarray):
                    metric_output = np.array([metric_output])
                # updates metric values in temporary array
                self.metric[metric_name]['temp'][t,] = metric_output
                self.metric[metric_name]['temp_ts'][t] = self.t if generation is None else generation
        
    def _update_metric_history(self):
        '''
        Updates the metric history for each active metric by appending the temporary metric values to the permanent metric values stored in the object.
        '''
        for metric_name in self.metric:
            if self.metric[metric_name]['active']:
                # gets temporary metric values
                temp_values = self.metric[metric_name]['temp']
                temp_ts = self.metric[metric_name]['temp_ts']
                # appends to permanent values
                if self.metric[metric_name]['values'] is None or self.metric[metric_name]['ts'] is None:
                    values = temp_values
                    ts = temp_ts
                else:
                    values = np.concatenate((self.metric[metric_name]['values'], temp_values), axis=0)
                    ts = np.concatenate((self.metric[metric_name]['ts'], temp_ts), axis=0)

                if self.metric_retention == 'disabled':
                    self.metric[metric_name]['values'] = None
                    self.metric[metric_name]['ts'] = None
                else:
                    if self.metric_retention == 'store_last_k':
                        if self.metric_last_k is None or self.metric_last_k <= 0:
                            raise ValueError(
                                "metric_last_k must be a positive integer when metric_retention='store_last_k'."
                            )
                        values = values[-self.metric_last_k:]
                        ts = ts[-self.metric_last_k:]
                    elif self.metric_retention == 'summary_only':
                        values = values[[-1]]
                        ts = ts[[-1]]
                    elif self.metric_retention != 'store_every':
                        raise ValueError(f"Unknown metric retention policy: {self.metric_retention}")
                    self.metric[metric_name]['values'] = values
                    self.metric[metric_name]['ts'] = ts
                # clears temporary metric values
                self.metric[metric_name].pop('temp', None)
                self.metric[metric_name].pop('temp_ts', None)

    def _initialize_metrics(self, **kwargs):
        '''
        Initializes the metric history for each active metric by computing metrics for the current (starting) generation and storing them in the metric values. Should be called after defining metrics with `_define_metric()`.
        Parameters:
            **kwargs: All extra arguments that are passed to any metric function. Only the parameters needed for each metric function are passed to that function. The pre-specified arguments set in `_define_metric()` are automatically passed to each metric function.
        '''
        self._prep_metrics(1)
        self._update_temp_metrics(0, generation=self.t, **kwargs)
        self._update_metric_history()
    
    ####################################
    #### Simulating forward in time ####
    ####################################
        
    def next_generation(self, s: Union[float, np.ndarray] = _PARAM_UNSET,
                        mu: Union[float, np.ndarray] = _PARAM_UNSET) -> np.ndarray:
        '''
        Simulates new generation. Doesn't simulate offspring directly, meaning that future offspring have haplotypes drawn randomly from allele frequencies. Automatically updates object.

        Parameters:
            s (float or 1D array): Selection coefficient, such that an individual with the alternate allele has a (1+s) relative fitness compared to the reference allele. Occurs before mutation. If only a single value is provided, it is treated as the selection coefficient for all variants. Otherwise, must be an array of length M. Default is 0 (no selection).
            mu (float or 1D array): Mutation rate, such that the probability of any individual allele flipping to its alternate in the next generation is given by mu. Occurs after selection (i.e. mutation occurs in germline of current generation). Default is 0 (no mutations).
        Returns:
            H (3D array): N*M*P array of next generation's haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
        '''
        if s is _PARAM_UNSET:
            s = self.params.s
        if mu is _PARAM_UNSET:
            mu = self.params.mu
        # assigns same selection coefficient/mutation rate to all variants if only single value specified
        if isinstance(s, (float, int)):
            s = np.full(self.M, s)
        if isinstance(mu, (float, int)):
            mu = np.full(self.M, mu)
        # analytical allele frequency from which next generations' alleles are drawn from
        p = self.p
        # effect of selection
        p = p * (1 + s) / (1 + p*s)
        # effect of mutation (in gametes that lead to new generation, i.e. post-selection)
        p = p*(1-mu) + (1-p)*mu
        # effect of genetic drift
        H = genome_structure.draw_binom_haplos(p, self.N, self.P)

        return H

    def simulate_generations(self, generations: int = 1, related_offspring: bool = None,
                             trait_updates: bool = None, verbose: bool = False,
                             n_offspring_dist: str = None,
                             params: Union[PopulationParams, dict] = None,
                             **kwargs):
        '''
        Simulates specified number of generations beyond current generation. Can simulate offspring directly. Automatically updates object. Recombination rates are extracted from population parameters.

        Parameters:
            generations (int): Number of generations to simulate (beyond the current generation). Default is 1.
            related_offspring (bool): Optional one-run override for
                `Population.params.related_offspring`. If false, future offspring
                have alleles drawn randomly from allele frequencies.
            trait_updates (bool): Optional one-run override for
                `Population.params.trait_updates`. However, sex is always updated
                in each generation regardless.
            verbose (bool): Whether to print a progress message after each simulated generation. Default is False.
            n_offspring_dist (str): Optional one-run override for
                `Population.params.n_offspring_dist`. Ignored when
                `related_offspring=False`.
            params (PopulationParams or dict): Optional complete parameter set used
                only for this simulation run.
            **kwargs: One-run population-parameter overrides such as `s`, `mu`,
                `R`, `AM_r`, `AM_trait`, and `AM_type`.
        '''
        overrides = dict(kwargs)
        if related_offspring is not None:
            overrides['related_offspring'] = related_offspring
        if n_offspring_dist is not None:
            overrides['n_offspring_dist'] = n_offspring_dist
        if trait_updates is not None:
            overrides['trait_updates'] = trait_updates
        sim_params = self._resolved_params(params=params, overrides=overrides)
        trait_updates = sim_params.trait_updates

        uses_assortative_mating = (
            sim_params.AM_trait is not None and not np.isclose(sim_params.AM_r, 0.0)
        )
        if generations > 1 and not trait_updates and uses_assortative_mating:
            raise ValueError(
                'Assortative mating can only be simulated if traits are updated for each generation. '
                'Set trait_updates=True when passing AM-related arguments.'
            )
        if generations > 1 and not trait_updates:
            traits_requiring_updates = {
                trait_name: trait.required_per_generation_update_effects()
                for trait_name, trait in self.traits.items()
                if trait.type == 'composite' and trait.requires_per_generation_updates()
            }
            if traits_requiring_updates:
                trait_labels = ', '.join(
                    f"{trait_name}({', '.join(effect_names)})"
                    for trait_name, effect_names in traits_requiring_updates.items()
                )
                raise ValueError(
                    'Some traits require per-generation updates during multi-generation simulation, '
                    f'but trait_updates=False: {trait_labels}. '
                    'Set trait_updates=True.'
                )
        if self.metric_retention == 'store_last_k' and (self.metric_last_k is None or self.metric_last_k <= 0):
            raise ValueError("metric_last_k must be a positive integer when metric_retention='store_last_k'.")

        # preps metrics for new generations
        self._prep_metrics(generations)
        
        # loops through each generation
        for t in range(generations):
            if sim_params.related_offspring:
                (H, relations, Haplos) = self.generate_offspring(
                    s=sim_params.s,
                    mu=sim_params.mu,
                    R=sim_params.R,
                    n_offspring_dist=sim_params.n_offspring_dist,
                    AM_r=sim_params.AM_r,
                    AM_trait=sim_params.AM_trait,
                    AM_type=sim_params.AM_type,
                )
            else:
                H = self.next_generation(s=sim_params.s, mu=sim_params.mu)
                Haplos = None
                if self.track_haplotypes:
                    Haplos = np.full(H.shape, -1, dtype=np.int32)
                relations = None
            # updates objects and past
            self._update_obj(H=H, update_past=True, relations=relations, Haplos=Haplos, update_pedigree=self.track_pedigree)
            self._drop_nondynamic_random_effects(reason='simulating a new generation')
            # records metrics
            self._update_temp_metrics(t, generation=self.t, G=self.G)
            if trait_updates:
                self.update_traits()
            else:
                self.update_traits(traits=['sex']) # always updated in each generation
            if verbose:
                print(f'Simulated generation {self.t}')
        self.T_breaks.append(self.t)
        if not trait_updates:
            self.update_traits()
        # saves metrics to object
        self._update_metric_history()

    def _pair_mates(self, AM_r: float = 0, AM_trait: Union[str, np.ndarray] = None,
                    AM_type: str = 'phenotypic') -> np.ndarray:
        '''
        Pairs individuals up monogamously to mate and generate offspring. Population size must be multiple of 2. Allows for assortative mating (AM) if specified. Also updates the spouses relationship matrix.
        Parameters:
            AM_r (float): Desired correlation between AM trait values of spouses. Default is 0 (no assortative mating).
            AM_trait (str or 1D array): If a string, the name of the trait to use for assortative mating (as stored in the object). If an array, the trait values to use for assortative mating. If None, no assortative mating is performed. Default is None.
            AM_type (str): Type of assortative mating to perform. If 'phenotypic', uses the trait values directly. If 'genetic', uses the genetic values (only for object's stored traits). Default is 'phenotypic'.

        Returns:
            tuple ((iM, iP), rel_spouses):
            Where:
            - iMs (1D array): Array of length N/2 containing indices of the mothers.
            - iPs (1D array): Array of length N/2 containing indices of the fathers.
            - rel_spouses (2D array): Relationship N*N matrix for spouses, where each element is True if the individuals are spouses and False otherwise.
        '''
        # checks for population size
        if self.N % 2 != 0:
            raise Exception('Population size must be multiple of 2.')
        N2 = self.N // 2
        spouse_idx = np.full(self.N, -1, dtype=np.int32)

        # extracts assortative mating value:
        if AM_r != 0 and AM_trait is not None:
            if type(AM_trait) == np.ndarray:
                AM_values = AM_trait
            elif type(AM_trait) == str:
                if AM_trait not in self.traits:
                    raise Exception(f'Trait {AM_trait} not found in population traits.')
                if AM_type == 'phenotypic':
                    AM_values = self.traits[AM_trait].y
                elif AM_type == 'genetic':
                    AM_values = self.traits[AM_trait].y_['A']
            # standardizes
            AM_values = (AM_values - AM_values.mean()) / AM_values.std()
        else:
            AM_values = np.zeros(self.N)

        # extracts mom (female) and dad (male) indices and shuffles their order
        iMs = np.random.choice(np.where(self.traits['sex'].y == 0)[0], N2, replace=False)
        iPs = np.random.choice(np.where(self.traits['sex'].y == 1)[0], N2, replace=False)

        # computes mate value
        mate_values = AM_r * AM_values + np.random.normal(scale = np.sqrt(1-AM_r**2), size=self.N)
        # sorts mothers and fathers by mate value
        iMs = iMs[np.argsort(AM_values[iMs])]
        iPs = iPs[np.argsort(mate_values[iPs])]    
        # updates compact spouse indices
        spouse_idx[iMs] = iPs
        spouse_idx[iPs] = iMs

        return ((iMs, iPs), spouse_idx)

    def _sample_offspring_pair_counts(self, P_mate: np.ndarray, N_offspring: int,
                                      n_offspring_dist: str = 'constant') -> np.ndarray:
        '''
        Draws the number of offspring assigned to each mate pair.

        Parameters:
            P_mate (1D array): Probability or expected-weight vector across mate pairs.
            N_offspring (int): Total number of offspring to allocate across mate pairs.
            n_offspring_dist (str): Distribution of offspring counts across mate pairs.
                Supported values are:
                - 'poisson': Equivalent to sampling each offspring's
                  mate pair independently from `P_mate`.
                - 'constant': Allocates offspring as evenly as possible across mate
                  pairs by taking `floor(N_offspring * P_mate)` and then assigning one
                  extra offspring to randomly chosen pairs based on the residual
                  fractions until the total reaches `N_offspring`.
        Returns:
            1D int array: Number of offspring assigned to each mate pair.
        '''
        n_offspring_dist = str(n_offspring_dist).strip().lower()
        P_mate = np.asarray(P_mate, dtype=float)

        if P_mate.ndim != 1:
            raise ValueError('`P_mate` must be a 1D array.')
        if P_mate.size == 0:
            raise ValueError('Must provide at least one mate pair.')
        if N_offspring < 0:
            raise ValueError('`N_offspring` must be non-negative.')

        P_mate_sum = P_mate.sum()
        if P_mate_sum <= 0:
            raise ValueError('`P_mate` must sum to a positive value.')
        P_mate = P_mate / P_mate_sum

        if n_offspring_dist == 'poisson':
            return np.random.multinomial(N_offspring, P_mate).astype(np.int32, copy=False)

        if n_offspring_dist != 'constant':
            raise ValueError("`n_offspring_dist` must be either 'poisson' or 'constant'.")

        target_counts = N_offspring * P_mate
        offspring_counts = np.floor(target_counts).astype(np.int32)
        remainder = int(N_offspring - offspring_counts.sum())

        if remainder > 0:
            fractional_counts = target_counts - offspring_counts
            fractional_sum = fractional_counts.sum()
            if np.isclose(fractional_sum, 0.0):
                extra_prob = np.full(P_mate.size, 1 / P_mate.size)
            else:
                extra_prob = fractional_counts / fractional_sum
            extra_pairs = np.random.choice(P_mate.size, size=remainder, replace=False, p=extra_prob)
            offspring_counts[extra_pairs] += 1

        return offspring_counts

    def get_spouse_corr(self, trait: str, type: str = 'phenotypic') -> float:
        '''
        Computes the correlation between spouses for the specified trait. Only works if spouses relationship matrix is stored in the object (i.e. if `_pair_mates()` has been called at least once).
        Parameters:
            trait (str): The name of the trait for which to compute spouse correlation.
            type (str): he type of trait values to use. If 'phenotypic', uses the phenotypic values. If 'genetic', uses the genetic values. Default is 'phenotypic'.
        Returns:
            float: Correlation between spouses for the specified trait.
        '''
        if 'spouses' not in self.relations:
            raise Exception('Spouse relationship matrix not found in object. Make sure to call `_pair_mates()` at least once to generate spouse relationships.')
        if trait not in self.traits:
            raise Exception(f'Trait {trait} not found in population traits.')
        if type == 'phenotypic':
            AM_values = self.traits[trait].y
        elif type == 'genetic':
            AM_values = self.traits[trait].y_['A']
        else:
            raise Exception(f'Unknown type: {type}. Must be "phenotypic" or "genetic".')
        
        spouse_idx = np.asarray(self.relations['spouses'], dtype=np.int32)
        i_spouse = np.flatnonzero((spouse_idx >= 0) & (np.arange(self.N) < spouse_idx))
        j_spouse = spouse_idx[i_spouse]

        if len(i_spouse) == 0:
            raise Exception('No spouse pairs found in spouse relationship matrix.')

        spouse_values_1 = AM_values[i_spouse]
        spouse_values_2 = AM_values[j_spouse]

        if spouse_values_1.std() == 0 or spouse_values_2.std() == 0:
            return np.nan

        return stats_utils.corr(spouse_values_1, spouse_values_2)

    def get_Gpar(self) -> np.ndarray:
        '''
        Returns an N*M matrix whose (i, j) entry is the *sum* of genotype at variant j
        across individual i's two parents from the previous generation.
        '''
        if getattr(self, 'G_par', None) is not None:
            return self.G_par

        if 'parents' not in self.relations:
            raise Exception("Parent relationship information not found in object. Make sure `relations['parents']` exists.")

        parent_ids = np.asarray(self.relations['parents'], dtype=np.int32)
        if parent_ids.shape != (self.N, 2):
            raise Exception('Parent index array has incompatible shape for current generation.')

        parent_N = int(self.relations.get('parent_N', self.N))
        parent_source = self.relations.get('parent_source')
        if parent_source is None:
            valid_mask = parent_ids >= 0
            if np.any(valid_mask):
                row_ids = np.broadcast_to(
                    np.arange(self.N, dtype=np.int32)[:, None],
                    parent_ids.shape,
                )
                # Older combined-population objects store parent rows later in the same
                # population, whereas ordinary simulated generations index past[1].
                parent_source = 'current' if np.all(parent_ids[valid_mask] > row_ids[valid_mask]) else 'past'
            else:
                parent_source = 'past'

        if parent_source == 'current':
            G_source = self.G
        elif parent_source == 'past':
            if not hasattr(self, 'past') or self.past is None or len(self.past) < 2 or self.past[1] is None:
                raise Exception('Previous generation not available. Make sure `pop.past[1]` exists before calling `get_Gpar()`.')
            if parent_N != self.past[1].N:
                raise Exception('Stored parent dimension is incompatible with the previous generation.')
            G_source = self.past[1].G
        else:
            raise Exception("Relation metadata 'parent_source' must be either 'past' or 'current'.")

        if np.any(parent_ids < 0) or np.any(parent_ids >= G_source.shape[0]):
            raise Exception('Parent index array contains invalid indices for the source generation.')

        unique_parents, inverse = np.unique(parent_ids, return_inverse=True)
        G_unique = G_source[unique_parents]
        inverse = inverse.reshape(self.N, 2)
        self.G_par = G_unique[inverse[:, 0]] + G_unique[inverse[:, 1]]
        return self.G_par

    def impute_Gpar(self, method: str,
                    compute_error: bool = False,
                    **kwargs) -> dict:
        '''
        Imputes the sum of parental genotypes for the current generation.
        Parameters:
            method (str): Imputation method. Supported values are:
                - 'sibs_linear': Uses the full-sib family mean and doubles it.
                - 'AF_pop': Uses the proband genotype plus twice the population allele frequency.
                - 'AF_PCs': Uses the proband genotype plus twice a PC-adjusted allele frequency.
            compute_error (bool): Whether to compare the imputed matrix to the true
                parental genotype matrix returned by `get_Gpar()`.
            **kwargs: Method-specific keyword arguments passed to the selected helper.
        Returns:
            dict: Dictionary containing the imputed matrix under key 'Gpar' and any
            method-specific diagnostics.
        '''
        method_key = str(method).strip().lower()
        method_map = {
            'sibs_linear': self._impute_Gpar_sibs_linear,
            'af_pop': self._impute_Gpar_AF_pop,
            'af_pcs': self._impute_Gpar_AF_PCs,
        }
        if method_key not in method_map:
            supported = ', '.join(sorted(method_map))
            raise ValueError(f"Unknown Gpar imputation method '{method}'. Supported values are: {supported}.")

        result = method_map[method_key](**kwargs)
        if not isinstance(result, dict) or 'Gpar' not in result:
            raise ValueError("Imputation helpers must return a dictionary containing key 'Gpar'.")

        Gpar_imputed = np.asarray(result['Gpar'], dtype=float)
        if Gpar_imputed.shape != (self.N, self.M):
            raise ValueError(
                f"Imputed Gpar matrix must have shape {(self.N, self.M)}, got {Gpar_imputed.shape}."
            )

        output = dict(result)
        output['Gpar'] = Gpar_imputed
        output['method'] = method_key
        if compute_error:
            output['error_metrics'] = utils_matrix_metrics.summarize_matrix_error(
                self.get_Gpar(),
                Gpar_imputed,
            )
        return output

    def _impute_Gpar_sibs_linear(self) -> dict:
        '''
        Imputes parental genotypes as twice the full-sib family mean genotype.
        '''
        Gpar = np.empty((self.N, self.M), dtype=float)
        n_sibs = np.ones(self.N, dtype=np.int32)
        for sibship in self._get_full_sibships():
            Gpar[sibship, :] = 2.0 * self.G[sibship, :].mean(axis=0, dtype=float)
            n_sibs[sibship] = sibship.size

        if np.any(n_sibs == 1):
            warnings.warn(
                'Some samples have n_sibs=1, so the sibs_linear imputed Gpar is '
                'perfectly collinear with the proband genotype for those samples.',
                UserWarning,
            )

        return {
            'Gpar': Gpar,
            'n_sibs': n_sibs,
        }

    def _impute_Gpar_AF_pop(self) -> dict:
        '''
        Imputes parental genotypes using population-level allele frequencies.
        '''
        return {
            'Gpar': np.asarray(self.G, dtype=float) + 2.0 * np.asarray(self.p, dtype=float)[None, :],
        }

    def _estimate_pc_adjusted_allele_frequencies(self,
                                                 n_components: int = 4,
                                                 pca: genome_pca.PCAResult = None,
                                                 clip: bool = True) -> tuple[np.ndarray, dict]:
        '''
        Estimates individual-specific allele frequencies from a PCA reconstruction.
        '''
        max_components = min(self.N, self.M)
        if max_components < 1:
            raise ValueError('Population must contain at least one sample and one variant.')

        if n_components is None:
            n_components = min(4, max_components)
        n_components = int(n_components)
        if n_components < 1:
            raise ValueError('`n_components` must be at least 1.')

        if pca is None:
            n_components = min(n_components, max_components)
            pca = self.compute_PCA(n_components=n_components)
        else:
            if pca.n_samples != self.N or pca.n_features != self.M:
                raise ValueError('Provided PCAResult is incompatible with the current population.')
            if n_components > pca.scores.shape[1]:
                raise ValueError(
                    f"Requested n_components={n_components}, but provided PCAResult only has "
                    f"{pca.scores.shape[1]} component(s)."
                )

        X = np.asarray(self.X, dtype=float)
        scores = np.asarray(pca.scores[:, :n_components], dtype=float)
        eigenvalues = np.asarray(pca.eigenvalues[:n_components], dtype=float)
        positive = eigenvalues > 0
        if not np.any(positive):
            raise ValueError('Selected principal components have zero eigenvalue and cannot be used.')
        if not np.all(positive):
            scores = scores[:, positive]
            eigenvalues = eigenvalues[positive]

        loadings = (X.T @ scores) / eigenvalues[None, :]
        X_hat = scores @ loadings.T

        p = np.asarray(self.p, dtype=float)
        scale = np.sqrt(2.0 * p * (1.0 - p))
        p_adjusted_raw = p[None, :] + 0.5 * scale[None, :] * X_hat
        fraction_af_clipped = float(np.mean((p_adjusted_raw < 0.0) | (p_adjusted_raw > 1.0)))
        p_adjusted = np.clip(p_adjusted_raw, 0.0, 1.0) if clip else p_adjusted_raw

        total_ss = float(np.sum(X * X))
        resid_ss = float(np.sum((X - X_hat) ** 2))
        reconstruction_r2 = np.nan if np.isclose(total_ss, 0.0) else float(1.0 - resid_ss / total_ss)

        diagnostics = {
            'n_components_used': int(scores.shape[1]),
            'pc_reconstruction_r2': reconstruction_r2,
            'fraction_af_clipped': fraction_af_clipped,
        }
        return p_adjusted, diagnostics

    def _impute_Gpar_AF_PCs(self, n_components: int = 4,
                            pca: genome_pca.PCAResult = None,
                            clip: bool = True) -> dict:
        '''
        Imputes parental genotypes using PC-adjusted allele frequencies.
        '''
        p_adjusted, diagnostics = self._estimate_pc_adjusted_allele_frequencies(
            n_components=n_components,
            pca=pca,
            clip=clip,
        )
        return {
            'Gpar': np.asarray(self.G, dtype=float) + 2.0 * p_adjusted,
            **diagnostics,
        }

    def generate_offspring(self, s: Union[float, np.ndarray] = _PARAM_UNSET,
                           mu: Union[float, np.ndarray] = _PARAM_UNSET,
                           n_offspring_dist: str = _PARAM_UNSET,
                           R: Union[float, np.ndarray] = _PARAM_UNSET,
                           AM_r: float = _PARAM_UNSET,
                           AM_trait: Union[str, np.ndarray] = _PARAM_UNSET,
                           AM_type: str = _PARAM_UNSET,
                           **kwargs) -> np.ndarray:
        '''
        Pairs up mates and generates offspring from parents' haplotypes. Only works
        for diploids. Recombination rates are extracted from object attributes. Also
        updates parent-child relationship matrices.
        Parameters:
            s (float or 1D array): Selection coefficient, such that an individual with the alternate allele has a (1+s) relative fitness compared to the reference allele. Occurs before mutation. If only a single value is provided, it is treated as the selection coefficient for all variants. Otherwise, must be an array of length M. Default is 0 (no selection).
            mu (float or 1D array): Mutation rate, such that the probability of any individual allele flipping to its alternate in the next generation is given by mu. Occurs after selection (i.e. mutation occurs in germline of current generation). Default is 0 (no mutations).
            n_offspring_dist (str): Distribution of offspring counts across mate
                pairs. Supported values are:
                - 'poisson': Each offspring's
                  mate pair is sampled independently from the pair weights.
                - 'constant' (default): Allocates offspring as evenly as possible across mate
                  pairs by taking the floor of each pair's expected offspring count and
                  randomly adding one extra offspring to some pairs until the total
                  reaches the target.
            **kwargs: All other arguments (related to assortative mating) are passed to the `_pair_mates` method. See that method for details.
        Returns:
            H (3D array): N*M*P array of offspring haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
            relations (dict): Dictionary containing relationship matrices for the current generation, including 'spouses' and 'parents'.
        '''
        if kwargs:
            unknown = ', '.join(sorted(kwargs))
            raise ValueError(f'Unknown population parameter(s): {unknown}.')
        if s is _PARAM_UNSET:
            s = self.params.s
        if mu is _PARAM_UNSET:
            mu = self.params.mu
        if n_offspring_dist is _PARAM_UNSET:
            n_offspring_dist = self.params.n_offspring_dist
        if R is _PARAM_UNSET:
            R = self.params.R
        else:
            R = self._validate_R(R)
        if AM_r is _PARAM_UNSET:
            AM_r = self.params.AM_r
        if AM_trait is _PARAM_UNSET:
            AM_trait = self.params.AM_trait
        if AM_type is _PARAM_UNSET:
            AM_type = self.params.AM_type

        # checks ploidy
        if self.P != 2:
            raise Exception('Offspring generation only works for diploids.')

        # Assortative Mating ####        
        # pairs up mates
        (iMs, iPs), spouse_idx = self._pair_mates(
            AM_r=AM_r,
            AM_trait=AM_trait,
            AM_type=AM_type,
        )

        # Selection ####
        if isinstance(s, (float, int)):
            if s == 0:
                W = np.ones(self.N, dtype=float)
            else:
                W = np.exp(np.log1p(self.G * s).sum(axis=1))
        else:
            s = np.asarray(s, dtype=float)
            W = np.exp(np.log1p(self.G * s[None, :]).sum(axis=1))
        # computes each pair's breeding weight
        W_pair = W[iMs] * W[iPs]
        # computs probability of each parent pair being chosen to mate for one offspring
        P_mate = W_pair / W_pair.sum()
        # determines population size of next generation (currently maintains population size)
        N_offspring = self.N
        # draws how many offspring each mate pair has, then expands to per-offspring
        # parent indices.
        offspring_counts = self._sample_offspring_pair_counts(
            P_mate=P_mate,
            N_offspring=N_offspring,
            n_offspring_dist=n_offspring_dist,
        )
        i_mate = np.repeat(np.arange(len(iMs), dtype=np.int32), offspring_counts)
        np.random.shuffle(i_mate)
        parents = np.stack((iMs[i_mate], iPs[i_mate]), axis=1).astype(np.int32, copy=False)
        family_ids = i_mate.astype(np.int32, copy=False)

        # Drift + Recombination ####
        # generates variants for which a crossover event happens for each parent of each offspring
        crossover_events = np.random.binomial(n=1, p=R.reshape(1, self.M, 1),
                                              size=(N_offspring, self.M, 2)).astype(bool)
        # determines the shift in haplotype phase for each parent's haplotype at each variant
        haplo_phase = np.logical_xor.accumulate(crossover_events, axis=1)
        # randomly chooses haplotype to start with for each parent of each offspring
        haplo_k0 = np.random.binomial(n=1, p=1/self.P, size=(N_offspring, 2)).astype(bool)
        haplo_ks = np.logical_xor(haplo_phase, haplo_k0[:, None, :]).astype(np.int8, copy=False)

        variant_idx = np.arange(self.M, dtype=np.int32)[None, :]
        maternal_idx = parents[:, 0][:, None]
        paternal_idx = parents[:, 1][:, None]

        H = np.empty((N_offspring, self.M, self.P), dtype=self.H.dtype)
        H[:, :, 0] = self.H[maternal_idx, variant_idx, haplo_ks[:, :, 0]]
        H[:, :, 1] = self.H[paternal_idx, variant_idx, haplo_ks[:, :, 1]]

        Haplos = None
        if self.track_haplotypes:
            self._require_haplotype_tracking()
            Haplos = np.empty((N_offspring, self.M, self.P), dtype=self.Haplos.dtype)
            Haplos[:, :, 0] = self.Haplos[maternal_idx, variant_idx, haplo_ks[:, :, 0]]
            Haplos[:, :, 1] = self.Haplos[paternal_idx, variant_idx, haplo_ks[:, :, 1]]

        # Mutations ####
        if isinstance(mu, (float, int)):
            if mu != 0:
                mutations = np.random.binomial(n=1, p=mu, size=H.shape).astype(H.dtype, copy=False)
                H ^= mutations
        else:
            mu = np.asarray(mu, dtype=float)
            mutations = np.random.binomial(n=1, p=mu[None, :, None], size=H.shape).astype(H.dtype, copy=False)
            H ^= mutations

        # updates relationship matrix for parent-child relationships
        relations = {
            'spouses': spouse_idx,
            'parents': parents,
            'parent_N': self.N,
            'parent_source': 'past',
            'full_sibs': family_ids,
        }

        return (H, relations, Haplos)

    def flatten_generations(self, generations: int = 1) -> 'Population':
        '''
        Combines the current generation with the specified number of past generations into a new single population object that is returned. Importantly, it updates the relationship and Pedigree matrices to reflect the relationships in the combined generations.
        Parameters:
            generations (int): Number of past generations to include in the new population object. Default is 1, meaning only the current and the previous generation are combined.
        '''
        from .superpopulation import SuperPopulation

        # creates a SuperPopulation object with the current generation and the specified number of past generations
        pops = []
        Ns = []
        for i in range(0, generations + 1):
            pops.append(self.past[i]) # at i=0, references itself
            Ns.append(self.past[i].N)
        spop = SuperPopulation(pops)
        # combines the populations together inside the SuperPopulation object
        pops_i = list(range(generations + 1))
        spop.join_populations(pops_i, shared_haplotypes=self.track_haplotypes)
        new_pop = spop.pops[-1] # the last population in the SuperPopulation is the combined one
        new_pop.set_params(keep_past_generations=self.keep_past_generations)
        new_pop.relations = pedigree_relations.initialize_relations(new_pop.N, N1=new_pop.N, parent_source='current')

        # updates relationship matrices in combined population
        Ns_cumsum = np.cumsum([0] + Ns)
        spouses = np.full(new_pop.N, -1, dtype=np.int32)
        full_sibs = np.full(new_pop.N, -1, dtype=np.int32)
        next_family = 0
        for gen, pop_gen in enumerate(pops):
            i_start = Ns_cumsum[gen]
            i_end = Ns_cumsum[gen + 1]

            spouse_ids = np.asarray(pop_gen.relations['spouses'], dtype=np.int32)
            valid_spouse = spouse_ids >= 0
            shifted_spouses = np.full(pop_gen.N, -1, dtype=np.int32)
            shifted_spouses[valid_spouse] = spouse_ids[valid_spouse] + i_start
            spouses[i_start:i_end] = shifted_spouses

            family_ids = np.asarray(pop_gen.relations['full_sibs'], dtype=np.int32)
            valid_family = family_ids >= 0
            shifted_family = np.full(pop_gen.N, -1, dtype=np.int32)
            if np.any(valid_family):
                _, inverse = np.unique(family_ids[valid_family], return_inverse=True)
                shifted_family[valid_family] = inverse + next_family
                next_family = shifted_family[valid_family].max() + 1
            full_sibs[i_start:i_end] = shifted_family

        for gen in range(generations):
            gen_parents = pops[gen].relations['parents'].copy()
            i_start = Ns_cumsum[gen]
            i_end = Ns_cumsum[gen + 1]
            valid_mask = gen_parents >= 0
            gen_parents[valid_mask] += Ns_cumsum[gen + 1]
            new_pop.relations['parents'][i_start:i_end, :] = gen_parents

        # creates accurate Pedigree object for combined population

        # starts by filling in diagonal blocks
        for gen in range(generations + 1):
            # copies over each generation's Pedigree paths
            for key, value in pops[gen].ped.paths.items():
                # need to shift indices based on generation
                key = (key[0] + Ns_cumsum[gen], key[1] + Ns_cumsum[gen])
                new_pop.ped.paths[key] = new_pop.ped.intern_path(value)
        # fills in off-diagonal blocks, by going through each adjacent diagonal at a time
        for gap in range(1, generations+1):
            for start_gen in range(generations+1 - gap):
                end_gen = start_gen + gap
                # gets nested list of ancestors at each generation for each individual in the younger generation
                ancestors = self.get_ancestors(base_gen=start_gen, end_gen=end_gen)
                # loops through pairs of individuals in start_gen and end_gen
                for i in range(Ns[start_gen]): # individual in younger generation
                    i_gen_ancs = ancestors[i][gap - 1] # ancestors of individual i at generation end_gen (generation of j)
                    for j in range(Ns[end_gen]): # individual in older generation
                        keys = ( (i_ancestor, j) for i_ancestor in i_gen_ancs ) # prepares keys for get_closest_path
                        # determines the closest path between i and j through i's ancestors
                        closest_anc_path, closest_anc_path_keys = new_pop.ped.get_closest_path(pops[end_gen].ped.paths, keys)
                        if closest_anc_path is None:
                            continue # no relationship, don't store anything
                        else:
                            closest_anc_path = new_pop.ped.intern_path(closest_anc_path)
                        
                        # first [0] extracts (arbitrarily) the first chain with the closest path
                        # the second [0] extracts the ID of i's ancestor for this chain
                        i_gen_closest = closest_anc_path_keys[0][0]
                        # converts i's ancestor ID (in end_gen indices) to index from i_gen_ancs
                        i_gen_closest_k = np.where(np.isin(i_gen_ancs, i_gen_closest))[0][0]
                        # converts this index to an array of bits, which informs whether a maternal or paternal meiosis happens as one goes up i's lineage to get to their closest ancestor to j
                        i_gen_closest_bits = np.array(misc_utils.to_bits(i_gen_closest_k, gap))
                        up_sexes = tuple( i_gen_closest_bits + 1 ) # converts to sex-informed path steps

                        # takes path of i's ancestor to j and extends it by the specified meioses
                        path_ij = new_pop.ped.extend_path(closest_anc_path, ups = up_sexes) # already interned
                        # gets reverse path
                        path_ji = new_pop.ped.reverse_path(path_ij)

                        # gets keys of each for new_pop's paths
                        key_ij = (i + Ns_cumsum[start_gen], j + Ns_cumsum[end_gen])
                        key_ji = (j + Ns_cumsum[end_gen], i + Ns_cumsum[start_gen])

                        # adds to paths
                        new_pop.ped.paths[key_ij] = path_ij
                        new_pop.ped.paths[key_ji] = path_ji
            
        # actually sets relations
        new_pop.relations['full_sibs'] = full_sibs
        new_pop.relations['spouses'] = spouses

        return new_pop

    def get_ancestors(self, base_gen: int = 0, end_gen: int = 1) -> list:
        '''
        Gets nested list of ancestors at each generation for each individual in the specified base generation.
        Parameters:
            base_gen (int): Generation index of individuals for whom to get ancestors. Default is 0 (current generation).
            end_gen (int): Generation index up to which ancestors are retrieved (inclusive). Must be greater than base_gen. Default is 1 (previous generation).
        Returns:
            ancestors (list): A list of length N, where each element is a list of length `end_gen - base_gen`, where each element is a list of ancestor indices at that generation.
        '''
        ancestors = []
        for i in range(self.past[base_gen].N):
            i_ancestors = [] # nested list of ancestors for individual i at each generation
            current_inds = [i] # indices of i's ancestors at current generation (initializes at gen=0)
            for gen in range(base_gen, end_gen): 
                next_inds = []
                for ind in current_inds:
                    # gets indices of parents for individual ind
                    par_inds = self.past[gen].relations['parents'][ind, :]
                    next_inds.extend(par_inds.tolist())
                i_ancestors.append(next_inds)
                current_inds = next_inds
            ancestors.append(i_ancestors)
        return ancestors
    
    def extract_IBD_segments(self, i_idxs: list = None, j_idxs: list = None, i_chrs: list = [0,1], j_chrs: list = [0,1]):
        '''
        Extracts IBD segments between all pairs of individuals of individuals (i,j). Returns a list of IBDSegment objects, which store information about the individuals inside it, among other things. IBD is not checked between the same individuals, and each pair is only uniquely tested once (i.e. if the same individuals are specified in both i_idxs and j_idxs, only the (i,j) pair is stored).

        Parameters:
            i_idxs (list): List of individual indices for whom to extract IBD segments. If None, uses all individuals in the population.
            j_idxs (list): List of individual indices to compare against for IBD segments. If None, uses all individuals in the population.
            i_chrs (list): List of haplotype indices (0 to P-1) for individual i to extract segments from. Default is [0,1] (both haplotypes are checked).
            j_chrs (list): List of haplotype indices (0 to P-1) for individual j to extract segments from. Default is [0,1] (both haplotypes are checked).
        Returns:
            IBD_segments (list): List of IBDSegment objects representing the IBD segments found between specified individuals.
        '''
        if isinstance(i_idxs, (int, np.integer)):
            i_idxs = [i_idxs]
        if isinstance(j_idxs, (int, np.integer)):
            j_idxs = [j_idxs]
        if i_idxs is None:
            i_idxs = list(range(self.N))
        if j_idxs is None:
            j_idxs = list(range(self.N))

        self._require_haplotype_tracking()
        IBD_segments = []
        for i in i_idxs:
            for j in j_idxs:
                if i in j_idxs and i >= j:
                    continue # skips same individual and duplicate pairs
                # gets IBD tensor between individuals i and j
                IBD_tensor_ij = pedigree_ibd.get_true_IBD_tensor(self.Haplos[i,:,:], self.Haplos[j,:,:])
                # gets list of IBD segments between individuals i and j
                IBD_segments_ij = pedigree_ibd.IBD_tensor_to_segments(IBD_tensor_ij, i_chrs=i_chrs, j_chrs=j_chrs)
                # adds individual i and j information to each segment
                for seg in IBD_segments_ij:
                    seg.i = i
                    seg.j = j
                # adds to running list
                IBD_segments.extend(IBD_segments_ij)
        return IBD_segments


    #######################
    #### Visualization ####
    #######################

    def plot_PCA(self, pca: genome_pca.PCAResult = None,
                 pcs: Tuple[int, int] = (1, 2),
                 color_by: str = None,
                 categorical: bool = None,
                 n_components: int = None,
                 title: str = 'Population PCA',
                 **kwargs):
        '''
        Plots a PCA for the current population.
        Parameters:
            pca (PCAResult): Optional pre-computed PCA result.
            pcs (tuple): Two 1-based PCs to plot.
            color_by (str): Optional trait name used to color points.
            categorical (bool): Whether to treat `color_by` values as categorical.
                Defaults to treating only `subpop` as categorical unless the trait values
                are non-numeric.
            n_components (int): Number of PCs to compute if `pca` is not provided.
                Defaults to the largest PC index requested in `pcs`.
            title (str): Plot title.
            **kwargs: Additional arguments passed to the PCA plotting helper.
        Returns:
            matplotlib axis: Axis containing the PCA plot.
        '''
        if pca is None:
            if n_components is None:
                n_components = max(pcs)
            pca = self.compute_PCA(n_components=n_components)

        values = None
        if color_by is not None:
            if color_by not in self.traits:
                raise ValueError(f"Trait '{color_by}' was not found in the population.")
            values = np.asarray(self.traits[color_by].y)
            if categorical is None and color_by == 'subpop':
                categorical = True

        return genome_plotting.plot_PCA(
            pca,
            pcs=pcs,
            values=values,
            categorical=categorical,
            title=title,
            color_label=color_by,
            **kwargs,
        )

    def plot_freq_over_time(self, j_keep: tuple = None,
                            legend=False, last_generations: int = None,
                            summarize: bool = False, quantiles: tuple = (0.25, 0.5, 0.75)):
        '''
        Plots variant allele frequencies over time.
        Parameters:
            j_keep (tuple): Variant indices to include when plotting. Defaults to all variants.
            legend (bool): Whether to include a legend in the plot for each line. Default is False.
            last_generations (int): Number specifying the number of most recent generations to plot. Defaults to all generations since beginning.
            summarize (bool): If true, instead of plotting individual variant trajectories, it plots the mean and specified quantiles of allele frequencies across variants at each generation. Default is False.
            quantiles (tuple): List of quantiles (e.g. 0.99) of allele frequencies across variants at each generation to plot. `summarize` must be set to True. Default is median, lower quartile, and upper quartile.
        '''
        # plots all variants if not specified
        if j_keep is None:
            j_keep = tuple( range(self.M) )
        # uses population's allele frequency history
        ps = self.metric['p']['values']
        ts_all = self.metric['p'].get('ts')
        if ps is None or ts_all is None:
            raise ValueError('Allele-frequency history is not available under the current metric retention policy.')
        # plots all generations if not specified
        if last_generations is None:
            keep_mask = np.ones(ps.shape[0], dtype=bool)
        else:
            keep_mask = ts_all >= (ts_all[-1] - last_generations + 1)
        # subsets to specified variants
        ts = ts_all[keep_mask]
        ps = ps[keep_mask][:, j_keep]
        
        # if True, gets mean and quartiles for variants over time, which are plotted instead
        if summarize:
            ps_mean, ps_quantile = genome_frequencies.summarize_ps(ps, quantiles)
            metrics = np.column_stack((ps_mean, ps_quantile.T))
            aes_line = {'color': ['deepskyblue'] + ['lightskyblue'] * len(quantiles),
                        'ls': ['--'] + [':'] * len(quantiles),
                        'labels': ['Mean'] + [f'{q*100}% percentile' for q in quantiles]}
        else:
            # gets metrics to plot
            metrics = ps
            aes_line = {'color': None, 'ls': '-',
                        'labels': [f'Variant {j_keep[j]}' for j in range(len(j_keep))]}
        # plot aesthetics
        aes = {'title': 'Allele Frequency Trajectories Over Time',
               'xlabel': 'Generation',
               'ylabel': 'Allele Frequency'}

        common_plotting.plot_over_time(metrics, ts, aes=aes, aes_line=aes_line, vlines = self.T_breaks, legend=legend)
        
    def plot_LD_matrix(self, LD_matrix: sparse.csr_matrix = None,
                      plot_range: Tuple[int, int] = None, type: str = 'LD',
                      omit_mono: bool = False):
        '''
        Plots LD/correlation between variants.

        Parameters:
            LD_matrix (sparse 2D matrix): Either an LD or correlation M*M matrix of variants. If not provided, uses object's LD/correlation matrix.
            plot_range (tuple): A tuple of length 2 containing the range of variant indices to plot. If not provided, plots all variants.
            type (str): Uses color scheme for either 'LD' (default) or 'corr' matrix.
            omit_mono (bool): Whether variants that are monomorphic (p = 0 or 1) should be skipped over when plotting. Default is False.
        '''
        # chooses between LD and correlation matrix
        if LD_matrix is None:
            if type == 'LD':
                LD_matrix = self.LD_matrix
            elif type == 'corr':
                LD_matrix = self.corr_matrix
        if plot_range is not None:
            start, stop = plot_range
        else:
            start = 0
            stop = LD_matrix.shape[0]

        # Convert the sparse matrix to a dense array
        LD_matrix_dense = LD_matrix[start:stop, start:stop].toarray()
        
        # Skips plotting monomorphic variants
        if omit_mono:
            p = self.p[start:stop]
            j_mono = (p == 0) | (p == 1)
            LD_matrix_dense = np.delete(LD_matrix_dense, j_mono, axis=0)
            LD_matrix_dense = np.delete(LD_matrix_dense, j_mono, axis=1)

        # Create the heatmap
        plt.figure(figsize=(10, 8))
        if type == 'LD':
            plt.imshow(LD_matrix_dense, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        elif type == 'corr':
            plt.imshow(LD_matrix_dense, cmap='seismic', interpolation='nearest', vmin=-1, vmax=1)
        plt.colorbar(label='LD Value')
        plt.title(f'{type} Matrix Heatmap')
        plt.xlabel('Variant Index')
        plt.ylabel('Variant Index')
        plt.show()

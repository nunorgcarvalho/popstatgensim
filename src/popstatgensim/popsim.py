'''
This file contains classes related to population and statistical genetics.
The classes present are:
- Population: Models a population of individuals with genotypes. Attributes store raw and summary data on such individuals, including traits.
- Trait: Models a trait in a population, including its individual components.
These classes largely contain wrapper methods that call functions from other files. These wrapper methods extract the necessary attributes from the class object and pass them as arguments to the functions.
These wrapper methods often just update the class attributes, instead of returning the results.
'''

# imports ####
# package imports
from . import core_functions as core
from . import popgen_functions as pop
from . import statgen_functions as stat
from .relative_types import REL_TYPES
# other imports
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, Dict, Optional, List
from dataclasses import dataclass
from scipy import sparse
import copy
import inspect
import warnings
from collections import namedtuple

class Population:
    '''
    Class for a population to simulate. Contains genotype information. Contains methods to simulate change in population over time.
    '''

    ########################
    #### Initialization ####
    ########################

    def __init__(self, N: int, M: int, P: int = 2,
                 p_init: Union[float, np.ndarray] = None,
                 R_type: str = 'blocks',
                 keep_past_generations: int = 1,
                 track_pedigree: bool = False,
                 track_haplotypes: bool = False,
                 metric_retention: str = 'store_every',
                 metric_last_k: int = None,
                 seed: int = None):
        '''
        Initializes a population, simulating initial genotypes from specified allele frequencies.
        Parameters:
            N (int): Population size of individuals (not haplotypes).
            M (int): Total number of variants in genome.
            P (int): Ploidy of genotpes. Default is 2 (diploid).
            p_init (float or array): Initial allele frequency of variants. If only a single value is provided, it is treated as the initial allele frequency for all variants. Alternatively, can be an array of length M for variant-specfic allele frequencies. If not provided, default is uniform distribution of allele frequencies between 0.05 and 0.95.
            R_type (str). Type of recombination rates to use for genome. Options are: 'blocks' (default) for LD blocks (see pop.generate_LD_blocks()), 'indep' for independent sites (see self.make_sites_indep()), or 'uniform' for uniform recombination rates across genome (see pop.generate_chromosomes()). Resulting recombination rate array is stored in Population.R.
            keep_past_generations (int): Number of past generations to keep in the object. Default is 1, meaning the past generation is kept (on top of the current generation).
            track_pedigree (bool): Whether to track pedigree information (stored in Population.ped). Must keep at least 1 past generation. Default is False.
            track_haplotypes (bool): Whether to store haplotype IDs for founder/IBD tracking. Default is False.
            metric_retention (str): How metric histories are retained over time. Options are 'store_every', 'store_last_k', 'summary_only', and 'disabled'. Default is 'store_every'.
            metric_last_k (int): Number of most recent generations to retain when `metric_retention='store_last_k'`.
            seed (int): Initial seed to use when simulating genotypes (and allele frequencies if necessary).
        '''
        # sets seed if specified
        if seed is not None:
            seed = np.random.seed(seed)
        
        # draws initial allele frequencies from uniform distribution between 0.05 and 0.95 if not specified
        if p_init is None:
            p_init = pop.draw_p_init(M, method = 'uniform', params = (0.05, 0.95))
        elif type(p_init) == float or type(p_init) == int:
            # if only single value is given, all variants have same initial frequency
            p_init = np.full(M, p_init)

        # generates initial genotypes and records allele frequencies
        H = pop.draw_binom_haplos(p_init, N, P)

        # passes haplotype to base constructor for further initialization
        self._initialize_H(
            H,
            R_type=R_type,
            keep_past_generations=keep_past_generations,
            track_pedigree=track_pedigree,
            track_haplotypes=track_haplotypes,
            metric_retention=metric_retention,
            metric_last_k=metric_last_k,
        )
    
    @classmethod
    def from_H(cls, H: np.ndarray, R_type: str = 'blocks',
               keep_past_generations: int = 1,
               track_pedigree: bool = False,
               track_haplotypes: bool = False,
               metric_retention: str = 'store_every',
               metric_last_k: int = None):
        '''
        Initializes a population from a given haplotype array.
        Parameters:
            H (3D array): N*M*P array of haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
            R_type (str). Type of recombination rates to use for genome. Options are: 'blocks' (default) for LD blocks (see pop.generate_LD_blocks()), 'indep' for independent sites (see self.make_sites_indep()), or 'uniform' for uniform recombination rates across genome (see pop.generate_chromosomes()). Resulting recombination rate array is stored in Population.R.
            keep_past_generations (int): Number of past generations to keep in the object. Default is 1, meaning the past generation is kept (on top of the current generation).
            track_pedigree (bool): Whether to track pedigree information (stored in Population.ped). Must keep at least 1 past generation. Default is False.
            track_haplotypes (bool): Whether to store haplotype IDs for founder/IBD tracking. Default is False.
            metric_retention (str): Metric retention policy. See `Population.__init__()`.
            metric_last_k (int): Number of retained generations when `metric_retention='store_last_k'`.
        Returns:
            Population: A new Population object initialized with the given haplotype array.
        '''
        # creates new instance of class
        pop = cls.__new__(cls)
        # initializes the object with the given haplotype array
        pop._initialize_H(
            H,
            R_type=R_type,
            keep_past_generations=keep_past_generations,
            track_pedigree=track_pedigree,
            track_haplotypes=track_haplotypes,
            metric_retention=metric_retention,
            metric_last_k=metric_last_k,
        )
        return pop

    def _initialize_H(self, H: np.ndarray, R_type: str = 'blocks',
                      keep_past_generations: int = 1,
                      track_pedigree: bool = False,
                      track_haplotypes: bool = False,
                      metric_retention: str = 'store_every',
                      metric_last_k: int = None):
        '''
        Initializes a population from a given haplotype array. See the `from_H` class method for details.
        '''
        # sets basic population attributes from haplotype array
        (self.N, self.M, self.P) = H.shape

        # initializes default/initial attributes
        self.t = 0 # generation
        self.T_breaks = [self.t] # simulation breaks
        self.traits = {}
        self.BPs = np.arange(self.M) # variant positions in base pairs (BPs)
        self.G_par = None
        self._X = None
        self._G_std = None
        self._G_par_std = None
        self._X_dtype = np.float32
        # recombination rates
        if R_type == 'blocks':
            self.R = pop.generate_LD_blocks(self.M)
        elif R_type == 'indep':
            self.make_sites_indep()
        elif R_type == 'uniform':
            self.R = pop.generate_chromosomes(self.M, chrs=1, meioses_per_chr=1)
        self.K = np.diag(np.ones(self.N)) # kinship matrix (initially identity, not functional yet)
        self.track_pedigree = track_pedigree # whether to track pedigree information
        self.track_haplotypes = bool(track_haplotypes)
        self.metric_retention = metric_retention
        self.metric_last_k = metric_last_k
        # how many past generations to keep in memory
        self.past = [self]
        self.update_keep_past_gens(keep_past_generations=keep_past_generations)
        Haplos = None
        if self.track_haplotypes:
            Haplos = np.full(H.shape, -1, dtype=np.int32)

        # further attributes
        self._update_obj(H=H, Haplos=Haplos, update_past=False)
        self.assign_sex() # assigns sex (F:0 / M:1)
        self.relations = pop.initialize_relations(self.N)
        self.ped = Pedigree(self.N)

        # defines metrics
        self.metric = {}
        self._define_metric('p', pop.compute_freqs, shape = [self.M], # allele frequency
                            P = self.P)
        self._initialize_metrics(G = self.G)
    
    def update_keep_past_gens(self, keep_past_generations: int):
        '''
        Small function for structuring the 'past' attribute to reflect the number of generations in the past that are stored in the current population object. If the number of past generations is reduced, the older generations are removed from the array. If the number if past generations is increased, 'past' array is padded with Nones for future compatibility.
        Parameters:
            keep_past_generations (int): Number of past generations to keep in the object.
        '''
        current_past_gens = len(self.past) - 1 # should also just be self.keep_past_generations
        if keep_past_generations < current_past_gens:
            self.past = self.past[0:keep_past_generations + 1]
        elif keep_past_generations > current_past_gens:
            for _ in range(current_past_gens, keep_past_generations):
                self.past.append(None) # initializes past generations' objects as None
        elif keep_past_generations > 0:
            warnings.warn(f'Number of past generations kept is already {keep_past_generations}')
        
        self.keep_past_generations = keep_past_generations


    def make_sites_indep(self):
        '''
        Changes the recombination rates to make all sites independent.
        '''
        self.R = 0.5 * np.ones(self.M)

    def set_founding_haplotypes(self):
        '''
        Generates a complementary haplotype array for each individual containing a haplotype identifier for each allele. This functions treats the current generation as founders (individuals are unrelated from each other) such that each of their chromosomes has a unique identifier for all alleles in it. Subsequent generations can then track the inheritance of these founding haplotypes. Haplotypes are given an integer in the order they appear in the haplotype array.
        '''
        self.track_haplotypes = True
        ids = np.arange(self.N * self.P, dtype=np.int32).reshape(self.N, self.P)
        Haplos = np.broadcast_to(ids[:, None, :], self.H.shape).copy()
        self._update_obj(Haplos=Haplos)

    @property
    def X(self) -> np.ndarray:
        '''
        Lazily computes and caches the standardized genotype matrix.
        '''
        if self._X is None:
            X = pop.standardize_G(self.G, self.p, self.P, impute=True, std_method='observed')
            self._X = np.asarray(X, dtype=self._X_dtype)
        return self._X

    @X.setter
    def X(self, value: np.ndarray):
        if value is None:
            self._X = None
        else:
            self._X = np.asarray(value, dtype=self._X_dtype)

    def _require_haplotype_tracking(self):
        if not self.track_haplotypes or self.Haplos is None:
            raise ValueError(
                'Haplotype IDs are not stored for this population. '
                'Initialize with track_haplotypes=True or call set_founding_haplotypes().'
            )

    def get_relation_matrix(self, relation: str, dtype: np.dtype = np.uint8) -> np.ndarray:
        '''
        Returns a dense matrix representation of one stored relation.
        '''
        return pop.get_relation_matrix(self.relations, relation, self.N, dtype=dtype)

    def get_G_std(self) -> np.ndarray:
        '''
        Returns cached genotype standard deviations for the current generation.
        '''
        if self._G_std is None:
            self._G_std = np.asarray(stat.get_G_std_for_effects(self.G, P=self.P), dtype=np.float32)
        return self._G_std

    def get_Gpar_std(self, G_par: np.ndarray = None) -> np.ndarray:
        '''
        Returns cached parental-genotype standard deviations for the current generation.
        '''
        if G_par is None:
            if self._G_par_std is None:
                G_par = self.get_Gpar()
                self._G_par_std = np.asarray(
                    stat.get_G_std_for_effects(G_par, P=2 * self.P),
                    dtype=np.float32,
                )
            return self._G_par_std
        return np.asarray(stat.get_G_std_for_effects(G_par, P=2 * self.P), dtype=np.float32)


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
            self.G = pop.make_G(self.H)
            self.G_par = None
            self._G_std = None
            self._G_par_std = None
            self.p = pop.compute_freqs(self.G, self.P)
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
        Stores boolean sparse matrix of variants (`neighbor_matrix`) within the specified LD window distance (`LDwindow`) based on object's variant positions (currently only supports `BPs`). Calls `make_neighbor_matrix()` from `popgen_functions.py`.
        Parameters:
            LDwindow (float): Maximum distance between variants to be considered neighbors. In the same units as the positions used. If not provided, defaults to infinite maximum distance.
        '''
        # by default, uses object's base pair positions
        positions = self.BPs
        self.neighbor_matrix = pop.make_neighbor_matrix(positions=positions, LDwindow=LDwindow)

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
        self.corr_matrix = pop.compute_corr_matrix(self.X, self.neighbor_matrix)
        self.LD_matrix = pop.compute_LD_matrix(self.corr_matrix)

    def store_GRM(self):
        '''
        Stores the genetic relationship matrix (GRM) in the `GRM` attribute using object's standardized genotype matrix. Calls `compute_GRM()` from `popgen_functions.py`.
        '''
        self.GRM = pop.compute_GRM(self.X)

    def get_RDR_SNP_GRMs(self, G_par: np.ndarray = None) -> list:
        '''
        Constructs the three SNP-based GRMs used for Related Disequilibrium Regression (RDR). This could be done more efficiently by reusing intermediate calculations, but is currently implemented in a straightforward way for clarity.\

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
        X_o = pop.standardize_G(G_o, self.p, self.P, impute=True, std_method='observed')

        if G_par is None:
            G_par = self.get_Gpar()
        p_par = G_par.mean(axis=0) / (2 * self.P)
        X_par = pop.standardize_G(G_par, p_par, 2 * self.P, impute=True,
                                  std_method='observed', target_var=2.0)

        M = G_o.shape[1]
        R_oo = pop.compute_GRM(X_o)
        R_pp = pop.compute_GRM(X_par) / 2
        R_op = (X_o @ X_par.T + X_par @ X_o.T) / (2 * M)

        return [R_oo, R_pp, R_op]

    def compute_PCA(self, n_components: int = 2, **kwargs) -> pop.PCAResult:
        '''
        Computes a PCA for the current population.
        Parameters:
            n_components (int): Number of leading PCs to compute.
            **kwargs: Additional arguments passed to `pop.compute_PCA()`.
        Returns:
            pop.PCAResult: PCA result object for the current population.
        '''
        return pop.compute_PCA(
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
        if 'Gpar' in kwargs:
            kwargs['inputs'] = kwargs.get('inputs', {})
            kwargs['inputs']['G_par'] = kwargs.pop('Gpar')

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
    
    def add_trait_from_effects(self, name: str, **kwargs):
        '''
        Backward-compatible wrapper that builds GeneticEffect objects from raw effect arrays.
        Parameters:
            name (str): Name of trait.
            **kwargs: Effect arrays and Trait inputs.
        '''
        if 'Gpar' in kwargs:
            kwargs['G_par'] = kwargs.pop('Gpar')

        effects_raw = kwargs.pop('effects')
        is_standardized = not kwargs.pop('per_allele', False)
        var_Eps = kwargs.pop('var_Eps', None)
        force_var = kwargs.pop('force_var', False)
        inputs = copy.deepcopy(kwargs.pop('inputs', {}))

        if isinstance(effects_raw, dict):
            effects_dict = effects_raw
        else:
            effects_dict = {'A': effects_raw}

        effect_objects = {}
        for effect_name, effect_values in effects_dict.items():
            effect_values = np.asarray(effect_values, dtype=float)
            G_std = kwargs.pop('G_std', None) if effect_name == 'A' else kwargs.pop('G_par_std', None)
            if G_std is None:
                if effect_name == 'A' and 'G' in kwargs:
                    G_std = stat.get_G_std_for_effects(np.asarray(kwargs['G']))
                elif effect_name == 'A_par' and 'G_par' in kwargs:
                    G_std = stat.get_G_std_for_effects(np.asarray(kwargs['G_par']))
            effect_objects[effect_name] = GeneticEffect.from_effects(
                effects=effect_values,
                is_standardized=is_standardized,
                G_std=G_std,
                name=effect_name,
                force_var=force_var,
            )

        if 'G' in kwargs:
            inputs['G'] = kwargs.pop('G')
        if 'G_par' in kwargs:
            inputs['G_par'] = kwargs.pop('G_par')

        if kwargs:
            raise ValueError(f'Unexpected keyword arguments: {sorted(kwargs)}')

        self.add_trait(name=name, effects=effect_objects, inputs=inputs, var_Eps=var_Eps)
    
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

    def assign_sex(self):
        '''
        Randomly assigns half of individuals to being female (0) and other half to being male (1). Requires population size to be even.
        '''
        if self.N % 2 != 0:
            raise Exception('Population size must be even to assign sex equally and for mating purposes.')
        sex_arr = np.zeros(self.N, dtype=np.uint8)
        # randomly assigns exactly half of indices to be 1s
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
        
    def next_generation(self, s: Union[float, np.ndarray] = 0.0,
                        mu: Union[float, np.ndarray] = 0.0) -> np.ndarray:
        '''
        Simulates new generation. Doesn't simulate offspring directly, meaning that future offspring have haplotypes drawn randomly from allele frequencies. Automatically updates object.

        Parameters:
            s (float or 1D array): Selection coefficient, such that an individual with the alternate allele has a (1+s) relative fitness compared to the reference allele. Occurs before mutation. If only a single value is provided, it is treated as the selection coefficient for all variants. Otherwise, must be an array of length M. Default is 0 (no selection).
            mu (float or 1D array): Mutation rate, such that the probability of any individual allele flipping to its alternate in the next generation is given by mu. Occurs after selection (i.e. mutation occurs in germline of current generation). Default is 0 (no mutations).
        Returns:
            H (3D array): N*M*P array of next generation's haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
        '''
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
        H = pop.draw_binom_haplos(p, self.N, self.P)

        return H

    def simulate_generations(self, generations: int = 1, related_offspring: bool = True,
                             trait_updates: bool = False, verbose: bool = False,
                             **kwargs):
        '''
        Simulates specified number of generations beyond current generation. Can simulate offspring directly. Automatically updates object. Recombination rates are extracted from object attributes.

        Parameters:
            generations (int): Number of generations to simulate (beyond the current generation). Default is 1.
            related_offspring (bool): Whether the offspring of the next generation should be directly related to parents from previous generation by simulating meiosis and haplotype transfer. Default is True. If false, future offspring have alleles drawn randomly from allele frequencies.
            trait_updates (bool): Whether to update traits after each generation. However, sex is always updated in each generation, regardless of this setting. Default is False, meaning that traits are only updated at the end of the simulation.
            verbose (bool): Whether to print a progress message after each simulated generation. Default is False.
            **kwargs: All other arguments are passed to the `next_generation` or `generate_offspring` methods. See those methods for details.
        '''
        am_kwargs = {'AM_r', 'AM_trait', 'AM_type'}
        if generations > 1 and not trait_updates and any(key in kwargs for key in am_kwargs):
            raise ValueError(
                'Assortative mating can only be simulated if traits are updated for each generation. '
                'Set trait_updates=True when passing AM-related arguments.'
            )
        if self.metric_retention == 'store_last_k' and (self.metric_last_k is None or self.metric_last_k <= 0):
            raise ValueError("metric_last_k must be a positive integer when metric_retention='store_last_k'.")

        # preps metrics for new generations
        self._prep_metrics(generations)
        
        # loops through each generation
        for t in range(generations):
            if related_offspring:
                (H, relations, Haplos) = self.generate_offspring(**kwargs)
            else:
                H = self.next_generation(**kwargs)
                Haplos = None
                if self.track_haplotypes:
                    Haplos = np.full(H.shape, -1, dtype=np.int32)
                relations = None
            # updates objects and past
            self._update_obj(H=H, update_past=True, relations=relations, Haplos=Haplos, update_pedigree=self.track_pedigree)
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

        return core.corr(spouse_values_1, spouse_values_2)

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

    def generate_offspring(self, s: Union[float, np.ndarray] = 0,
                           mu: Union[float, np.ndarray] = 0,
                           **kwargs) -> np.ndarray:
        '''
        Pairs up mates and generates offspring for parents' haplotypes. Only works for diploids. Each pair always has two offspring. Recombination rates are extracted from object attributes. Also updates parent-child relationship matrix.
        Parameters:
            s (float or 1D array): Selection coefficient, such that an individual with the alternate allele has a (1+s) relative fitness compared to the reference allele. Occurs before mutation. If only a single value is provided, it is treated as the selection coefficient for all variants. Otherwise, must be an array of length M. Default is 0 (no selection).
            mu (float or 1D array): Mutation rate, such that the probability of any individual allele flipping to its alternate in the next generation is given by mu. Occurs after selection (i.e. mutation occurs in germline of current generation). Default is 0 (no mutations).
            **kwargs: All other arguments (related to assortative mating) are passed to the `_pair_mates` method. See that method for details.
        Returns:
            H (3D array): N*M*P array of offspring haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
            relations (dict): Dictionary containing relationship matrices for the current generation, including 'spouses' and 'parents'.
        '''
        # checks ploidy
        if self.P != 2:
            raise Exception('Offspring generation only works for diploids.')

        # Assortative Mating ####        
        # pairs up mates
        (iMs, iPs), spouse_idx = self._pair_mates(**kwargs)

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
        # draws indices of parents for each offspring
        i_mate = np.random.choice(np.arange(len(iMs)), size=N_offspring, p=P_mate)
        parents = np.stack((iMs[i_mate], iPs[i_mate]), axis=1).astype(np.int32, copy=False)
        family_ids = i_mate.astype(np.int32, copy=False)

        # Drift + Recombination ####
        # generates variants for which a crossover event happens for each parent of each offspring
        crossover_events = np.random.binomial(n=1, p=self.R.reshape(1, self.M, 1),
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
        new_pop.keep_past_generations = self.keep_past_generations
        new_pop.relations = pop.initialize_relations(new_pop.N, N1=new_pop.N, parent_source='current')

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
                        i_gen_closest_bits = np.array( core.to_bits(i_gen_closest_k, gap) )
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
                IBD_tensor_ij = pop.get_true_IBD_tensor(self.Haplos[i,:,:], self.Haplos[j,:,:])
                # gets list of IBD segments between individuals i and j
                IBD_segments_ij = pop.IBD_tensor_to_segments(IBD_tensor_ij, i_chrs=i_chrs, j_chrs=j_chrs)
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

    def plot_PCA(self, pca: pop.PCAResult = None,
                 pcs: Tuple[int, int] = (1, 2),
                 color_by: str = None,
                 categorical: bool = None,
                 n_components: int = None,
                 title: str = 'Population PCA',
                 **kwargs):
        '''
        Plots a PCA for the current population.
        Parameters:
            pca (pop.PCAResult): Optional pre-computed PCA result.
            pcs (tuple): Two 1-based PCs to plot.
            color_by (str): Optional trait name used to color points.
            categorical (bool): Whether to treat `color_by` values as categorical.
                Defaults to treating only `subpop` as categorical unless the trait values
                are non-numeric.
            n_components (int): Number of PCs to compute if `pca` is not provided.
                Defaults to the largest PC index requested in `pcs`.
            title (str): Plot title.
            **kwargs: Additional arguments passed to `pop.plot_PCA()`.
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

        return pop.plot_PCA(
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
            ps_mean, ps_quantile = pop.summarize_ps(ps, quantiles)
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

        core.plot_over_time(metrics, ts, aes=aes, aes_line=aes_line, vlines = self.T_breaks, legend=legend)
        
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




class Effect:
    '''
    Base class for a component-generating rule in a trait.
    '''
    required_inputs: Tuple[str, ...] = ()

    def __init__(self, name: str):
        self.name = name

    def refresh_from_inputs(self, inputs: dict):
        '''
        Updates any cached state that depends on the current trait inputs.
        '''
        return None

    def generate_component(self, inputs: dict, pop: Population = None) -> np.ndarray:
        '''
        Generates realized values for this component from the provided inputs.
        '''
        raise NotImplementedError


class GeneticEffect(Effect):
    '''
    Stores the effect sizes for a single genetic component of a trait.
    '''
    VALID_NAMES = {'A', 'A_par'}

    def __init__(self, var_indep: float, M: int, M_causal: int = None,
                 dist: str = 'normal', name: str = 'A',
                 force_var: bool = False, G: np.ndarray = None,
                 G_std: np.ndarray = None):
        if name not in self.VALID_NAMES:
            raise ValueError(f"Unknown genetic effect name: {name}")
        super().__init__(name)
        self.var_indep = float(var_indep)
        self.M = int(M)
        self.M_causal = self.M if M_causal is None else int(M_causal)
        self.force_var = bool(force_var)

        self.input_name = 'G' if name == 'A' else 'G_par'
        self.g_std_input_name = 'G_std' if name == 'A' else 'G_par_std'
        self.required_inputs = (self.input_name, self.g_std_input_name)

        self.G_std = None
        self.effects_standardized = None
        self.effects_per_allele = None
        self._rescale_to_var_indep = None

        effects_standardized, j_causal = stat.generate_causal_effects(
            self.M, self.M_causal, self.var_indep, dist
        )
        self.effects_standardized = effects_standardized
        self.j_causal = j_causal

        if G is not None or G_std is not None:
            self.update_G_std(G=G, G_std=G_std)

    @classmethod
    def from_effects(cls, effects: np.ndarray, is_standardized: bool = True,
                     G: np.ndarray = None, G_std: np.ndarray = None, name: str = 'A',
                     force_var: bool = False, var_indep: float = None) -> 'GeneticEffect':
        effects = np.asarray(effects, dtype=float)
        if effects.ndim != 1:
            raise ValueError('effects must be a 1D array.')

        obj = cls.__new__(cls)
        Effect.__init__(obj, name)
        if name not in cls.VALID_NAMES:
            raise ValueError(f"Unknown genetic effect name: {name}")

        obj.input_name = 'G' if name == 'A' else 'G_par'
        obj.g_std_input_name = 'G_std' if name == 'A' else 'G_par_std'
        obj.required_inputs = (obj.input_name, obj.g_std_input_name)
        obj.name = name
        obj.force_var = bool(force_var)
        obj.M = effects.shape[0]
        obj.j_causal = np.flatnonzero(effects != 0)
        obj.M_causal = len(obj.j_causal)
        obj.G_std = None
        obj.effects_standardized = None
        obj.effects_per_allele = None
        obj._rescale_to_var_indep = None

        if is_standardized:
            obj.effects_standardized = effects.copy()
        else:
            obj.effects_per_allele = effects.copy()

        if G is not None or G_std is not None:
            obj.update_G_std(G=G, G_std=G_std)

        if var_indep is None and obj.effects_standardized is not None and force_var:
            obj.var_indep = float(np.sum(obj.effects_standardized ** 2))
            warnings.warn(
                f"force_var=True with var_indep=None for effect {name}; "
                f"component variance will be scaled to sum(effects_standardized^2)={obj.var_indep:.6g}."
            )
        elif var_indep is None and obj.effects_standardized is not None:
            obj.var_indep = obj.M * obj.effects_standardized.var()
        elif var_indep is None:
            obj.var_indep = None
        else:
            obj.var_indep = float(var_indep)

        if (not obj.force_var) and obj.var_indep is not None:
            if obj.effects_standardized is not None:
                warnings.warn(
                    f"force_var=False with var_indep specified for effect {name}; "
                    "rescaling standardized effects to match the requested independent variance."
                )
                obj._scale_standardized_effects_to_var_indep(obj.var_indep)
            else:
                warnings.warn(
                    f"force_var=False with var_indep specified for effect {name}; "
                    "standardized effects will be rescaled once G or G_std is available."
                )
                obj._rescale_to_var_indep = obj.var_indep
        return obj

    def _scale_standardized_effects_to_var_indep(self, target_var_indep: float):
        if self.effects_standardized is None:
            raise ValueError('Standardized effects are not available.')
        current_var_indep = float(np.sum(self.effects_standardized ** 2))
        if np.isclose(current_var_indep, 0.0):
            if np.isclose(target_var_indep, 0.0):
                self.effects_standardized = np.zeros_like(self.effects_standardized)
            else:
                raise ValueError(f'Cannot rescale zero standardized effects for effect {self.name}.')
        else:
            self.effects_standardized = (
                self.effects_standardized * np.sqrt(target_var_indep / current_var_indep)
            )

        if self.G_std is not None:
            self.effects_per_allele = stat.get_standardized_effects(
                self.effects_standardized, self.G_std, std2allelic=True
            )
        self._rescale_to_var_indep = None

    def update_G_std(self, G: np.ndarray = None, G_std: np.ndarray = None,
                     update_var: bool = False):
        '''
        Updates stored genotype standard deviations and the corresponding standardized effects.
        '''
        if G is None and G_std is None:
            raise ValueError('Must provide either G or G_std.')
        if G is not None and G_std is not None:
            raise ValueError('Provide only one of G or G_std.')

        if G_std is None:
            G = np.asarray(G)
            if G.ndim != 2:
                raise ValueError('G must be a 2D array.')
            G_std = stat.get_G_std_for_effects(G, P=int(G.max()) if G.size > 0 else None)
        else:
            G_std = np.asarray(G_std, dtype=float)
            if G_std.ndim != 1:
                raise ValueError('G_std must be a 1D array.')

        if G_std.shape[0] != self.M:
            raise ValueError('Length of G_std must match M.')

        self.G_std = G_std.copy()

        if self.effects_per_allele is None:
            if self.effects_standardized is None:
                raise ValueError('No effects are stored for this GeneticEffect.')
            self.effects_per_allele = stat.get_standardized_effects(
                self.effects_standardized, self.G_std, std2allelic=True
            )

        self.effects_standardized = stat.get_standardized_effects(
            self.effects_per_allele, self.G_std, std2allelic=False
        )

        if self._rescale_to_var_indep is not None:
            self._scale_standardized_effects_to_var_indep(self._rescale_to_var_indep)

        if update_var:
            self.var_indep = self.M * self.effects_standardized.var()

    def refresh_from_inputs(self, inputs: dict):
        '''
        Refreshes the standardized effects from the current genotype standard deviations.
        '''
        if self.g_std_input_name in inputs:
            self.update_G_std(G_std=inputs[self.g_std_input_name])
        elif self.input_name in inputs:
            self.update_G_std(G=inputs[self.input_name])
        elif self.effects_per_allele is not None and self.effects_standardized is not None:
            return None
        else:
            raise ValueError(
                f"Missing required input '{self.g_std_input_name}' or '{self.input_name}' for effect {self.name}."
            )

    def generate_component(self, inputs: dict, pop: Population = None) -> np.ndarray:
        '''
        Computes realized genetic values from the current genotype input.
        '''
        if self.input_name not in inputs:
            raise ValueError(f"Missing required input '{self.input_name}' for effect {self.name}.")
        G = np.asarray(inputs[self.input_name])
        if G.ndim != 2:
            raise ValueError(f"Input '{self.input_name}' must be a 2D array.")
        if self.effects_per_allele is None:
            raise ValueError(f'Per-allele effects are not available for effect {self.name}.')
        if G.shape[1] != self.M:
            raise ValueError(f"Input '{self.input_name}' has incompatible number of variants for effect {self.name}.")
        values = stat.compute_genetic_value(G, self.effects_per_allele)

        if self.force_var:
            if self.var_indep is None:
                raise ValueError(f'force_var=True requires var_indep to be stored for effect {self.name}.')
            current_var = values.var()
            if np.isclose(current_var, 0.0):
                if np.isclose(self.var_indep, 0.0):
                    return np.zeros_like(values)
                raise ValueError(f'Cannot rescale zero-variance component for effect {self.name}.')
            values = values * np.sqrt(self.var_indep / current_var)

        return values


class FixedEffect(Effect):
    '''
    Stores the coefficient for a single fixed-effect covariate.
    '''

    def __init__(self, name: str, beta: float = None, var: float = None,
                 input_name: str = None, is_trait: bool = False):
        super().__init__(name)
        if beta is None and var is None:
            beta = 1.0
        if beta is not None and var is not None:
            warnings.warn(f'Both beta and var were provided for fixed effect {name}; var overrides beta.')
        self.beta = None if beta is None else float(beta)
        self.var = None if var is None else float(var)
        self.input_name = name if input_name is None else input_name
        self.is_trait = bool(is_trait)
        self.required_inputs = () if self.is_trait else (self.input_name,)

    def generate_component(self, inputs: dict, pop: Population = None) -> np.ndarray:
        '''
        Computes realized fixed-effect values from the current covariate input.
        '''
        if self.is_trait:
            if pop is None:
                raise ValueError(
                    f"Fixed effect {self.name} requires a Population object when is_trait=True."
                )
            if self.input_name not in pop.traits:
                raise ValueError(
                    f"Trait '{self.input_name}' was not found in the population for fixed effect {self.name}."
                )
            x = np.asarray(pop.traits[self.input_name].y, dtype=float)
        else:
            if self.input_name not in inputs:
                raise ValueError(f"Missing required input '{self.input_name}' for effect {self.name}.")
            x = np.asarray(inputs[self.input_name], dtype=float)
        if x.ndim != 1:
            raise ValueError(f"Input '{self.input_name}' must be a 1D array.")
        values = x if self.beta is None else self.beta * x
        if self.var is None:
            return values

        current_var = values.var()
        if np.isclose(current_var, 0.0):
            if np.isclose(self.var, 0.0):
                return np.zeros_like(values)
            raise ValueError(f'Cannot rescale zero-variance fixed effect {self.name}.')
        return values * np.sqrt(self.var / current_var)


class CorrelatedRandomEffect(Effect):
    '''
    Generates one random effect whose realized values are correlated with an
    already-realized component or trait.
    '''

    def __init__(self, name: str, var: float, r: float,
                 reference_component: str = None, reference_trait: str = None,
                 cluster_source: Union[str, np.ndarray] = None,
                 Z: np.ndarray = None, A: np.ndarray = None):
        super().__init__(name)
        self.var = float(var)
        self.r = float(r)
        self.reference_component = reference_component
        self.reference_trait = reference_trait
        self.cluster_source = copy.deepcopy(cluster_source)
        self.Z = None if Z is None else np.asarray(Z, dtype=float).copy()
        self.A = None if A is None else np.asarray(A, dtype=float).copy()
        self.required_inputs = ('N',)

        if self.var < 0:
            raise ValueError(f'var must be non-negative for correlated random effect {name}.')
        if abs(self.r) > 1:
            raise ValueError(f'r must be between -1 and 1 for correlated random effect {name}.')
        if self.reference_trait is None and self.reference_component is None:
            raise ValueError(
                f'Correlated random effect {name} must reference either a component in the same trait '
                'or a component/trait in another trait.'
            )

    def _reference_label(self) -> str:
        if self.reference_trait is None:
            return self.reference_component
        if self.reference_component is None:
            return self.reference_trait
        return f'{self.reference_trait}.{self.reference_component}'

    def _resolve_reference_values(self, inputs: dict, pop: Population = None) -> np.ndarray:
        if self.reference_trait is None:
            trait_components = inputs.get('_trait_components')
            if trait_components is None or self.reference_component not in trait_components:
                raise ValueError(
                    f"Correlated random effect {self.name} requires component "
                    f"'{self.reference_component}' to be generated earlier in the same trait."
                )
            values = trait_components[self.reference_component]
        else:
            if pop is None:
                raise ValueError(
                    f"Correlated random effect {self.name} requires a Population object "
                    'when referencing another trait.'
                )
            if self.reference_trait not in pop.traits:
                raise ValueError(
                    f"Trait '{self.reference_trait}' was not found in the population for "
                    f"correlated random effect {self.name}."
                )
            reference_trait = pop.traits[self.reference_trait]
            if self.reference_component is None:
                values = reference_trait.y
            else:
                if self.reference_component not in reference_trait.y_:
                    raise ValueError(
                        f"Trait '{self.reference_trait}' does not contain component "
                        f"'{self.reference_component}' for correlated random effect {self.name}."
                    )
                values = reference_trait.y_[self.reference_component]

        values = np.asarray(values, dtype=float)
        if values.ndim != 1:
            raise ValueError(f'Reference values for correlated random effect {self.name} must be 1D.')
        return values

    def _resolve_kernel(self, N: int, pop: Population = None) -> tuple[np.ndarray, np.ndarray]:
        if self.Z is not None:
            Z = np.asarray(self.Z, dtype=float)
            if Z.ndim != 2:
                raise ValueError(f'Z must be a 2D array for correlated random effect {self.name}.')
            if Z.shape[0] != N:
                raise ValueError(
                    f'Z for correlated random effect {self.name} must have {N} rows.'
                )
            A = np.eye(Z.shape[1], dtype=float) if self.A is None else np.asarray(self.A, dtype=float)
            return (Z, A)

        if self.cluster_source is None:
            A = np.eye(N, dtype=float) if self.A is None else np.asarray(self.A, dtype=float)
            return (None, A)

        if isinstance(self.cluster_source, str):
            if pop is None:
                raise ValueError(
                    f"Correlated random effect {self.name} requires a Population object "
                    f"to resolve cluster_source='{self.cluster_source}'."
                )
            if self.cluster_source not in pop.relations:
                raise ValueError(
                    f"cluster_source='{self.cluster_source}' was not found in Population.relations "
                    f"for correlated random effect {self.name}."
                )
            groups = np.asarray(pop.relations[self.cluster_source])
        else:
            groups = np.asarray(self.cluster_source)

        if groups.ndim != 1 or groups.shape[0] != N:
            raise ValueError(
                f'cluster_source for correlated random effect {self.name} must resolve to a length-{N} 1D array.'
            )

        Z = stat.build_design_matrix_from_groups(groups)
        A = np.eye(Z.shape[1], dtype=float) if self.A is None else np.asarray(self.A, dtype=float)
        return (Z, A)

    def _generate_component_identity_cluster_fast(self, reference_values: np.ndarray,
                                                  Z: np.ndarray, A: np.ndarray) -> Optional[np.ndarray]:
        '''
        Fast path for identity cluster relationship matrices, which avoids dense
        N x N kernel eigendecompositions.
        '''
        if A is None or not stat.is_identity_matrix(A):
            return None

        if Z is None:
            assignments = np.arange(reference_values.shape[0], dtype=np.int32)
        else:
            assignments = stat.get_group_assignments_from_design(Z)
            if assignments is None:
                return None

        reference_values = np.asarray(reference_values, dtype=float)
        u_fixed = reference_values - reference_values.mean()
        reference_var = float(u_fixed.var())
        if np.isclose(reference_var, 0.0):
            if np.isclose(self.r, 0.0):
                return np.zeros_like(reference_values)
            raise ValueError(
                f'Correlated random effect {self.name} cannot target non-zero correlation with '
                f'zero-variance reference {self._reference_label()}.'
            )

        y_fixed = u_fixed / np.sqrt(reference_var)
        propagated = stat.apply_identity_cluster_kernel_sqrt(assignments, y_fixed)
        trace_random = stat.get_identity_cluster_kernel_trace(assignments)
        rho = stat._calibrate_random_fixed_loading_from_propagated(
            u_fixed=u_fixed,
            propagated=propagated,
            trace_random=trace_random,
            target_corr=self.r,
            fixed_name=self._reference_label(),
            random_name=self.name,
        )

        latent_noise = np.random.normal(size=reference_values.shape[0])
        latent_values = rho * y_fixed + np.sqrt(max(1.0 - rho * rho, 0.0)) * latent_noise
        raw_values = stat.apply_identity_cluster_kernel_sqrt(assignments, latent_values)
        return stat._center_and_scale_random_effect(raw_values, self.var, self.name)

    def generate_component(self, inputs: dict, pop: Population = None) -> np.ndarray:
        '''
        Generates a random effect correlated with one previously realized reference.
        '''
        if 'N' not in inputs:
            raise ValueError(f"Missing required input 'N' for effect {self.name}.")
        N = int(inputs['N'])

        reference_values = self._resolve_reference_values(inputs, pop=pop)
        if reference_values.shape[0] != N:
            raise ValueError(
                f'Reference values for correlated random effect {self.name} have incompatible length.'
            )

        Z, A = self._resolve_kernel(N=N, pop=pop)

        fast_values = self._generate_component_identity_cluster_fast(
            reference_values=reference_values,
            Z=Z,
            A=A,
        )
        if fast_values is not None:
            return fast_values

        reference_var = float(reference_values.var())
        if np.isclose(reference_var, 0.0):
            if not np.isclose(self.r, 0.0):
                raise ValueError(
                    f'Correlated random effect {self.name} cannot target non-zero correlation with '
                    f'zero-variance reference {self._reference_label()}.'
                )
            random_effects = stat.get_random_effects(
                Zs=[Z],
                As=[A],
                variances=[self.var],
                names=[self.name],
            )
            return random_effects['values'][self.name]

        reference_name = f'{self.name}__reference'
        random_effects = stat.get_random_effects(
            Zs=[None, Z],
            As=[np.eye(N, dtype=float), A],
            variances=[reference_var, self.var],
            C=np.array([[1.0, self.r], [self.r, 1.0]], dtype=float),
            names=[reference_name, self.name],
            replace_random=[reference_values, None],
        )
        return random_effects['values'][self.name]


class NoiseEffect(Effect):
    '''
    Stores the generation rule for a noise component.
    '''
    VALID_NAMES = {'Eps'}
    required_inputs = ('N',)

    def __init__(self, name: str = 'Eps', var: float = 0.0, force_var: bool = True):
        if name not in self.VALID_NAMES:
            raise ValueError(f"Unknown noise effect name: {name}")
        super().__init__(name)
        self.var = var
        self.force_var = bool(force_var)

    def generate_component(self, inputs: dict, pop: Population = None) -> np.ndarray:
        '''
        Generates realized noise values for N individuals.
        '''
        if 'N' not in inputs:
            raise ValueError("Missing required input 'N' for effect Eps.")
        values = stat.generate_noise_value(int(inputs['N']), self.var)
        if not self.force_var:
            return values

        current_var = values.var()
        if np.isclose(current_var, 0.0):
            if np.isclose(self.var, 0.0):
                return np.zeros_like(values)
            raise ValueError('Cannot rescale zero-variance noise component.')
        return values * np.sqrt(self.var / current_var)


class Trait:
    '''
    Class for a trait belonging to a Population object.
    '''
    VALID_GENETIC_COMPONENTS = {'A', 'A_par'}
    DERIVED_INPUTS = {'G_std', 'G_par_std'}

    def __init__(self, effects: dict, inputs: dict, var_Eps: float = None,
                 pop: Population = None, name: str = None):
        '''
        Initializes and generates a trait from pre-defined effect objects.
        '''
        self._initialize_empty()
        self.pop = pop
        self.name = name
        if not isinstance(effects, dict) or len(effects) == 0:
            raise ValueError('effects must be a non-empty dictionary of Effect objects.')
        if not isinstance(inputs, dict):
            raise ValueError('inputs must be a dictionary.')

        self.effects = copy.deepcopy(effects)
        for name, effect in self.effects.items():
            if not isinstance(effect, Effect):
                raise ValueError(f'effects[{name}] must be an Effect object.')
            if effect.name != name:
                raise ValueError(f"Effect stored under key '{name}' must have matching name attribute.")

        if var_Eps is not None:
            self.effects['Eps'] = NoiseEffect('Eps', var=var_Eps)

        self.update_inputs(copy.deepcopy(inputs))
        self._validate_effect_definitions()
        self.generate_trait()
        self.validate()

    def _initialize_empty(self):
        self.pop = None
        self.name = None
        self.y = np.array([], dtype=float)
        self.y_ = {}
        self.var = {}
        self.var_initial = {}
        self.effects = {}
        self.inputs = {}
        self.type = 'composite'

    def _refresh_derived_inputs(self):
        '''
        Updates derived inputs from the currently stored primary inputs.
        '''
        N_candidates = []
        if 'G' in self.inputs:
            G = np.asarray(self.inputs['G'])
            if G.ndim != 2:
                raise ValueError("Trait input 'G' must be a 2D array.")
            self.inputs['G'] = G
            G_std = self.inputs.get('G_std')
            if G_std is not None:
                G_std = np.asarray(G_std, dtype=float)
                if G_std.ndim != 1 or G_std.shape[0] != G.shape[1]:
                    raise ValueError("Trait input 'G_std' must be a 1D array matching the number of variants in G.")
                self.inputs['G_std'] = G_std
            else:
                self.inputs['G_std'] = stat.get_G_std_for_effects(G, P=int(G.max()) if G.size > 0 else None)
            N_candidates.append(G.shape[0])
        if 'G_par' in self.inputs:
            G_par = np.asarray(self.inputs['G_par'])
            if G_par.ndim != 2:
                raise ValueError("Trait input 'G_par' must be a 2D array.")
            self.inputs['G_par'] = G_par
            G_par_std = self.inputs.get('G_par_std')
            if G_par_std is not None:
                G_par_std = np.asarray(G_par_std, dtype=float)
                if G_par_std.ndim != 1 or G_par_std.shape[0] != G_par.shape[1]:
                    raise ValueError("Trait input 'G_par_std' must be a 1D array matching the number of variants in G_par.")
                self.inputs['G_par_std'] = G_par_std
            else:
                self.inputs['G_par_std'] = stat.get_G_std_for_effects(G_par, P=int(G_par.max()) if G_par.size > 0 else None)
            N_candidates.append(G_par.shape[0])

        fixed_input_names = {
            effect.input_name for effect in self.effects.values()
            if isinstance(effect, FixedEffect) and not effect.is_trait
        }
        for input_name in fixed_input_names:
            if input_name not in self.inputs:
                continue
            x = np.asarray(self.inputs[input_name], dtype=float)
            if x.ndim != 1:
                raise ValueError(f"Trait input '{input_name}' must be a 1D array.")
            self.inputs[input_name] = x
            N_candidates.append(x.shape[0])

        if N_candidates:
            N_current = int(N_candidates[0])
            if any(n != N_current for n in N_candidates):
                raise ValueError('Trait inputs have inconsistent numbers of individuals.')
            if 'N' in self.inputs and int(self.inputs['N']) != N_current:
                raise ValueError("Trait input 'N' is inconsistent with the provided matrices.")
            self.inputs['N'] = N_current
        elif 'N' in self.inputs:
            self.inputs['N'] = int(self.inputs['N'])
        elif self.pop is not None:
            self.inputs['N'] = int(self.pop.N)
        else:
            raise ValueError("Trait input 'N' must always be available.")

    def update_inputs(self, inputs: dict = None, copy_inputs: bool = True, **kwargs):
        '''
        Updates the stored inputs for the trait and refreshes any derived inputs.
        '''
        new_inputs = {} if inputs is None else (copy.deepcopy(inputs) if copy_inputs else dict(inputs))
        new_inputs.update(kwargs)

        # When primary genotype inputs change, invalidate cached genotype SDs so they are
        # recomputed from the new matrices unless the caller explicitly overrides them.
        if 'G' in new_inputs and 'G_std' not in new_inputs:
            self.inputs.pop('G_std', None)
        if 'G_par' in new_inputs and 'G_par_std' not in new_inputs:
            self.inputs.pop('G_par_std', None)

        self.inputs.update(new_inputs)
        self._refresh_derived_inputs()

    def _update_empirical_variances(self):
        self.var = {name: values.var() for name, values in self.y_.items()}

    def _update_initial_variances(self):
        self.var_initial = {}
        for name, effect in self.effects.items():
            if isinstance(effect, GeneticEffect) and effect.var_indep is not None:
                self.var_initial[name] = effect.var_indep
            elif isinstance(effect, NoiseEffect):
                self.var_initial[name] = effect.var
            elif name in self.var:
                self.var_initial[name] = self.var[name]
        for name, value in self.var.items():
            self.var_initial.setdefault(name, value)

    def _validate_effect_definitions(self):
        '''
        Validates effect types and dependency ordering independently of realized values.
        '''
        if not isinstance(self.effects, dict):
            raise ValueError('Trait.effects must be a dictionary.')
        if not isinstance(self.inputs, dict):
            raise ValueError('Trait.inputs must be a dictionary.')

        trait_names = list(self.pop.traits.keys()) if self.pop is not None else []
        current_name = self.name
        if current_name is None and self.pop is not None:
            for trait_name, trait_obj in self.pop.traits.items():
                if trait_obj is self:
                    current_name = trait_name
                    break
        effect_order = list(self.effects.keys())

        for idx, (name, effect) in enumerate(self.effects.items()):
            if name in self.VALID_GENETIC_COMPONENTS:
                if not isinstance(effect, GeneticEffect):
                    raise ValueError(f'effects[{name}] must be a GeneticEffect object.')
            elif name == 'Eps':
                if not isinstance(effect, NoiseEffect):
                    raise ValueError('effects[Eps] must be a NoiseEffect object.')
            elif not isinstance(effect, (FixedEffect, CorrelatedRandomEffect)):
                raise ValueError(
                    f'effects[{name}] must be a GeneticEffect, FixedEffect, '
                    'CorrelatedRandomEffect, or NoiseEffect object.'
                )

            for input_name in effect.required_inputs:
                if input_name not in self.inputs:
                    raise ValueError(f"Missing required trait input '{input_name}' for effect {name}.")

            if isinstance(effect, FixedEffect) and effect.is_trait and self.pop is not None:
                if effect.input_name not in self.pop.traits:
                    raise ValueError(
                        f"Trait-backed fixed effect '{name}' requires population trait "
                        f"'{effect.input_name}', but it was not found."
                    )
                if current_name in trait_names:
                    if trait_names.index(effect.input_name) >= trait_names.index(current_name):
                        raise ValueError(
                            f"Trait-backed fixed effect '{name}' in trait '{current_name}' "
                            f"must depend on an earlier trait; got '{effect.input_name}'."
                        )

            if isinstance(effect, CorrelatedRandomEffect):
                if isinstance(effect.cluster_source, str) and self.pop is None:
                    raise ValueError(
                        f"Correlated random effect '{name}' requires a Population object "
                        f"to resolve cluster_source='{effect.cluster_source}'."
                    )

                if effect.reference_trait is None:
                    if effect.reference_component is None:
                        raise ValueError(
                            f"Correlated random effect '{name}' must reference a component "
                            'when no reference_trait is provided.'
                        )
                    if effect.reference_component not in effect_order:
                        raise ValueError(
                            f"Correlated random effect '{name}' references unknown same-trait "
                            f"component '{effect.reference_component}'."
                        )
                    if effect_order.index(effect.reference_component) >= idx:
                        raise ValueError(
                            f"Correlated random effect '{name}' must come after its referenced "
                            f"same-trait component '{effect.reference_component}'."
                        )
                else:
                    if current_name is not None and effect.reference_trait == current_name:
                        raise ValueError(
                            f"Correlated random effect '{name}' should use reference_trait=None "
                            'for same-trait component dependencies.'
                        )
                    if self.pop is None:
                        raise ValueError(
                            f"Correlated random effect '{name}' references another trait and "
                            'therefore requires a Population object.'
                        )
                    if effect.reference_trait not in self.pop.traits:
                        raise ValueError(
                            f"Correlated random effect '{name}' references missing trait "
                            f"'{effect.reference_trait}'."
                        )
                    reference_trait = self.pop.traits[effect.reference_trait]
                    if effect.reference_component is not None and effect.reference_component not in reference_trait.y_:
                        raise ValueError(
                            f"Correlated random effect '{name}' references missing component "
                            f"'{effect.reference_component}' in trait '{effect.reference_trait}'."
                        )
                    if current_name in trait_names:
                        if trait_names.index(effect.reference_trait) >= trait_names.index(current_name):
                            raise ValueError(
                                f"Correlated random effect '{name}' in trait '{current_name}' "
                                f"must depend on an earlier trait; got '{effect.reference_trait}'."
                            )

    def validate(self):
        '''
        Ensures the trait has a consistent internal structure.
        '''
        if not isinstance(self.y_, dict):
            raise ValueError('Trait.y_ must be a dictionary.')
        if not isinstance(self.var, dict):
            raise ValueError('Trait.var must be a dictionary.')
        if not isinstance(self.var_initial, dict):
            raise ValueError('Trait.var_initial must be a dictionary.')
        if not isinstance(self.effects, dict):
            raise ValueError('Trait.effects must be a dictionary.')
        if not isinstance(self.inputs, dict):
            raise ValueError('Trait.inputs must be a dictionary.')
        if 'N' not in self.inputs:
            raise ValueError("Trait.inputs must contain 'N'.")

        self.y = np.asarray(self.y)
        if self.y.ndim != 1:
            raise ValueError('Trait.y must be a 1D array.')
        N = self.y.shape[0]
        if int(self.inputs['N']) != N:
            raise ValueError("Trait input 'N' is inconsistent with Trait.y.")

        self._validate_effect_definitions()

        component_sum = np.zeros(N, dtype=float)
        for name, values in self.y_.items():
            values = np.asarray(values, dtype=float)
            if values.ndim != 1:
                raise ValueError(f'Component {name} must be a 1D array.')
            if values.shape[0] != N:
                raise ValueError(f'Component {name} has incompatible length.')
            self.y_[name] = values
            component_sum += values
            if name not in self.var:
                raise ValueError(f'Missing empirical variance for component {name}.')
            if not np.isclose(self.var[name], values.var()):
                raise ValueError(f'Empirical variance for component {name} is inconsistent with its values.')
            if name not in self.var_initial:
                raise ValueError(f'Missing initial variance for component {name}.')

        if self.y_ and not np.allclose(self.y, component_sum):
            raise ValueError('Trait.y must equal the sum of Trait.y_ components.')

        for name, effect in self.effects.items():
            if name not in self.y_:
                raise ValueError(f'Missing realized component for effect {name}.')


    @classmethod
    def from_fixed_values(cls, y: np.ndarray, trait_type: str = 'fixed',
                          pop: Population = None, name: str = None) -> 'Trait':
        '''
        Initializes a trait from a given array of fixed trait values.
        '''
        if trait_type not in {'fixed', 'permanent'}:
            raise ValueError("trait_type must be 'fixed' or 'permanent'.")
        trait = cls.__new__(cls)
        trait._initialize_empty()
        trait.pop = pop
        trait.name = name
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError('y must be a 1D array.')
        trait.y = y.copy()
        trait.y_ = {'fixed': y.copy()}
        trait.inputs = {'N': y.shape[0]}
        trait.type = trait_type
        trait._update_empirical_variances()
        trait._update_initial_variances()
        trait.validate()
        return trait

    def generate_trait(self, inputs: dict = None, **kwargs):
        '''
        Generates or updates trait values from the stored effects and current inputs.
        '''
        if self.type in {'fixed', 'permanent'}:
            raise ValueError(f"Cannot regenerate a {self.type} trait from stored effects.")
        if inputs is not None or kwargs:
            self.update_inputs(inputs=inputs, copy_inputs=False, **kwargs)

        self._validate_effect_definitions()
        self.y_ = {}
        total = None
        self.inputs['_trait_components'] = self.y_
        try:
            for name, effect in self.effects.items():
                effect.refresh_from_inputs(self.inputs)
                values = effect.generate_component(self.inputs, pop=self.pop)
                self.y_[name] = values
                if total is None:
                    total = np.array(values, copy=True)
                else:
                    total += values
        finally:
            self.inputs.pop('_trait_components', None)

        self.y = np.zeros(int(self.inputs['N']), dtype=float) if total is None else total
        self.type = 'composite'
        self._update_empirical_variances()
        self._update_initial_variances()

    def set_force_var(self, force_var: bool,
                      names: Union[str, list[str]] = None,
                      force_scale_effects: bool = False):
        '''
        Updates `force_var` for one or more GeneticEffect objects and regenerates the trait.
        If `force_scale_effects` is True, per-allele effects are rescaled so that the
        unforced genetic component variance in the current generation matches `var_indep`.
        '''
        if self.type in {'fixed', 'permanent'}:
            raise ValueError(f"Cannot update force_var for a {self.type} trait.")

        if names is None:
            target_names = [name for name, effect in self.effects.items()
                            if isinstance(effect, GeneticEffect)]
        elif isinstance(names, str):
            target_names = [names]
        else:
            target_names = list(names)

        if len(target_names) == 0:
            raise ValueError('No GeneticEffect objects were selected.')

        for name in target_names:
            if name not in self.effects:
                raise ValueError(f"Unknown effect name '{name}'.")
            if not isinstance(self.effects[name], GeneticEffect):
                raise ValueError(f"Effect '{name}' is not a GeneticEffect.")

        if force_scale_effects:
            warnings.warn(
                'force_scale_effects=True rescales per-allele effects for the selected '
                'GeneticEffect objects; individuals\' trait values may change as a result.'
            )

        for name in target_names:
            effect = self.effects[name]
            effect.refresh_from_inputs(self.inputs)

            if force_scale_effects:
                if effect.var_indep is None:
                    raise ValueError(
                        f"Effect '{name}' must store var_indep when force_scale_effects=True."
                    )
                if effect.effects_per_allele is None:
                    raise ValueError(
                        f"Effect '{name}' must have per-allele effects defined when force_scale_effects=True."
                    )
                if effect.input_name not in self.inputs:
                    raise ValueError(
                        f"Trait is missing input '{effect.input_name}' required for effect '{name}'."
                    )

                G_current = np.asarray(self.inputs[effect.input_name])
                values_unscaled = stat.compute_genetic_value(G_current, effect.effects_per_allele)
                current_var = values_unscaled.var()
                target_var = effect.var_indep

                if np.isclose(current_var, 0.0):
                    if np.isclose(target_var, 0.0):
                        effect.effects_per_allele = np.zeros_like(effect.effects_per_allele)
                    else:
                        raise ValueError(
                            f"Cannot rescale zero-variance component for effect '{name}'."
                        )
                else:
                    scale_factor = np.sqrt(target_var / current_var)
                    effect.effects_per_allele = effect.effects_per_allele * scale_factor

                if effect.g_std_input_name in self.inputs:
                    effect.update_G_std(G_std=self.inputs[effect.g_std_input_name])
                else:
                    effect.update_G_std(G=self.inputs[effect.input_name])

            effect.force_var = bool(force_var)

        self.generate_trait()

    def get_h2(self, method: str = 'additive_covariance', force_independence: bool = False) -> float:
        '''
        Returns heritability under one of several definitions.
        '''
        if 'A' not in self.y_:
            raise Exception("Trait must have an additive component 'A' to compute heritability.")

        if method == 'additive_covariance':
            var_A = self.y_['A'].var()
            if var_A == 0:
                return np.nan
            cov_YA = np.mean((self.y - self.y.mean()) * (self.y_['A'] - self.y_['A'].mean()))
            var_a = cov_YA**2 / var_A
        elif method == 'additive_variance':
            var_a = self.y_['A'].var()
        elif method == 'additive_effects':
            if 'A' not in self.effects:
                raise Exception("Trait must have additive genetic effects to compute heritability with method='additive_effects'.")
            var_a = np.sum(self.effects['A'].effects_standardized**2)
        else:
            raise ValueError(f"Unknown heritability method: {method}")

        if force_independence:
            var_y = np.sum([comp.var() for comp in self.y_.values()])
        else:
            var_y = self.y.var()
        if var_y == 0:
            return np.nan

        return var_a / var_y

    def get_components_matrix(self, exclude = None, include_y = True) -> np.ndarray:
        '''
        Returns a matrix of the trait components, where each column is a component and each row is an individual.
        '''
        components = []
        for key in self.y_.keys():
            if exclude is not None and key in exclude:
                continue
            components.append(self.y_[key])

        if include_y:
            components.append(self.y)

        component_matrix = np.column_stack(components)
        return component_matrix

    def get_vcov(self, exclude = None, include_y = True,
                 prettify: bool = True, corr: bool = False,
                 scale_by_y_var: bool = False) -> np.ndarray:
        '''
        Returns the covariance or correlation matrix of the trait components.
        '''
        component_matrix = self.get_components_matrix(exclude=exclude, include_y=include_y)
        vcov = np.corrcoef(component_matrix.T) if corr else np.cov(component_matrix.T)
        if scale_by_y_var:
            y_var = self.y.var()
            if y_var == 0:
                raise ValueError('Cannot scale by Trait.y variance when Trait.y has zero variance.')
            vcov = vcov / y_var
        if prettify:
            vcov = np.array([[f"{value:.4f}" for value in row] for row in vcov], dtype=object)
        return vcov

    @classmethod
    def concatenate_traits(cls, traits: list, G: np.ndarray = None,
                           G_par: np.ndarray = None, pop: Population = None,
                           name: str = None) -> 'Trait':
        '''
        Concatenates multiple Trait objects into a single Trait object.
        '''
        if len(traits) == 0:
            raise ValueError('Must provide at least one trait to concatenate.')

        first = traits[0]
        if any(trait.type != first.type for trait in traits):
            raise ValueError('All concatenated traits must have the same type.')
        if any(set(trait.y_.keys()) != set(first.y_.keys()) for trait in traits):
            raise ValueError('All concatenated traits must have the same component keys.')

        trait_new = cls.__new__(cls)
        trait_new._initialize_empty()
        trait_new.pop = pop
        trait_new.name = name
        trait_new.type = first.type
        if first.type in {'fixed', 'permanent'}:
            trait_new.y_ = {
                name: np.concatenate([trait.y_[name] for trait in traits])
                for name in first.y_.keys()
            }
            trait_new.y = np.sum(np.column_stack(list(trait_new.y_.values())), axis=1)
            trait_new._update_empirical_variances()
            trait_new.var_initial = copy.deepcopy(first.var_initial)
            trait_new.inputs = {'N': sum(int(trait.inputs['N']) for trait in traits)}
            return trait_new

        trait_new.effects = copy.deepcopy(first.effects)
        if any(isinstance(effect, GeneticEffect) for effect in trait_new.effects.values()) and len(traits) > 1:
            warnings.warn('Using the effects objects from the first trait when concatenating traits.')
        if any(isinstance(effect, GeneticEffect) and effect.force_var
               for effect in trait_new.effects.values()):
            warnings.warn(
                'Joining traits with GeneticEffect(force_var=True) will change genetic component '
                'values because the joined population triggers a new variance-matching rescaling, '
                'even though per-allele effects remain unchanged.'
            )
        if any(isinstance(effect, FixedEffect) and effect.var is not None
               for effect in trait_new.effects.values()):
            warnings.warn(
                'Joining traits with FixedEffect(var=...) may change fixed-effect values because '
                'differences in fixed-effect input variance across subpopulations affect rescaling. '
                'To preserve values exactly across joins, define fixed effects using beta only.'
            )

        trait_new.inputs = {'N': sum(int(trait.inputs['N']) for trait in traits)}
        shared_input_keys = set.intersection(*[set(trait.inputs.keys()) for trait in traits])
        for key in shared_input_keys - {'N'} - cls.DERIVED_INPUTS:
            values = [trait.inputs[key] for trait in traits]
            first_value = values[0]
            if key == 'G' and G is not None:
                trait_new.inputs['G'] = np.asarray(G)
                continue
            if key == 'G_par' and G_par is not None:
                trait_new.inputs['G_par'] = np.asarray(G_par)
                continue
            if isinstance(first_value, np.ndarray) and first_value.ndim >= 1 and first_value.shape[0] == int(traits[0].inputs['N']):
                trait_new.inputs[key] = np.concatenate(values, axis=0)
            else:
                trait_new.inputs[key] = copy.deepcopy(first_value)

        if 'A' in trait_new.effects and G is not None:
            trait_new.inputs['G'] = np.asarray(G)
        if 'A_par' in trait_new.effects and G_par is not None:
            trait_new.inputs['G_par'] = np.asarray(G_par)
        trait_new._refresh_derived_inputs()

        trait_new.y_ = {}
        total = None
        for name, effect in trait_new.effects.items():
            if isinstance(effect, NoiseEffect):
                values = np.concatenate([trait.y_[name] for trait in traits])
                trait_new.y_[name] = values
                if total is None:
                    total = np.array(values, copy=True)
                else:
                    total += values
                continue
            effect.refresh_from_inputs(trait_new.inputs)
            values = effect.generate_component(trait_new.inputs, pop=trait_new.pop)
            trait_new.y_[name] = values
            if total is None:
                total = np.array(values, copy=True)
            else:
                total += values

        trait_new.y = np.zeros(int(trait_new.inputs['N']), dtype=float) if total is None else total
        trait_new._update_empirical_variances()
        trait_new._update_initial_variances()
        return trait_new

    def index_trait(self, i_keep: np.ndarray, G: np.ndarray = None,
                    G_already_indexed: bool = False) -> 'Trait':
        '''
        Returns a Trait object that contains only the specified individuals.
        '''
        trait_new = self.__class__.__new__(self.__class__)
        trait_new._initialize_empty()
        trait_new.pop = self.pop
        trait_new.name = self.name
        trait_new.type = self.type
        trait_new.y = self.y[i_keep].copy()
        trait_new.y_ = {
            name: values[i_keep].copy()
            for name, values in self.y_.items()
        }
        trait_new._update_empirical_variances()
        trait_new.var_initial = copy.deepcopy(self.var_initial)
        trait_new.effects = copy.deepcopy(self.effects)

        trait_new.inputs = {'N': len(i_keep)}
        N_old = int(self.inputs['N']) if 'N' in self.inputs else None
        for key, value in self.inputs.items():
            if key == 'N' or key in self.DERIVED_INPUTS:
                continue
            if isinstance(value, np.ndarray) and value.ndim >= 1 and N_old is not None and value.shape[0] == N_old:
                trait_new.inputs[key] = value[i_keep].copy()
            else:
                trait_new.inputs[key] = copy.deepcopy(value)
        trait_new._refresh_derived_inputs()

        trait_new._update_initial_variances()
        return trait_new

class SuperPopulation:
    '''
    Class for a superpopulation, which contains multiple populations. Allows for multiple populations to be simulated forward in time together.
    '''

    #######################
    #### Intialization ####
    #######################
    def __init__(self, pops: Union[Population, list]):
        '''
        Initializes a superpopulation with a list of populations.
        Parameters:
            pops (list): List of Population objects to include in the superpopulation. Can also just be a single Population object.
        '''
        if isinstance(pops, Population):
            # if only a single population is passed, convert it to a list
            pops = [pops]
        self.pops = pops.copy() # makes a copy of the list of populations
        # initializes basic attributes
        self.era = 0 # will get updated to 1 by _update_era() below
        # creates active vector
        self.active = [True] * len(pops) # active populations as boolean
        self._update_era() # initializes active indices and history
        # creates lineage graph (adjacency matrix) for populations
        self._expand_graph()
    
    def _update_era(self):
        ''' Updates the list of active population indices based on the current state of the `active` attribute, the current era, and the history of active populations (as boolean matrix).'''
        # gets indices of active populations
        self.active_i = [i for i, active in enumerate(self.active) if active]  # active indices

        # updates history of active populations (and updates era)
        self.era += 1        
        active_history = np.zeros((self.era, len(self.pops)), dtype=bool)
        if hasattr(self, 'active_history'):
            active_history[:(self.era-1), :self.active_history.shape[1]] = self.active_history
        
        active_history[self.era-1, :] = self.active
        self.active_history = active_history

    ##################################
    #### Updating population list ####
    ##################################
    def add_population(self, pops: Union[Population, list], active_new: Union[bool, list] = True,
                       update_era: bool = True):
        '''
        Adds one or more new populations to the superpopulation.
        Parameters:
            list (Population or list): Population object to add. If a list is passed, it is assumed to be a list of Population objects, all of which are added in the order provided.
            active_new (bool or list): Whether the new populations should be active. If a single boolean is passed, it applies to all new populations. If a list is passed, it should have the same length as the number of new populations and specifies whether each population is active. Default is True, meaning all new populations are active.
            update_era (bool): Whether to update the era after inactivating the populations. Default is True.
        '''
        if isinstance(pops, Population):
            # if only a single population is passed, convert it to a list
            pops = [pops]
        
        self.pops.extend(pops)
        # updates active populations
        if isinstance(active_new, bool):
            # if only a single boolean is passed, apply it to all new populations
            active_new = [active_new] * len(pops)
        elif len(active_new) != len(pops):
            raise ValueError("Length of active_new must match the number of new populations.")
    
        # updates active status
        self.active.extend(active_new)
        if update_era: 
            self._update_era() # updates era, active indices, and history
        # updates lineage graph
        self._expand_graph()
        
    def inactivate_population(self, pop_i: Union[int, list], update_era: bool = True):
        '''
        Inactivates one or more populations from the superpopulation.
        Parameters:
            pop_i (int or list): Index of the population to inactivate. If a list is passed, it is assumed to be a list of indices of populations to inactivate.
            update_era (bool): Whether to update the era after inactivating the populations. Default is True.
        '''
        if isinstance(pop_i, int):
            pop_i = [pop_i]
        for i in pop_i:
            self.active[i] = False
        if update_era: 
            self._update_era() # updates era, active indices, and history

    def activate_population(self, pop_i: Union[int, list], update_era: bool = True):
        '''
        Activates one or more populations in the superpopulation.
        Parameters:
            pop_i (int or list): Index of the population to activate. If a list is passed, it is assumed to be a list of indices of populations to activate.
            update_era (bool): Whether to update the era after activating the populations. Default is True.
        '''
        if isinstance(pop_i, int):
            pop_i = [pop_i]
        for i in pop_i:
            self.active[i] = True
        if update_era: 
            self._update_era() # updates era, active indices, and history

    def _join_populations_build(self, pops: list, shared_haplotypes: bool = False,
                                keep_past_generations: int = 0,
                                generation: int = 0) -> Optional[Population]:
        '''
        Internal helper for joining population objects without mutating the superpopulation.
        `generation` indexes which entry of each source population's `past` list is being joined.
        '''
        if len(pops) < 2:
            raise ValueError("Must specify at least two populations to join.")

        gen_pops = []
        for pop_i in pops:
            if generation == 0:
                gen_pop = pop_i
            else:
                if (pop_i.past is None or len(pop_i.past) <= generation or
                        pop_i.past[generation] is None):
                    return None
                gen_pop = pop_i.past[generation]
            gen_pops.append(gen_pop)

        # merges haplotypes of specified populations
        H = np.concatenate([pop_i.H for pop_i in gen_pops], axis=0)
        track_pedigree = any(pop_i.track_pedigree for pop_i in gen_pops)
        track_haplotypes = any(pop_i.track_haplotypes for pop_i in gen_pops)
        if shared_haplotypes and any(not pop_i.track_haplotypes or pop_i.Haplos is None for pop_i in gen_pops):
            raise ValueError(
                'shared_haplotypes=True requires haplotype IDs to be stored for every joined population.'
            )
        new_pop = Population.from_H(
            H,
            keep_past_generations=keep_past_generations,
            track_pedigree=track_pedigree,
            track_haplotypes=track_haplotypes,
            metric_retention=gen_pops[0].metric_retention,
            metric_last_k=gen_pops[0].metric_last_k,
        )

        # preserves shared genome metadata from the first population
        new_pop.R = gen_pops[0].R.copy()
        new_pop.BPs = gen_pops[0].BPs.copy()
        new_pop.t = gen_pops[0].t
        new_pop.T_breaks = copy.deepcopy(gen_pops[0].T_breaks)

        # merging Haplotype IDs
        if track_haplotypes:
            if shared_haplotypes:
                Haplos = np.concatenate([pop_i.Haplos for pop_i in gen_pops], axis=0)
            else:
                shift = 0
                P = gen_pops[0].P
                Haplos_list = []
                for pop_i in gen_pops:
                    if pop_i.track_haplotypes and pop_i.Haplos is not None:
                        Haplos_i = pop_i.Haplos.copy()
                        valid_mask = Haplos_i >= 0
                        Haplos_i[valid_mask] += shift
                    else:
                        Haplos_i = np.full(pop_i.H.shape, -1, dtype=np.int32)
                    Haplos_list.append(Haplos_i)
                    shift += P * pop_i.N
                Haplos = np.concatenate(Haplos_list, axis=0)
            new_pop.Haplos = Haplos
        else:
            new_pop.Haplos = None

        # recursively joins older generations from the same source populations
        joined_prev = None
        if keep_past_generations > 0:
            joined_prev = self._join_populations_build(
                pops,
                shared_haplotypes=shared_haplotypes,
                keep_past_generations=keep_past_generations - 1,
                generation=generation + 1
            )
            if joined_prev is not None:
                new_pop.past[1] = joined_prev
                for gen in range(2, keep_past_generations + 1):
                    prev_gen = new_pop.past[gen - 1]
                    if (prev_gen is None or prev_gen.past is None or
                            len(prev_gen.past) < 2):
                        break
                    new_pop.past[gen] = prev_gen.past[1]

        # updates relations
        prev_N = joined_prev.N if joined_prev is not None else None
        relations = pop.initialize_relations(new_pop.N, N1=prev_N)
        row_offsets = np.cumsum([0] + [pop_i.N for pop_i in gen_pops])

        next_family = 0
        for k, pop_i in enumerate(gen_pops):
            i_start = row_offsets[k]
            i_end = row_offsets[k + 1]

            spouse_ids = np.asarray(pop_i.relations['spouses'], dtype=np.int32)
            valid_spouse = spouse_ids >= 0
            shifted_spouses = np.full(pop_i.N, -1, dtype=np.int32)
            shifted_spouses[valid_spouse] = spouse_ids[valid_spouse] + i_start
            relations['spouses'][i_start:i_end] = shifted_spouses

            family_ids = np.asarray(pop_i.relations['full_sibs'], dtype=np.int32)
            valid_family = family_ids >= 0
            shifted_family = np.full(pop_i.N, -1, dtype=np.int32)
            if np.any(valid_family):
                _, inverse = np.unique(family_ids[valid_family], return_inverse=True)
                shifted_family[valid_family] = inverse + next_family
                next_family = shifted_family[valid_family].max() + 1
            relations['full_sibs'][i_start:i_end] = shifted_family

        if joined_prev is not None:
            prev_sizes = [int(pop_i.relations.get('parent_N', pop_i.N)) for pop_i in gen_pops]
            if sum(prev_sizes) != joined_prev.N:
                raise ValueError(
                    "Joined parent population size is incompatible with the source "
                    "populations' parent relationship matrices."
                )
            col_offsets = np.cumsum([0] + prev_sizes)
            parents = np.full((new_pop.N, 2), -1, dtype=np.int32)

            for k, pop_i in enumerate(gen_pops):
                i_start = row_offsets[k]
                i_end = row_offsets[k + 1]
                j_start = col_offsets[k]

                pop_parent_ids = np.asarray(pop_i.relations['parents'], dtype=np.int32).copy()
                if pop_parent_ids.shape != (pop_i.N, 2):
                    raise ValueError(
                        "Population compact parent representation has incompatible shape."
                    )
                valid_mask = pop_parent_ids >= 0
                pop_parent_ids[valid_mask] += j_start
                parents[i_start:i_end, :] = pop_parent_ids

            relations['parents'] = parents
            relations['parent_N'] = joined_prev.N
            relations['parent_source'] = 'past'

        new_pop.relations = relations

        if joined_prev is not None and track_pedigree:
            new_pop.ped = Pedigree(new_pop.N, par_idx=relations['parents'],
                                   par_Ped=joined_prev.ped)
            new_pop.ped.construct_paths()

        # adds Trait objects by concatenating them, assumes the first population has all traits
        for name in gen_pops[0].traits.keys():
            traits = [pop_i.traits[name] for pop_i in gen_pops]
            has_A_par = any('A_par' in trait.effects for trait in traits)
            can_compute_G_par = (
                has_A_par and new_pop.past is not None and len(new_pop.past) >= 2
                and new_pop.past[1] is not None and 'parents' in new_pop.relations
            )
            G_par_new = new_pop.get_Gpar() if can_compute_G_par else None
            trait_new = Trait.concatenate_traits(
                traits, new_pop.G, G_par=G_par_new, pop=new_pop, name=name
            )
            new_pop.traits[name] = trait_new

        return new_pop

    def join_populations(self, pop_i: list = None, shared_haplotypes: bool = False,
                         keep_past_generations: int = 0):
        '''
        Joins multiple populations into a single population. Inactivates the original populations and creates a new population from the merged haplotypes. The new population is added to the superpopulation as an active population.
        Parameters:
            pop_i (list): List of indices of populations to join. If None, joins all active populations.
            shared_haplotypes (bool): Whether haplotype IDs already refer to the same underlying founders across populations. If False, each population's non-negative haplotype IDs are shifted to remain unique after joining. Default is False.
            keep_past_generations (int): Number of previous generations to preserve in the joined population. If greater than 0, the specified populations' stored past generations are recursively joined and attached to the new population's `past` attribute.
        '''
        if pop_i is None:
            pop_i = list(self.active_i)
        pops = [self.pops[i] for i in pop_i]
        new_pop = self._join_populations_build(
            pops,
            shared_haplotypes=shared_haplotypes,
            keep_past_generations=keep_past_generations
        )
        
        # updates superpopulation
        self.add_population(new_pop, active_new=True, update_era=False)
        self.inactivate_population(pop_i)

        # updates graph
        for i in pop_i:
            # updates graph to reflect that the populations are now joined
            self.graph[i, len(self.pops)-1] = 1

    def split_population(self, pop_i: int, N_new: Union[int, list] = 2):
        '''
        Splits a population into two or more populations. This is done by randomly drawing individuals from the source population. The new populations are created from the haplotypes of the original population. The original population is inactivated and the new populations are added to the superpopulation as active populations.
        Parameters:
            pop_i (int): Index of the population to split.
            N_new (int or list): Number of individuals in each new population. If a single integer is passed (Default = 2), it is treated as the number of equally-sized populations to split into. If a list is passed, it should have the same length as the number of new populations and specifies the number of individuals in each population. The list must sum to less than or equal to the number of individuals in the original population.
        '''
        source = self.pops[pop_i]
        # checks if population can be split
        if isinstance(N_new, int):
            # if only a single integer is passed, split into that many equally-sized populations
            N_new = [source.N // N_new] * N_new # rounds down to nearest integer
        elif len(N_new) < 2:
            raise ValueError("Must specify at least two new populations to split into.")
        elif sum(N_new) > source.N:
            raise ValueError("Sum of N_new must be less than or equal to the number of individuals in the original population.")
        K = len(N_new)  # number of new populations
        # shuffles indices of individuals in the source population
        i_shuffled = np.arange(source.N)
        np.random.shuffle(i_shuffled)  # shuffles haplotypes
        i_start = 0
        new_pops = []
        for i in range(K):
            # gets indices of individuals for the new population
            i_end = i_start + N_new[i]
            i_new = i_shuffled[i_start:i_end]
            # creates new population from the haplotypes of the original population
            H_new = source.H[i_new, :, :]
            new_pop = Population.from_H(
                H_new,
                track_haplotypes=source.track_haplotypes,
                metric_retention=source.metric_retention,
                metric_last_k=source.metric_last_k,
            )
            if source.track_haplotypes and source.Haplos is not None:
                new_pop.Haplos = source.Haplos[i_new, :, :].copy()
            # updates traits
            for name, trait in source.traits.items():
                trait_new = trait.index_trait(i_new, source.G, G_already_indexed=False)
                trait_new.pop = new_pop
                trait_new.name = name
                new_pop.traits[name] = trait_new
            
            new_pops.append( new_pop )
            # updates next index to start pulling from
            i_start = i_end
        # adds new population to the superpopulation
        new_pops_i = np.arange(len(self.pops), len(self.pops) + K)
        self.add_population(new_pops, active_new=True, update_era=True)
        self.inactivate_population(pop_i)
        # updates graph
        for i in new_pops_i:
            # updates graph to reflect that the populations are now split
            self.graph[pop_i, i] = 1

    #######################
    #### Lineage Graph ####
    #######################
    def _expand_graph(self):
        '''
        Expands lineage graph to match current number of populations. Can also be used to initialize the graph if it doesn't exist yet.
        '''
        graph = np.zeros((len(self.pops), len(self.pops)), dtype=int)
        # updates graph attribute
        if hasattr(self, 'graph'):
            if len(self.pops) < len(self.graph):
                raise ValueError("List of populations has shrunk! This shouldn't happen.")
            graph[:len(self.graph), :len(self.graph)] = self.graph
        self.graph = graph

    ################
    #### Traits ####
    ################
    def _resolve_population_indices(self, pop_i: Union[int, list] = None) -> list:
        '''
        Resolves population indices, defaulting to the currently active populations.
        '''
        if pop_i is None:
            pop_i = list(self.active_i)
        elif isinstance(pop_i, int):
            pop_i = [pop_i]
        else:
            pop_i = list(pop_i)

        if len(pop_i) == 0:
            raise ValueError('Must select at least one population.')
        for i in pop_i:
            if i < 0 or i >= len(self.pops):
                raise IndexError(f'Population index {i} is out of bounds.')
        return pop_i

    def add_trait(self, name: str, **kwargs):
        '''
        Adds a trait to all active populations in the superpopulation.
        Parameters:
            name (str): Name of the trait to add.
            **kwargs: Arguments analogous to `Population.add_trait()`. The same `effects`
                dictionary is passed to each active population. For other keyword arguments,
                if a Python list is passed, it is assumed to contain one entry per active
                population; otherwise the same value is used for all active populations.
        '''
        effects = kwargs.pop('effects', None)
        if effects is None:
            raise ValueError('SuperPopulation.add_trait requires an effects dictionary.')

        for effect_name, effect in effects.items():
            if isinstance(effect, GeneticEffect) and effect.G_std is None and not effect.force_var:
                warnings.warn(
                    f"Genetic effect {effect_name} has no stored G_std and force_var=False; "
                    "per-allele effects across subpopulations may differ, while standardized effects will be identical."
                )

        for i, pop_i in enumerate(self.active_i):
            pop = self.pops[pop_i]
            pop_kwargs = core.get_pop_kwargs(i, **kwargs)
            pop.add_trait(name=name, effects=effects, **pop_kwargs)

    def add_subpop_trait(self, pop_i: Union[int, list] = None):
        '''
        Adds a permanent trait named `subpop` to selected populations in the
        superpopulation. Each individual receives the index of that population in the
        superpopulation's `pops` list. Existing `subpop` traits are left unchanged.
        '''
        pop_indices = self._resolve_population_indices(pop_i)
        for idx in pop_indices:
            pop = self.pops[idx]
            if 'subpop' in pop.traits:
                continue
            y = np.full(pop.N, idx, dtype=int)
            pop.add_trait_from_fixed_values(name='subpop', y=y, trait_type='permanent')

    ####################
    #### Simulating ####
    ####################
    def simulate_generations(self, verbose: bool = False, **kwargs):
        '''
        Simulates generations for all active populations in the superpopulation. 
        Parameters:
            verbose (bool): Whether to print progress messages for each population and generation. Default is False.
            **kwargs: All arguments are passed to the `simulate_generations()` method of each Population object. See that method for details. For each parameter, if a Python list is passed, it is assumed to be a list of arguments for each population in the superpopulation. If a list isn't passed, it is used for all populations.
        '''
        # iterates through each active population
        for i, pop_i in enumerate(self.active_i):
            pop = self.pops[pop_i]
            # creates kwargs list for each population
            pop_kwargs = {}
            # simulates generations for the population using population-specific kwargs
            pop_kwargs = core.get_pop_kwargs(i, **kwargs)
            pop_kwargs['verbose'] = verbose
            if verbose:
                print(f'Simulating population {pop_i}')
            pop.simulate_generations(**pop_kwargs)

    #######################
    #### Visualization ####
    #######################
    def compute_PCA(self, pop_i: Union[int, list] = None,
                    n_components: int = 2, **kwargs) -> pop.PCAResult:
        '''
        Computes a PCA across one or more populations in the superpopulation.
        Parameters:
            pop_i (int or list): Population indices to include. Defaults to all active
                populations.
            n_components (int): Number of leading PCs to compute.
            **kwargs: Additional arguments passed to `pop.compute_PCA()`.
        Returns:
            pop.PCAResult: PCA result object for the selected individuals.
        '''
        pop_indices = self._resolve_population_indices(pop_i)
        pops_selected = [self.pops[i] for i in pop_indices]

        M = pops_selected[0].M
        P = pops_selected[0].P
        for pop_sel in pops_selected[1:]:
            if pop_sel.M != M or pop_sel.P != P:
                raise ValueError(
                    'All populations included in a joint PCA must share the same M and P.'
                )

        G = np.concatenate([pop_sel.G for pop_sel in pops_selected], axis=0)
        p = pop.compute_freqs(G, P)
        pca = pop.compute_PCA(
            G=G,
            p=p,
            P=P,
            n_components=n_components,
            **kwargs,
        )
        pca.metadata['pop_i'] = pop_indices
        return pca

    def plot_PCA(self, pca: pop.PCAResult = None,
                 pop_i: Union[int, list] = None,
                 pcs: Tuple[int, int] = (1, 2),
                 color_by: str = 'subpop',
                 categorical: bool = None,
                 n_components: int = None,
                 title: str = 'SuperPopulation PCA',
                 **kwargs):
        '''
        Plots a PCA across one or more populations in the superpopulation.
        Parameters:
            pca (pop.PCAResult): Optional pre-computed PCA result.
            pop_i (int or list): Population indices to include. Defaults to all active
                populations.
            pcs (tuple): Two 1-based PCs to plot.
            color_by (str): Trait used to color points. Defaults to `subpop`.
            categorical (bool): Whether to treat `color_by` values as categorical.
                Defaults to treating only `subpop` as categorical unless the trait values
                are non-numeric.
            n_components (int): Number of PCs to compute if `pca` is not provided.
                Defaults to the largest PC index requested in `pcs`.
            title (str): Plot title.
            **kwargs: Additional arguments passed to `pop.plot_PCA()`.
        Returns:
            matplotlib axis: Axis containing the PCA plot.
        '''
        pop_indices = self._resolve_population_indices(pop_i)
        pops_selected = [self.pops[i] for i in pop_indices]

        if color_by == 'subpop':
            self.add_subpop_trait(pop_i=pop_indices)

        if pca is None:
            if n_components is None:
                n_components = max(pcs)
            pca = self.compute_PCA(pop_i=pop_indices, n_components=n_components)

        values = None
        if color_by is not None:
            values = []
            for idx, pop_sel in zip(pop_indices, pops_selected):
                if color_by not in pop_sel.traits:
                    raise ValueError(
                        f"Trait '{color_by}' was not found in population {idx}."
                    )
                values.append(np.asarray(pop_sel.traits[color_by].y))
            values = np.concatenate(values, axis=0)
            if categorical is None and color_by == 'subpop':
                categorical = True

        return pop.plot_PCA(
            pca,
            pcs=pcs,
            values=values,
            categorical=categorical,
            title=title,
            color_label=color_by,
            **kwargs,
        )

    def print_attributes(self, attribute: str, only_active: bool = True):
        '''
        Prints the attributes of all populations in the superpopulation.
        Parameters:
            only_active (bool): Whether to only print attributes of active populations. Default is True.
        '''
        # iterates through each population
        for i, pop in enumerate(self.pops):
            if only_active and not self.active[i]:
                continue
            print(f'Population {i}:')
            print(getattr(pop, attribute, 'Attribute not found.'))

# data types
PedPath = Tuple[int, ...] # stores chain of meioses up/down, modified from [Williams et al. 2025 Genetics]
PedKey2 = Tuple[int,int] # stores pair of individual indices (i,j)
PathSig = namedtuple('PathSig', ['ups', 'downs', 'up_first','up_last', 'up3s', 'down3s']) # signature of a path for hashing

@dataclass
class RelObj:
    '''
    Class for storing relationship object information.
    Attributes:
    '''
    sigs: PathSig # signatures corresponding to this relationship, see extract_signatures()
    degree: Optional[int] = None # degree of relatedness, e.g. 1 for first-degree relatives
    possible_inbreeding: Optional[bool] = None # whether this relationship may involve inbreeding based on the path
    coeff_of_relatedness: Optional[float] = None # expected coefficient of relatedness given degree
    
    # key properties of relationship
    direction: Optional[str] = None # direction of relationship, e.g. 'ancestor', 'descendant'
    full_half: Optional[str] = None # 'full' or 'half' relationship
    parental_line: Optional[str] = None # 'maternal' or 'paternal' or 'both' relationship

    # naming of relationship at different levels
    relationship: Optional[str] = None # name of relationship, broad-level e.g. 'sibling'
    relationship2: Optional[str] = None # name of relationship, specific-level e.g. 'full_sibling'
    relationship3: Optional[str] = None # name of relationship, detailed-level e.g. 'maternal_full_sibling'

class Pedigree:
    def __init__(self, N: int, par_idx: np.ndarray = None, par_Ped: 'Pedigree' = None):
        '''
        Initializes a Pedigree object.
        Parameters:
            N (int): population size of current generation
            par_idx (N*2 array): stores the indices of individual i's two parents. These parental indices should correspond to the parent generation's indices. Default is None (e.g. founding generation).
            par_Ped (Pedigree): pointer to the parent generation's Pedigree object. Default is None (e.g. founding generation).
        '''
        # initializes attributes
        self.N = N # population size
        self.paths: Dict[PedKey2, PedPath] = {} # pairwise relationship path dictionary
        self.degs: Dict[PedKey2, int] = {} # pairwise degree of relatedness dictionary
        self.par_idx = par_idx if par_idx is not None else None # parental indices
        self.par_Ped = par_Ped if par_Ped is not None else None # parental Pedigree object

        # maps PedPath to a canonical PedPath for storage efficiency
        self._paths: Dict[PedPath, PedPath] = {}
        # maps PathSig to Relatinship Object for quick lookup
        self._relobjs: Dict[PedPath, RelObj] = {}
        self.rels: Dict[PedKey2, RelObj] = {} # pairwise relationship object dictionary

        # fills in paths with self-relationships (denoted by () )
        self.fill_paths_self()

    def intern_path(self, path: PedPath) -> PedPath:
        '''
        Return the canonical tuple object for this PedPath. Small wrapper around dict.setdefault for clarity.
        '''
        return self._paths.setdefault(path, path)

    def extend_path(self, path: PedPath, ups: tuple = (), downs: tuple = ()) -> PedPath:
        '''
        Given a path, extends it by adding the specified number of meioses up and down. Also interns the path before returning.
        Parameters:
            path (PedPath): The path to extend.
            ups (tuple): Tuple of meioses to add going up the pedigree (added at beginning of path). Should be positive. Default is (), i.e. no meioses added.
            downs (tuple): Tuple of meioses to add going down the pedigree (added at end of path). Should be negative. Default is (), i.e. no meioses added.
        Returns:
            PedPath: The extended path.
        '''
        # formats tuples properly if length 1 integer is passed
        if isinstance(ups, int):
            ups = (ups,)
        if isinstance(downs, int):
            downs = (downs,)
        extended_path = ups + path + downs
        return self.intern_path(extended_path)
    
    @staticmethod
    def get_closest_path(paths: Dict, keys: Tuple) -> Tuple[Optional[PedPath], List]:
        '''
        Given a dictionary of relationships and a list of keys, returns the closest path among the keys.
        Parameters:
            paths (Dict): Dictionary of relationships (paths attribute in a Pedigree object).
            keys (Tuple): Tuple of keys to check.
        Returns:
            tuple ((closest_path, closest_path_keys)):
            Where:
            - closest_path (PedPath): The closest path found among the keys.
            - closest_path_keys (List): List of keys that correspond to the closest path.
        '''
        closest_path = None # PedPath object
        closest_path_keys = []
        for key in keys:
            path = paths.get(key)
            # skips if pair is unrelated
            if path is None:
                continue
            
            # series of checks to determine if this path is the closest yet
            if closest_path is None:
                closest_path = path
                closest_path_keys = [key]
                continue
            # first check: path cannot be longer than closest path so far
            if len(path) < len(closest_path):
                closest_path = path
                closest_path_keys = [key]
                continue
            elif len(path) > len(closest_path):
                continue
            
            # second check: path cannot have less (+3) entries than closest path so far 
            if path.count(3) > closest_path.count(3):
                closest_path = path
                closest_path_keys = [key]
                continue
            elif path.count(3) < closest_path.count(3):
                continue
            # third check: path cannot have less (-1) entries than closest path so far
            if path.count(-1) > closest_path.count(-1):
                closest_path = path
                closest_path_keys = [key]
                continue
            elif path.count(-1) < closest_path.count(-1):
                continue

            # if all these checks are passed, we set this as the closest path
            # if this path is identical to the previously stored closest path, we append the key
            if path == closest_path:
                closest_path_keys.append(key)
            else:
                closest_path = path
                closest_path_keys = [key]
            
        return closest_path, closest_path_keys
    
    def reverse_path(self, path: PedPath) -> PedPath:
        '''
        Reverse a PedPath (flip order and sign) and intern it. Example: (2,-3,-1) -> (1,3,-2). Also interns the path before returning.
        Parameters:
            path (PedPath): The path to reverse.
        Returns:
            PedPath: The reversed path.
        '''
        reversed_path = tuple(-step for step in path[::-1])
        return self.intern_path(reversed_path)

    def fill_paths_self(self):
        '''
        Fills in the paths dictionary with self-relationships only.
        '''
        self_path = self.intern_path( () ) # empty path for self
        for i in range(self.N):
            self.paths[(i,i)] = self_path

    def construct_paths(self):
        '''
        Constructs a dictionary with the relationship paths between every related individual in the current population. Pedigree object must have par_idx and par_Ped attributes. See __init__(). For individual i and j, the shortest relationship path between the 4 pairs of parents of i and j is used, and then extended by one meiosis up and down to get the relationship path between i and j. This can erase inbreeding information in the current generation. Only entries i>j are explicitly computed, since the reverse relationships are automatically filled in.
        '''
        # checks to ensure par_idx and par_Ped are in the object
        if self.par_idx is None or self.par_Ped is None:
            raise ValueError("par_idx and par_Ped must be attributed of the object to construct paths.")
        # iterates over all pairs of individuals (i > j)
        for i in range(self.N):
            for j in range(i+1, self.N):
                i_pars = np.array(self.par_idx[i, :])  # (mom ID, dad ID)
                j_pars = np.array(self.par_idx[j, :])  # (mom ID, dad ID)
                # gets shortest path and the keys that produced it
                keys = [ (i_pars[k], j_pars[l]) for k in (0,1) for l in (0,1) ]
                pargen_path, pargen_keys = self.get_closest_path(self.par_Ped.paths, keys)

                # if all parent pairs are unrelated, then so is the offspring pair
                if pargen_path is None:
                    continue
                # determines which parent(s) (0 or 1) for i and j produced the closest path
                i_par_closest = [(np.where(i_pars == i_k)[0][0]) for (i_k,j_l) in pargen_keys]
                j_par_closest = [(np.where(j_pars == j_l)[0][0]) for (i_k,j_l) in pargen_keys]
                if 0 in i_par_closest and 1 in i_par_closest:
                    up_sex =  3 # if both parents of i have a closest path, use +3 encoding
                else:
                    up_sex = i_par_closest[0] + 1 # otherwise, use the parent that produced the closest path (incremented by 1 for encoding)
                if 0 in j_par_closest and 1 in j_par_closest:
                    down_sex = -3 # if both parents of j have a closest path, use -3 encoding
                else:
                    down_sex = -(j_par_closest[0] + 1) # otherwise, use the parent that produced the closest path (incremented by 1 and then flipped sign for encoding)

                # extends closest parental path by adding sex-encoded up/down meioses
                path_ij = self.extend_path(pargen_path, ups=(up_sex,), downs=(down_sex,) )
                # passes paths through intern_path() to ensure canonical storage
                path_ij = self.intern_path(path_ij)
                # constructs the reverse relationship of j to i
                path_ji = self.reverse_path(path_ij) # (path is interned inside function)

                # stores both relationships
                self.paths[(i,j)] = path_ij
                self.paths[(j,i)] = path_ji

        # handles self-relationships
        self.fill_paths_self()

    def extract_signatures(self, path: PedPath) -> PathSig:
        '''
        Extracts a signature from a PedPath for hashing purposes. The signature consists of the number of ups, number of downs, whether the first step is up or down, and whether the last step is up or down.
        Parameters:
            path (PedPath): The path to extract the signature from.
        Returns:
            PathSig: The signature of the path.
        '''
        positives = [step for step in path if step > 0]
        negatives = [step for step in path if step < 0]
        # extract number of up meioses (i.e. positive entries)
        ups = len(positives)
        # extract number of down meioses (i.e. negative entries)
        downs = len(negatives)
        # extract the value of the first positive step
        up_first = positives[0] if ups > 0 else -9
        # extract the value of the last positive step
        up_last = positives[-1] if ups > 0 else -9 

        # extract number of +3s and -3s
        up3s = positives.count(3)
        down3s = negatives.count(-3)

        # returns PathSig namedtuple
        return PathSig(ups=ups, downs=downs, up_first=up_first, up_last=up_last, up3s=up3s, down3s=down3s)
    
    def path_to_relationship(self, path: PedPath) -> RelObj:
        '''
        Takes a PedPath object and converts it to a relationship object (RelObj) based on predefined signature rules.
        Parameters:
            path (PedPath): The path to convert to a relationship object.
        Returns:
            RelObj: The relationship object corresponding to the path.
        '''
        # extracts signatures of path
        sigs = self.extract_signatures(path)
        # begins building relationship object
        rel_obj = RelObj(sigs=sigs)

        # loops through each attribute of a relationship (i.e. general descriptors)
        for rel_attribute in REL_TYPES.keys():
            # loops through each type within that attribute
            for rel_type, rel_info in REL_TYPES[rel_attribute].items():
                sigs_match = True
                # loops through each signature constraint of that type
                for sig_key, sig_value in rel_info['sigs'].items():
                    attr_value = getattr(sigs, sig_key)
                    # if the ruleset only has a single value, it's converted to a list
                    if isinstance(sig_value, int):
                        sig_value = [sig_value]
                    # if list, treated like set of values
                    if isinstance(sig_value, list):
                        if attr_value not in sig_value:
                            sigs_match = False
                            break
                    # if tuple, is treated like a range
                    if isinstance(sig_value, tuple) and len(sig_value) == 2:
                        if attr_value < sig_value[0] or attr_value > sig_value[1]:
                            sigs_match = False
                            break
                # assigns relationship attribute if signatures match
                if sigs_match:
                    setattr(rel_obj, rel_attribute, rel_type)
                    break # each attribute can only have one type
        
        # builds detailed relationship names (e.g. full vs half and maternal vs paternal)
        # by default, relationship2 is the same as relationship
        rel_obj.relationship2 = rel_obj.relationship
        # determines if full_half is relevant
        fh = True
        if 'fh' in REL_TYPES['relationship'][rel_obj.relationship]:
            fh = REL_TYPES['relationship'][rel_obj.relationship]['fh']
        # if full_half is relevant, adds it to relationship2
        if fh:
            rel_obj.relationship2 = rel_obj.full_half + '_' + rel_obj.relationship
        # by default, relationship3 is the same as relationship2
        rel_obj.relationship3 = rel_obj.relationship2
        # if relationship is either half or parental, adds parental_line to relationship3
        if rel_obj.parental_line == 'maternal' or rel_obj.parental_line == 'paternal':
            rel_obj.relationship3 = rel_obj.parental_line + '_' + rel_obj.relationship2

        # computes degree

        # when up3s and down3s are unequal, inbreeding may be involved, and degree may be inaccurate
        if sigs.up3s != sigs.down3s:
            rel_obj.possible_inbreeding = True
        else:
            rel_obj.possible_inbreeding = False
        # NOTE: For certain inbred pedigrees, the paths cannot unambiguously distinguish them, and thus the degree of relatedness may be incorrect. An example is double-first cousins vs two individuals whose 4 parents are all full-siblings with each other. Both have a path of [+3 +3 -3 -3], but the former is degree 2, and the latter is degree 1. This method will classify both as degree 2, and will not flag it as possible inbreeding. Such cases are rare in practice, however.
        rel_obj.degree = int( sigs.ups + sigs.downs - (sigs.up3s + sigs.down3s)*0.5 )

        # stores expected genome-wide coefficient of relatedness
        rel_obj.coeff_of_relatedness = 2**(-rel_obj.degree)

        return rel_obj
    
    def assign_relationships(self):
        '''
        Fills in the 'rels' dictionary with the relationship objects between every related individual in the current population. Uses the paths attribute to determine relationships, which means the 'paths' dictionary must be filled in first. See construct_paths().
        '''
        self._relobjs = {path: self.path_to_relationship(path) for path in self._paths.keys()}
        self.rels = {key: self._relobjs[path] for key, path in self.paths.items()}

    # small class to allow pretty printing of results of count_relationships() to console
    class CountRelDict:
        def __init__(self, data):
            self.data = data
        def __str__(self):
            lines = [
                f"{str(k):20} {v['count']}"
                for k, v in self.data.items()
            ]
            return "\n".join(lines)
        def __repr__(self):
            return f"{self.__str__()}"

    def count_relationships(self, attribute: list = ['relationship'], idx: int = None) -> Dict[str, int]:
        '''
        Counts the number of each type of relationship, or any other specified attribute of the RelObj, in the 'rels' dictionary.
        Parameters:
            attribute (list): A list of attributes of the RelObj to summarize, where each unique combination of the provided attributes is summarized. If there is a period inside the attribute name, it is interpreted as accessing a nested attribute. Default is ['relationship'].
            idx (int): If specified, only counts relationships involving individual idx. Default is None, meaning all relationships are counted.
        Returns:
            rel_summary (Dict): Dictionary summarizing the number of each type of relationship.
        '''
        rel_summary: Dict[str, int] = {}
        for rel_key, rel_obj in self.rels.items():
            # if idx is specified, only counts relationships involving individual idx
            # only check first slot since the relationship of B to A is stored under (A,B)
            if idx is not None and rel_key[0] != idx: 
                continue
            
            # checks if only a single attribute is provided
            if isinstance(attribute, str):
                attribute = [attribute]
            attr_names = []
            for attr in attribute:
                # iterates through attr if nested attribute
                attrs = attr.split('.')
                val = rel_obj
                for a in attrs:
                    val = getattr(val, a)
                attr_names.append( val )
            
            # stores attributes as a tuple if combinations of >1 are used for summary
            if len(attr_names) == 1:
                attr_names = attr_names[0]
            else:
                attr_names = tuple(attr_names)
            if attr_names in rel_summary:
                rel_summary[attr_names]['count'] += 1
                rel_summary[attr_names]['keys'].append( (rel_key) )
            else:
                rel_summary[attr_names] = {'count': 1,
                                         'keys': [(rel_key)]}
        # sorts keys
        rel_summary = dict(sorted(rel_summary.items()))
        return self.CountRelDict(rel_summary)
    
    @staticmethod
    def summarize_per_relationship(count_rel_dict: 'Pedigree.CountRelDict', data: np.ndarray, summary_stats: list = ['mean', 'std', 'min', 'max', 'count']) -> Dict[str, Dict]:
        '''
        Summarizes a specified data value of the inputted data matrix/dictionary per relationship type.
        Parameters:
            count_rel_dict (Pedigree.CountRelDict): The output of count_relationships() method.
            data (np.ndarray): Some object that when indexed by the keys provided in count_rel_dict[relationship]['keys'] returns some value which is to be summarized.
            summary_stats (list): List of summary statistics to compute. Options are 'mean', 'std', 'min', 'max', and 'count'. Default is all five.
        Returns:
            rel_attribute_summary (Dict): Dictionary summarizing the specified attribute per relationship type. Mean, std, min, max, and count are provided.
        '''
        rel_attribute_summary: Dict[str, Dict] = {}
        for rel_type, rel_info in count_rel_dict.data.items():
            values = []
            for key in rel_info['keys']:
                if isinstance(data, dict):
                    if key not in data:
                        continue
                value = data[key]
                values.append(value)
            if len(values) == 0:
                continue
            values = np.array(values)
            summary_dict = {}
            for summary in summary_stats:
                if summary == 'mean':
                    summary_dict['mean'] = np.mean(values)
                elif summary == 'std':
                    summary_dict['std'] = np.std(values)
                elif summary == 'min':
                    summary_dict['min'] = np.min(values)
                elif summary == 'max':
                    summary_dict['max'] = np.max(values)
                elif summary == 'count':
                    summary_dict['count'] = len(values)
            rel_attribute_summary[rel_type] = summary_dict
        return rel_attribute_summary

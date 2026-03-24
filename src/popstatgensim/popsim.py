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
from scipy.linalg import block_diag
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
        self._initialize_H(H, R_type=R_type, keep_past_generations=keep_past_generations, track_pedigree=track_pedigree)
    
    @classmethod
    def from_H(cls, H: np.ndarray, R_type: str = 'blocks', keep_past_generations: int = 1, track_pedigree: bool = False):
        '''
        Initializes a population from a given haplotype array.
        Parameters:
            H (3D array): N*M*P array of haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
            R_type (str). Type of recombination rates to use for genome. Options are: 'blocks' (default) for LD blocks (see pop.generate_LD_blocks()), 'indep' for independent sites (see self.make_sites_indep()), or 'uniform' for uniform recombination rates across genome (see pop.generate_chromosomes()). Resulting recombination rate array is stored in Population.R.
            keep_past_generations (int): Number of past generations to keep in the object. Default is 1, meaning the past generation is kept (on top of the current generation).
            track_pedigree (bool): Whether to track pedigree information (stored in Population.ped). Must keep at least 1 past generation. Default is False.
        Returns:
            Population: A new Population object initialized with the given haplotype array.
        '''
        # creates new instance of class
        pop = cls.__new__(cls)
        # initializes the object with the given haplotype array
        pop._initialize_H(H, R_type=R_type, keep_past_generations=keep_past_generations, track_pedigree=track_pedigree)
        return pop

    def _initialize_H(self, H: np.ndarray, R_type: str = 'blocks', keep_past_generations: int = 1, track_pedigree: bool = False):
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
        # recombination rates
        if R_type == 'blocks':
            self.R = pop.generate_LD_blocks(self.M)
        elif R_type == 'indep':
            self.make_sites_indep()
        elif R_type == 'uniform':
            self.R = pop.generate_chromosomes(self.M, chrs=1, meioses_per_chr=1)
        self.K = np.diag(np.ones(self.N)) # kinship matrix (initially identity, not functional yet)
        self.track_pedigree = track_pedigree # whether to track pedigree information
        # how many past generations to keep in memory
        self.past = [self]
        self.update_keep_past_gens(keep_past_generations=keep_past_generations)
        Haplos = np.full_like(H, -1, dtype=int) # initializes haplotype ID array with -1s

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
        ids = np.arange(self.N * self.P, dtype=np.int32).reshape(self.N, self.P)
        Haplos = np.broadcast_to(ids[:, None, :], self.H.shape).copy()
        self._update_obj(Haplos=Haplos)


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
            self.p = pop.compute_freqs(self.G, self.P)
            self.X = pop.standardize_G(self.G, self.p, self.P, impute=True, std_method='observed')
        if Haplos is not None:
            self.Haplos = Haplos
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
                self.relations['spouses'] = np.zeros((self.N, self.N), dtype=np.uint8) # resets spouses relationship matrix for current gen
            else:
                self.relations[key] = relations[key] # sets other relationship matrices for current generation

    def _update_pedigree(self):
        '''
        Updates the pedigree information in the population object for a new generation.
        '''
        if self.keep_past_generations < 1:
            raise Exception('Must keep at least 1 past generation to track pedigree.')
        # makes new Pedigree object from scratch, and adds parent infices and Pedigree pointer
        self.ped = Pedigree(self.N, par_idx = self.relations['par_idx'], par_Ped = self.past[1].ped)
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

    def get_RDR_SNP_GRMs(self) -> list:
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

        G_par = self.get_Gpar()
        p_par = G_par.mean(axis=0) / (2 * self.P)
        X_par = pop.standardize_G(G_par, p_par, 2 * self.P, impute=True,
                                  std_method='observed', target_var=2.0)

        M = G_o.shape[1]
        R_oo = pop.compute_GRM(X_o)
        R_pp = pop.compute_GRM(X_par) / 2
        R_op = (X_o @ X_par.T + X_par @ X_o.T) / (2 * M)

        return [R_oo, R_pp, R_op]

    def add_trait(self, name: str, seed: int = None, **kwargs):
        '''
        Initializes and generates trait. See Trait.__init__ for details.
        Parameters:
            name (str): Name of trait.
            seed (int): Seed for random number generation.
            **kwargs: All other arguments are passed to the Trait constructor. See Trait.__init__ for details.
        '''
        if seed is not None:
            np.random.seed(seed)

        if 'var_Gpar' in kwargs:
            kwargs['var_A_par'] = kwargs.pop('var_Gpar')
        if 'Gpar' in kwargs:
            kwargs['G_par'] = kwargs.pop('Gpar')
        if 'var_A_par' in kwargs and kwargs['var_A_par'] != 0 and 'G_par' not in kwargs:
            kwargs['G_par'] = self.get_Gpar()
        trait = Trait(self.G, **kwargs)
        trait.pop = self
        self.traits[name] = trait
    
    def add_trait_from_effects(self, name: str, **kwargs):
        '''
        Initializes and generates trait from specified effects.
        Parameters:
            name (str): Name of trait.
            **kwargs: All other arguments are passed to the Trait.from_effects() constructor.
        '''
        if 'Gpar' in kwargs:
            kwargs['G_par'] = kwargs.pop('Gpar')
        G_input = kwargs.pop('G', None)
        effects = kwargs.get('effects')
        if G_input is None:
            if isinstance(effects, dict) and 'A_par' in effects:
                G_input = {'A': self.G, 'A_par': self.get_Gpar()}
            else:
                G_input = self.G
        trait = Trait.from_effects(G_input, **kwargs)
        trait.pop = self
        self.traits[name] = trait
    
    def add_trait_from_fixed_values(self, name: str, y: np.ndarray):
        '''
        Initializes and generates trait from specified fixed trait values. No sub component or effects (e.g. genetic, random effects) are stored.
        Parameters:
            name (str): Name of trait.
            y (1D array): Trait values for each individual in the population.
        '''
        trait = Trait.from_fixed_values(y)
        trait.pop = self
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
                trait_inputs = {'G': self.G}
                if 'A_par' in trait.effects:
                    trait_inputs['G_par'] = self.get_Gpar()
                trait.update_inputs(**trait_inputs)
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
            'values': None # the values of the metric over generations, initialized to None
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

    def _update_temp_metrics(self, t: int, **kwargs):
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
        
    def _update_metric_history(self):
        '''
        Updates the metric history for each active metric by appending the temporary metric values to the permanent metric values stored in the object.
        '''
        for metric_name in self.metric:
            if self.metric[metric_name]['active']:
                # gets temporary metric values
                temp_values = self.metric[metric_name]['temp']
                # appends to permanent values
                if self.metric[metric_name]['values'] is None:
                    self.metric[metric_name]['values'] = temp_values
                else:
                    self.metric[metric_name]['values'] = np.concatenate((self.metric[metric_name]['values'], temp_values), axis=0)
                # clears temporary metric values
                self.metric[metric_name].pop('temp', None)

    def _initialize_metrics(self, **kwargs):
        '''
        Initializes the metric history for each active metric by computing metrics for the current (starting) generation and storing them in the metric values. Should be called after defining metrics with `_define_metric()`.
        Parameters:
            **kwargs: All extra arguments that are passed to any metric function. Only the parameters needed for each metric function are passed to that function. The pre-specified arguments set in `_define_metric()` are automatically passed to each metric function.
        '''
        self._prep_metrics(1)
        self._update_temp_metrics(0, **kwargs)
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

    def simulate_generations(self, generations: int, related_offspring: bool = False,
                             trait_updates: bool = False, **kwargs):
        '''
        Simulates specified number of generations beyond current generation. Can simulate offspring directly. Automatically updates object. Recombination rates are extracted from object attributes.

        Parameters:
            generations (int): Number of generations to simulate (beyond the current generation).
            related_offspring (bool): Whether the offspring of the next generation should be directly related to parents from previous generation by simulating meiosis and haplotype transfer. Default is False, meaning that future offspring have haplotypes drawn randomly from allele frequencies.
            trait_updates (bool): Whether to update traits after each generation. However, sex is always updated in each generation, regardless of this setting. Default is False, meaning that traits are only updated at the end of the simulation.
            **kwargs: All other arguments are passed to the `next_generation` or `generate_offspring` methods. See those methods for details.
        '''
        # preps metrics for new generations
        self._prep_metrics(generations)
        previous_gens = self.metric['p']['values'].shape[0]
        
        # loops through each generation
        for t in range(generations):
            if related_offspring:
                (H, relations, Haplos) = self.generate_offspring(**kwargs)
            else:
                H = self.next_generation(**kwargs)
                Haplos = np.full_like(H, -1, dtype=int) # unrelated haplotypes
                relations = None
            # updates objects and past
            self._update_obj(H=H, update_past=True, relations=relations, Haplos=Haplos, update_pedigree=self.track_pedigree)
            # records metrics
            self._update_temp_metrics(t, G=self.G)
            if trait_updates:
                self.update_traits()
            else:
                self.update_traits(traits=['sex']) # always updated in each generation
        self.T_breaks.append(previous_gens + generations)
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
        rel_spouses = np.zeros((self.N,self.N), dtype=bool) # relationship matrix for spouses

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
        # updates relationship matrix for spouses
        rel_spouses[iMs, iPs] = 1
        rel_spouses[iPs, iMs] = 1

        return ((iMs, iPs), rel_spouses)

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
        
        spouses = self.relations['spouses'].astype(bool)
        i_spouse, j_spouse = np.where(np.triu(spouses, k=1))

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
        if not hasattr(self, 'past') or self.past is None or len(self.past) < 2 or self.past[1] is None:
            raise Exception('Previous generation not available. Make sure `pop.past[1]` exists before calling `get_Gpar()`.')
        if 'parents' not in self.relations:
            raise Exception("Parent relationship matrix not found in object. Make sure `relations['parents']` exists.")

        parents = self.relations['parents']
        G_prev = self.past[1].G

        if parents.shape != (self.N, self.past[1].N):
            raise Exception('Parent relationship matrix has incompatible shape for previous generation.')

        parent_counts = parents.sum(axis=1)
        if not np.all(parent_counts == 2):
            raise Exception('Each individual must have exactly two recorded parents to compute `Gpar`.')

        return (parents @ G_prev)

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
        (iMs, iPs), rel_spouses = self._pair_mates(**kwargs)

        # Selection ####
        if isinstance(s, (float, int)):
            s = np.full(self.M, s)
        # computes individuals' breeding weight (assumes linear additive fitness effects)
        W = np.exp( (np.log(1 + self.G * s[None, :])).sum(axis=1) )
        # computes each pair's breeding weight
        W_pair = W[iMs] * W[iPs]
        # computs probability of each parent pair being chosen to mate for one offspring
        P_mate = W_pair / W_pair.sum()
        # determines population size of next generation (currently maintains population size)
        N_offspring = self.N
        # draws indices of parents for each offspring
        i_mate = np.random.choice(np.arange(len(iMs)), size=N_offspring, p=P_mate)
        parents = np.stack((iMs[i_mate], iPs[i_mate]), axis=1)
        # makes relationship matrix for full siblings
        M_mate = np.zeros((N_offspring, len(iMs)), dtype=bool)
        M_mate[np.arange(N_offspring), i_mate] = 1
        rel_fullsibs = M_mate @ M_mate.T
        np.fill_diagonal(rel_fullsibs, 0)

        # Drift + Recombination ####
        # generates variants for which a crossover event happens for each parent of each offspring
        crossover_events = np.random.binomial(n=1, p=self.R.reshape(1, self.M, 1),
                                              size=(N_offspring, self.M, 2))
        # determines the shift in haplotype phase for each parent's haplotype at each variant
        haplo_phase = np.cumsum(crossover_events, axis=1)
        # randomly chooses haplotype to start with for each parent of each offspring
        haplo_k0 = np.random.binomial(n=1, p=1/self.P, size = (N_offspring ,2))
        # adds the starting shift and gets the modulo so that haplotype phase acts as an index
        haplo_ks = (haplo_k0[:, None, :] + haplo_phase) % 2

        rel_parents = np.zeros((N_offspring, self.N), dtype=bool) # relationship matrix for parent-child relationships
        par_idx = np.full((N_offspring, 2), -1, dtype=int) # initializes parent index array
        H = np.empty((N_offspring, self.M, self.P), dtype=int)
        Haplos = np.full_like(H, -1) # initializes haplotype ID array with -1s
        for i in np.arange(N_offspring):
            iM = parents[i,0]
            iP = parents[i,1]
            # extract allele from correct haplotype of each parent
            haploM = self.H[iM, np.arange(self.M), haplo_ks[i, :, 0]]
            haploP = self.H[iP, np.arange(self.M), haplo_ks[i, :, 1]]
            haplos = np.stack((haploM, haploP), axis = 1)
            # shuffles haplotypes around. [EDIT]: nevermind, want to keep consistent maternal vs paternal order
            # chr_order = np.random.choice(self.P, size=self.P, replace=False)
            chr_order = np.arange(self.P)
            haplos = haplos[:,chr_order]
            H[i,:,:] = haplos

            # does same inheritance for haplotype IDs (instead of allele dosages)
            # (yes, I know it's confusing that I am using H/haplo to refer to both)
            HaploM = self.Haplos[iM, np.arange(self.M), haplo_ks[i, :, 0]]
            HaploP = self.Haplos[iP, np.arange(self.M), haplo_ks[i, :, 1]]
            Haplos_i = np.stack((HaploM, HaploP), axis = 1)

            Haplos_i = Haplos_i[:,chr_order]
            Haplos[i,:,:] = Haplos_i

            # updates relationship matrix for parent-child relationships, and stores parent indices
            rel_parents[i, iM] = 1
            rel_parents[i, iP] = 1
            par_idx[i, :] = [iM, iP] # mom is always col 0, dad is always col 1

        # Mutations ####
        if isinstance(mu, (float, int)):
            mu = np.full(self.M, mu)
        mutations = np.random.binomial(n=1, p=mu[None,:,None], size = (N_offspring, self.M, self.P))
        # Apply mutations by flipping alleles (0 to 1 or 1 to 0) based on the mutation matrix
        H = (H + mutations) % 2

        # updates relationship matrix for parent-child relationships
        relations = {'spouses': rel_spouses.astype(np.uint8), # occupies more space than a bool, but cleaner to see
                     'parents': rel_parents.astype(np.uint8),
                     'par_idx': par_idx, # stores actual indices of parents
                     'full_sibs': rel_fullsibs.astype(np.uint8),
                     'household': M_mate.astype(np.uint8)}

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
        spop.join_populations(pops_i, shared_haplotypes=True)
        new_pop = spop.pops[-1] # the last population in the SuperPopulation is the combined one
        new_pop.keep_past_generations = self.keep_past_generations
        new_pop.relations = pop.initialize_relations(new_pop.N)
        del new_pop.relations['parents'] # default parents matrix is not accurate here

        # updates relationship matrices in combined population
        # full_sibs
        full_sibs = block_diag(*[pop.relations['full_sibs'] for pop in pops])
        # spouses
        spouses = block_diag(*[pop.relations['spouses'] for pop in pops])
        # parent-child
        parent_child = np.zeros((new_pop.N, new_pop.N), dtype=np.uint8)
        Ns_cumsum = np.cumsum([0] + Ns)
        for gen in range(generations):
            gen_parents = pops[gen].relations['parents']
            i_start = Ns_cumsum[gen]
            i_end = Ns_cumsum[gen + 1]
            j_start = Ns_cumsum[gen + 1]
            j_end = Ns_cumsum[gen + 2]
            parent_child[i_start:i_end, j_start:j_end] = gen_parents
            parent_child[j_start:j_end, i_start:i_end] = gen_parents.T

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
        new_pop.relations['parent_child'] = parent_child

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
                    par_inds = self.past[gen].relations['par_idx'][ind, :]
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
        # plots all generations if not specified
        if last_generations is None:
            t_start = 0
        else:
            t_start = max(0,ps.shape[0] - last_generations)
        t_keep = tuple( range(t_start, ps.shape[0]))
        # subsets to specified variants
        ts = np.arange(t_start, ps.shape[0])
        ps = ps[np.ix_(t_keep, j_keep)]
        
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

    def generate_component(self, inputs: dict) -> np.ndarray:
        '''
        Generates realized values for this component from the provided inputs.
        '''
        raise NotImplementedError


class GeneticEffect(Effect):
    '''
    Stores the effect sizes for a single genetic component of a trait.
    '''
    VALID_NAMES = {'A', 'A_par'}

    def __init__(self, name: str, effects: np.ndarray, G: np.ndarray = None,
                 G_std: np.ndarray = None, per_allele: bool = False):
        if name not in self.VALID_NAMES:
            raise ValueError(f"Unknown genetic effect name: {name}")
        super().__init__(name)

        effects = np.asarray(effects, dtype=float)
        if effects.ndim != 1:
            raise ValueError('effects must be a 1D array.')

        self.input_name = 'G' if name == 'A' else 'G_par'
        self.g_std_input_name = 'G_std' if name == 'A' else 'G_par_std'
        self.required_inputs = (self.input_name, self.g_std_input_name)

        if G_std is None:
            if G is None:
                raise ValueError('Must provide either G or G_std when initializing GeneticEffect.')
            G_std = stat.get_G_std_for_effects(G, P=int(G.max()) if G.size > 0 else None)
        else:
            G_std = np.asarray(G_std, dtype=float)
            if G_std.ndim != 1:
                raise ValueError('G_std must be a 1D array.')
        self.G_std = G_std.copy()

        if self.G_std.shape[0] != effects.shape[0]:
            raise ValueError('Length of effects must match length of G_std.')

        self.set_effects(effects, per_allele=per_allele)

    def set_effects(self, effects: np.ndarray, per_allele: bool = False):
        '''
        Stores effects in both standardized and per-allele units.
        '''
        effects = np.asarray(effects, dtype=float)
        if effects.ndim != 1:
            raise ValueError('effects must be a 1D array.')
        if effects.shape[0] != self.G_std.shape[0]:
            raise ValueError('Length of effects must match length of G_std.')

        if per_allele:
            self.effects_per_allele = effects.copy()
            self.effects = stat.get_standardized_effects(
                self.effects_per_allele, self.G_std, std2allelic=False
            )
        else:
            self.effects = effects.copy()
            self.effects_per_allele = stat.get_standardized_effects(
                self.effects, self.G_std, std2allelic=True
            )

        self.j_causal = np.where(self.effects != 0)[0]
        self.M_causal = len(self.j_causal)

    def refresh_from_inputs(self, inputs: dict):
        '''
        Refreshes the standardized effects from the current genotype standard deviations.
        '''
        if self.g_std_input_name not in inputs:
            raise ValueError(f"Missing required input '{self.g_std_input_name}' for effect {self.name}.")
        G_std = np.asarray(inputs[self.g_std_input_name], dtype=float)
        if G_std.ndim != 1:
            raise ValueError(f"Input '{self.g_std_input_name}' must be a 1D array.")
        if G_std.shape[0] != self.effects_per_allele.shape[0]:
            raise ValueError(f"Input '{self.g_std_input_name}' has incompatible length for effect {self.name}.")
        self.G_std = G_std.copy()
        self.effects = stat.get_standardized_effects(
            self.effects_per_allele, self.G_std, std2allelic=False
        )

    def generate_component(self, inputs: dict) -> np.ndarray:
        '''
        Computes realized genetic values from the current genotype input.
        '''
        if self.input_name not in inputs:
            raise ValueError(f"Missing required input '{self.input_name}' for effect {self.name}.")
        G = np.asarray(inputs[self.input_name])
        if G.ndim != 2:
            raise ValueError(f"Input '{self.input_name}' must be a 2D array.")
        if G.shape[1] != self.effects.shape[0]:
            raise ValueError(f"Input '{self.input_name}' has incompatible number of variants for effect {self.name}.")
        return stat.compute_genetic_value(G, self.effects_per_allele)


class NoiseEffect(Effect):
    '''
    Stores the generation rule for a noise component.
    '''
    VALID_NAMES = {'Eps'}
    required_inputs = ('N',)

    def __init__(self, name: str = 'Eps', var: float = 0.0):
        if name not in self.VALID_NAMES:
            raise ValueError(f"Unknown noise effect name: {name}")
        super().__init__(name)
        self.var = var

    def generate_component(self, inputs: dict) -> np.ndarray:
        '''
        Generates realized noise values for N individuals.
        '''
        if 'N' not in inputs:
            raise ValueError("Missing required input 'N' for effect Eps.")
        return stat.generate_noise_value(int(inputs['N']), self.var)


class Trait:
    '''
    Class for a trait belonging to a Population object.
    '''
    VALID_GENETIC_COMPONENTS = {'A', 'A_par'}
    VALID_EFFECT_COMPONENTS = {'A', 'A_par', 'Eps'}
    DERIVED_INPUTS = {'G_std', 'G_par_std'}

    def __init__(self, G: np.ndarray, M_causal: int = None, dist: str = 'normal',
                 var_A: float = 1.0, var_Eps: float = 0.0,
                 var_A_par: float = 0.0, G_par: np.ndarray = None):
        '''
        Initializes and generates a trait from variance components.
        '''
        self._initialize_empty()

        effects_A, _ = stat.generate_causal_effects(G.shape[1], M_causal, var_A, dist)
        self.effects['A'] = GeneticEffect('A', effects_A, G=G)
        self.var_initial['A'] = var_A

        if var_A_par != 0:
            if G_par is None:
                raise ValueError('Must provide G_par when var_A_par is non-zero.')
            if var_A <= 0:
                raise ValueError('var_A must be positive to construct correlated A_par effects.')
            effects_A_par = self.effects['A'].effects * np.sqrt(var_A_par / var_A)
            self.effects['A_par'] = GeneticEffect('A_par', effects_A_par, G=G_par)
            self.var_initial['A_par'] = var_A_par

        self.effects['Eps'] = NoiseEffect('Eps', var=var_Eps)
        self.var['Eps'] = var_Eps
        self.var_initial['Eps'] = var_Eps

        init_inputs = {'G': G, 'G_std': self.effects['A'].G_std}
        if G_par is not None:
            init_inputs['G_par'] = G_par
        if 'A_par' in self.effects:
            init_inputs['G_par_std'] = self.effects['A_par'].G_std
        self.update_inputs(**init_inputs)
        self.generate_trait()

    @classmethod
    def _coerce_effects_dict(cls, effects: Union[np.ndarray, dict]) -> dict:
        if isinstance(effects, dict):
            effects_dict = {key: np.asarray(value, dtype=float) for key, value in effects.items()}
        else:
            effects_dict = {'A': np.asarray(effects, dtype=float)}

        invalid = set(effects_dict) - cls.VALID_GENETIC_COMPONENTS
        if invalid:
            raise ValueError(f'Unknown genetic effect names: {sorted(invalid)}')
        for key, value in effects_dict.items():
            if value.ndim != 1:
                raise ValueError(f'Effects for {key} must be a 1D array.')
        return effects_dict

    @classmethod
    def _coerce_genotype_dict(cls, G: Union[np.ndarray, dict] = None, G_par: np.ndarray = None) -> dict:
        G_dict = {}
        if G is not None:
            if isinstance(G, dict):
                G_dict = {key: np.asarray(value) for key, value in G.items()}
            else:
                G_dict = {'A': np.asarray(G)}

        if G_par is not None:
            G_dict['A_par'] = np.asarray(G_par)

        invalid = set(G_dict) - cls.VALID_GENETIC_COMPONENTS
        if invalid:
            raise ValueError(f'Unknown genotype input names: {sorted(invalid)}')
        for key, value in G_dict.items():
            if value.ndim != 2:
                raise ValueError(f'Genotype matrix for {key} must be a 2D array.')
        return G_dict

    @staticmethod
    def _coerce_per_allele(per_allele: Union[bool, dict], keys: set) -> dict:
        if isinstance(per_allele, bool):
            return {key: per_allele for key in keys}
        if isinstance(per_allele, dict):
            invalid = set(per_allele) - set(keys)
            if invalid:
                raise ValueError(f'Unknown per_allele keys: {sorted(invalid)}')
            return {key: bool(per_allele.get(key, False)) for key in keys}
        raise ValueError('per_allele must be a bool or a dictionary.')

    def _initialize_empty(self):
        self.pop = None
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

        if N_candidates:
            N_current = int(N_candidates[0])
            if any(n != N_current for n in N_candidates):
                raise ValueError('Trait inputs have inconsistent numbers of individuals.')
            if 'N' in self.inputs and int(self.inputs['N']) != N_current:
                raise ValueError("Trait input 'N' is inconsistent with the provided matrices.")
            self.inputs['N'] = N_current
        elif 'N' in self.inputs:
            self.inputs['N'] = int(self.inputs['N'])
        else:
            raise ValueError("Trait input 'N' must always be available.")

    def update_inputs(self, inputs: dict = None, **kwargs):
        '''
        Updates the stored inputs for the trait and refreshes any derived inputs.
        '''
        new_inputs = {} if inputs is None else copy.deepcopy(inputs)
        new_inputs.update(kwargs)

        if 'G' in new_inputs or 'A' in new_inputs or 'A_par' in new_inputs:
            G_dict = self._coerce_genotype_dict(new_inputs.pop('G', None), G_par=new_inputs.pop('G_par', None))
            if 'A' in new_inputs:
                G_dict['A'] = np.asarray(new_inputs.pop('A'))
            if 'A_par' in new_inputs:
                G_dict['A_par'] = np.asarray(new_inputs.pop('A_par'))
            if 'A' in G_dict:
                new_inputs['G'] = G_dict['A']
            if 'A_par' in G_dict:
                new_inputs['G_par'] = G_dict['A_par']

        self.inputs.update(new_inputs)
        self._refresh_derived_inputs()

    def _update_empirical_variances(self):
        self.var = {name: values.var() for name, values in self.y_.items()}

    def _fill_missing_initial_variances(self):
        for name, value in self.var.items():
            self.var_initial.setdefault(name, value)

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

        invalid = set(self.effects) - self.VALID_EFFECT_COMPONENTS
        if invalid:
            raise ValueError(f'Unknown effect names: {sorted(invalid)}')
        for name, effect in self.effects.items():
            if name in self.VALID_GENETIC_COMPONENTS:
                if not isinstance(effect, GeneticEffect):
                    raise ValueError(f'effects[{name}] must be a GeneticEffect object.')
            elif name == 'Eps':
                if not isinstance(effect, NoiseEffect):
                    raise ValueError('effects[Eps] must be a NoiseEffect object.')
            for input_name in effect.required_inputs:
                if input_name not in self.inputs:
                    raise ValueError(f"Missing required trait input '{input_name}' for effect {name}.")
            if name not in self.y_:
                raise ValueError(f'Missing realized component for effect {name}.')

        if self.type == 'composite' and 'Eps' not in self.effects:
            raise ValueError('Composite traits must contain an Eps effect.')

    @classmethod
    def from_effects(cls, G: Union[np.ndarray, dict], effects: Union[np.ndarray, dict],
                     per_allele: Union[bool, dict] = False,
                     var_Eps: float = 0.0, G_par: np.ndarray = None) -> 'Trait':
        '''
        Initializes a trait from one or more specified genetic effect arrays.
        '''
        trait = cls.__new__(cls)
        trait._initialize_empty()

        effects_dict = trait._coerce_effects_dict(effects)
        G_dict = trait._coerce_genotype_dict(G, G_par=G_par)
        per_allele_dict = trait._coerce_per_allele(per_allele, set(effects_dict))

        missing = set(effects_dict) - set(G_dict)
        if missing:
            raise ValueError(f'Missing genotype matrices for effects: {sorted(missing)}')

        for name, effect_array in effects_dict.items():
            G_input = G_dict['A'] if name == 'A' else G_dict['A_par']
            trait.effects[name] = GeneticEffect(
                name, effect_array, G=G_input, per_allele=per_allele_dict[name]
            )

        trait.effects['Eps'] = NoiseEffect('Eps', var=var_Eps)
        trait.var['Eps'] = var_Eps
        trait.var_initial['Eps'] = var_Eps

        init_inputs = {}
        if 'A' in G_dict:
            init_inputs['G'] = G_dict['A']
            init_inputs['G_std'] = trait.effects['A'].G_std
        if 'A_par' in G_dict:
            init_inputs['G_par'] = G_dict['A_par']
            init_inputs['G_par_std'] = trait.effects['A_par'].G_std
        trait.update_inputs(**init_inputs)
        trait.generate_trait()
        for name in trait.effects:
            if name == 'Eps':
                continue
            trait.var_initial.setdefault(name, trait.var[name])
        trait.validate()
        return trait

    @classmethod
    def from_fixed_values(cls, y: np.ndarray) -> 'Trait':
        '''
        Initializes a trait from a given array of fixed trait values.
        '''
        trait = cls.__new__(cls)
        trait._initialize_empty()
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError('y must be a 1D array.')
        trait.y = y.copy()
        trait.y_ = {'fixed': y.copy()}
        trait.inputs = {'N': y.shape[0]}
        trait.type = 'fixed'
        trait._update_empirical_variances()
        trait._fill_missing_initial_variances()
        trait.validate()
        return trait

    def generate_trait(self, inputs: dict = None, **kwargs):
        '''
        Generates or updates trait values from the stored effects and current inputs.
        '''
        if self.type == 'fixed':
            raise ValueError('Cannot regenerate a fixed trait from stored effects.')
        if inputs is not None or kwargs:
            self.update_inputs(inputs=inputs, **kwargs)

        self.y_ = {}
        for name, effect in self.effects.items():
            effect.refresh_from_inputs(self.inputs)
            self.y_[name] = effect.generate_component(self.inputs)

        self.y = np.sum(np.column_stack(list(self.y_.values())), axis=1)
        self.type = 'composite'
        self._update_empirical_variances()
        self._fill_missing_initial_variances()
        self.validate()

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
            var_a = np.sum(self.effects['A'].effects**2)
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

    @classmethod
    def concatenate_traits(cls, traits: list, G: np.ndarray = None) -> 'Trait':
        '''
        Concatenates multiple Trait objects into a single Trait object.
        '''
        if len(traits) == 0:
            raise ValueError('Must provide at least one trait to concatenate.')

        for trait in traits:
            trait.validate()

        first = traits[0]
        if any(trait.type != first.type for trait in traits):
            raise ValueError('All concatenated traits must have the same type.')
        if any(set(trait.y_.keys()) != set(first.y_.keys()) for trait in traits):
            raise ValueError('All concatenated traits must have the same component keys.')

        trait_new = cls.__new__(cls)
        trait_new._initialize_empty()
        trait_new.type = first.type
        trait_new.y_ = {
            name: np.concatenate([trait.y_[name] for trait in traits])
            for name in first.y_.keys()
        }
        trait_new.y = np.sum(np.column_stack(list(trait_new.y_.values())), axis=1)
        trait_new._update_empirical_variances()
        trait_new.var_initial = copy.deepcopy(first.var_initial)
        trait_new.effects = copy.deepcopy(first.effects)
        if any(isinstance(effect, GeneticEffect) for effect in trait_new.effects.values()) and len(traits) > 1:
            warnings.warn('Using the effects objects from the first trait when concatenating traits.')

        trait_new.inputs = {'N': sum(int(trait.inputs['N']) for trait in traits)}
        shared_input_keys = set.intersection(*[set(trait.inputs.keys()) for trait in traits])
        for key in shared_input_keys - {'N'} - cls.DERIVED_INPUTS:
            values = [trait.inputs[key] for trait in traits]
            first_value = values[0]
            if isinstance(first_value, np.ndarray) and first_value.ndim >= 1 and first_value.shape[0] == int(traits[0].inputs['N']):
                trait_new.inputs[key] = np.concatenate(values, axis=0)
            else:
                trait_new.inputs[key] = copy.deepcopy(first_value)
        trait_new._refresh_derived_inputs()

        trait_new._fill_missing_initial_variances()
        trait_new.validate()
        return trait_new

    def index_trait(self, i_keep: np.ndarray, G: np.ndarray = None,
                    G_already_indexed: bool = False) -> 'Trait':
        '''
        Returns a Trait object that contains only the specified individuals.
        '''
        self.validate()
        trait_new = self.__class__.__new__(self.__class__)
        trait_new._initialize_empty()
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

        trait_new._fill_missing_initial_variances()
        trait_new.validate()
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

    def join_populations(self, pop_i: list, shared_haplotypes: bool = False):
        '''
        Joins multiple populations into a single population. Inactivates the original populations and creates a new population from the merged haplotypes. The new population is added to the superpopulation as an active population.
        Parameters:
            pop_i (list): List of indices of populations to join.
            shared_haplotypes (bool): Whether the haplotype IDs stored inside the Haplos object of each population refer to the same haplotypes across populations. This is applicable when the populations being joined are either derived from the same ancestral population or are multiple generations of the same population. If true, haplotype IDs of a population are shifted by the number of individuals in the populations preceding it in the array. Default is False.
        '''
        if len(pop_i) < 2:
            raise ValueError("Must specify at least two populations to join.")
        
        # merges haplotypes of specified populations
        H = np.concatenate([self.pops[i].H for i in pop_i], axis=0)
        # creates new population from merged haplotypes
        new_pop = Population.from_H(H, keep_past_generations=0)

        # merging Haplotype IDs
        if shared_haplotypes:
            Haplos = np.concatenate([self.pops[i].Haplos for i in pop_i], axis=0)
        else:
            shift = 0
            P = self.pops[pop_i[0]].Haplos.shape[2]  # ploidy: number of haplotypes per individual
            Haplos_list = []
            for i in pop_i:
                Haplos_list.append( self.pops[i].Haplos + shift )
                shift += P * self.pops[i].N
            Haplos =  np.concatenate(Haplos_list, axis=0)

        new_pop.Haplos = Haplos
        # adds Trait objects by concatenating them, assumes the first population has all traits
        for name in self.pops[pop_i[0]].traits.keys():
            # concatenates traits from all populations being joined
            traits = [self.pops[i].traits[name] for i in pop_i]
            trait_new = Trait.concatenate_traits(traits, new_pop.G)
            trait_new.pop = new_pop
            new_pop.traits[name] = trait_new
        
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
            new_pop = Population.from_H(H_new)
            # updates traits
            for name, trait in source.traits.items():
                trait_new = trait.index_trait(i_new, source.G, G_already_indexed=False)
                trait_new.pop = new_pop
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
    def add_trait(self, name: str, per_allele_p_pop: int = None, **kwargs):
        '''
        Adds a trait to all active populations in the superpopulation.
        Parameters:
            name (str): Name of the trait to add.
            per_allele_p_pop (int): The index of the population from which to pull allele frequencies in order to generate per-allele effect sizes. The method first generates standardized per-allele effects, but for biological realism, effects are fixed across populations by scaling them to be per-allele effect sizes. This requires having a set of allele frequencies (or more precisely, genotype standard deviations) to scale effects by, which can be extracted from the specified population index. If not provided (Default), the method will join the active populations together and get the standard deviation of the genotype matrix across them.
            **kwargs: All arguments are analogous to the `Trait` constructor method. See that method for details. However, the per-allele genetic effects are shared across all populations. For each parameter except `var_G` and `M_causal`, if a Python list is passed, it is assumed to be a list of arguments for each population in the superpopulation. If a list isn't passed, it is used for all populations.
        '''
        # generates genetic effects
        M = self.pops[0].M  # assumes all populations have the same number of variants
        M_causal = kwargs.get('M_causal', M)
        var_G = kwargs.get('var_G', 1.0)
        effects, _ = stat.generate_causal_effects(M, M_causal, var_G)
        if per_allele_p_pop is not None:
            # gets standard deviation of genotype matrix from specified population
            G_std = stat.get_G_std_for_effects(self.pops[per_allele_p_pop].G)
        else:
            # gets average standard deviation of genotype matrix across all active populations
            G_std = np.mean([stat.get_G_std_for_effects(pop.G) for pop, is_active
                             in zip(self.pops, self.active) if is_active], axis=0)
        
        effects_per_allele = stat.get_standardized_effects(effects, G_std, std2allelic=True)

        # iterates through each active population
        for i, pop_i in enumerate(self.active_i):
            pop = self.pops[pop_i]
            pop_kwargs = core.get_pop_kwargs(i, **kwargs)
            # Remove keys not accepted by add_trait_from_effects
            pop_kwargs.pop('var_G', None)
            pop_kwargs.pop('M_causal', None)
            # adds trait to population using pre-computed effects
            pop.add_trait_from_effects(name=name, effects=effects_per_allele, per_allele=True, **pop_kwargs)

    def add_subpop_trait(self):
        '''
        Adds a fixed trait named `subpop` to each active population in the superpopulation.
        Each individual in an active population receives the index of that population in
        the superpopulation's `pops` list.
        '''
        for pop_i in self.active_i:
            pop = self.pops[pop_i]
            y = np.full(pop.N, pop_i, dtype=int)
            pop.add_trait_from_fixed_values(name='subpop', y=y)

    ####################
    #### Simulating ####
    ####################
    def simulate_generations(self, **kwargs):
        '''
        Simulates generations for all active populations in the superpopulation. 
        Parameters:
            **kwargs: All arguments are passed to the `simulate_generations()` method of each Population object. See that method for details. For each parameter, if a Python list is passed, it is assumed to be a list of arguments for each population in the superpopulation. If a list isn't passed, it is used for all populations.
        '''
        # iterates through each active population
        for i, pop_i in enumerate(self.active_i):
            pop = self.pops[pop_i]
            # creates kwargs list for each population
            pop_kwargs = {}
            # simulates generations for the population using population-specific kwargs
            pop_kwargs = core.get_pop_kwargs(i, **kwargs)
            pop.simulate_generations(**pop_kwargs)

    #######################
    #### Visualization ####
    #######################
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

"""SuperPopulation class for coordinating multiple populations."""

from __future__ import annotations

import copy
import warnings
from typing import Optional, Tuple, Union

import numpy as np

from .. import core_functions as core
from .. import popgen_functions as pop
from ..pedigree.pedigree import Pedigree
from ..traits.effects import FixedEffect, GeneticEffect, NoiseEffect
from ..traits.trait import Trait
from .population import Population


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
            new_pops.append(source.subset_individuals(i_new, keep_past_generations=0))
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


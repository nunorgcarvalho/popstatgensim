"""Pedigree data structures and relationship classification logic."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .relative_types import REL_TYPES


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

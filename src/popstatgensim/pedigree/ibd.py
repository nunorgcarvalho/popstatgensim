"""IBD and relatedness helper functions."""

from dataclasses import dataclass

import numpy as np


def get_true_IBD1(chr_i: np.ndarray, chr_j: np.ndarray) -> np.ndarray:
    '''
    Computes true IBD1 segments between two haplotypes.
    Parameters:
        chr_i (1D array): Array of length M containing haplotype IDs for haplotype i.
        chr_j (1D array): Array of length M containing haplotype IDs for haplotype j.
    Returns:
        IBD1 (1D boolean array): Array of length M where True indicates that the two haplotypes are IBD1 at that variant.
    '''
    IBD1 = (chr_i == chr_j)
    # the -1 identifier is used to denote unrelated haplotypes
    unrelated_mask = (chr_i == -1) | (chr_j == -1)
    IBD1[unrelated_mask] = False
    return IBD1

def get_true_IBD_tensor(haplos_i:np.ndarray, haplos_j:np.ndarray) -> np.ndarray:
    '''
    Returns a 3D (M*P*P) tensor indicating IBD1 status for each pair of haplotypes between two individuals. Genotypes must be diploid (P=2).
    Parameters:
        haplos_i (2D array): M*2 array containing haplotype IDs for individual i.
        haplos_j (2D array): M*2 array containing haplotype IDs for individual j.
    Returns:
        IBD_tensor (3D boolean array): M*P*P array where each element at (m, p_i, p_j) indicates whether haplotype p_i of individual i and haplotype p_j of individual j are IBD1 at variant m.
    '''
    # extracts relevant genotype dimensions (# of variants and ploidy)
    (M, P) = haplos_i.shape # assumes same dimensions for both individuals
    # checks if diploid
    if P != 2:
        raise ValueError("Genotypes must be diploid to compute IBD2.")
    # makes empty list that is P*P*M
    IBD_tensor = np.zeros((M,P,P), dtype=bool)
    # computes IBD1 status for each pair of haplotypes
    for (i_haplo, j_haplo) in [(0,0), (0,1), (1,0), (1,1)]:
        IBD_hi_hj = get_true_IBD1(haplos_i[:, i_haplo], haplos_j[:, j_haplo])
        IBD_tensor[:, i_haplo, j_haplo] = IBD_hi_hj
    
    return IBD_tensor

def get_true_IBD_arr(haplos_i:np.ndarray, haplos_j:np.ndarray, return_tensor: bool = False) -> np.ndarray:
    '''
    Computes the IBD state (0,1,2) between two individuals based on their haplotype IDs. Genotypes must be diploid (P=2). Assumes that an individuals two haplotypes are different. So, e.g., if comparing (A,A) and (A,B), this is treated as IBD1, not IBD2.
    Parameters:
        haplos_i (2D array): M*2 array containing haplotype IDs for individual i.
        haplos_j (2D array): M*2 array containing haplotype IDs for individual j.
        return_tensor (bool): If True, also returns the full IBD tensor (M*P*P array) indicating IBD1 status for each haplotype pair.
    Returns:
        tuple ((IBD_arr, IBD_tensor)):
        Where:
        - IBD_arr (1D array): Array of length M where each element is 0, 1, or 2 indicating the IBD state between the two individuals at that variant.
        - IBD_tensor (3D array, optional): If return_tensor is True, also returns the full IBD tensor (see get_true_IBD_tensor).
    '''
    
    IBD_tensor = get_true_IBD_tensor(haplos_i, haplos_j)
    (M, P, _) = IBD_tensor.shape
    # sums IBD1 statuses to get IBD2 status
    # by assuming that an individual's two haplotypes are different, it means IBD2 state can only occur when two distinct pairs of haplotype indices are IBD1 (e.g. (0,1) and (1,0), or (0,0) and (1,1), NOT (0,0) and (0,1))
    IBD_arr = np.zeros(M, dtype=np.uint8)
    # IBD1 status is first given to any variant where any haplotype pair is IBD1
    IBD_arr[np.any(IBD_tensor, axis=(1,2))] = 1
    # IBD2 status is then given to any variant where either (0,0) and (1,1) are both IBD1, or (0,1) and (1,0) are both IBD1
    IBD2_mask = ( (IBD_tensor[:,0,0] & IBD_tensor[:,1,1]) |
                  (IBD_tensor[:,0,1] & IBD_tensor[:,1,0]) )
    IBD_arr[IBD2_mask] = 2

    if return_tensor:
        return (IBD_arr, IBD_tensor)
    else:
        return (IBD_arr)

def get_coeff_kinship(haplos_i:np.ndarray, haplos_j:np.ndarray, return_arr: bool = False) -> float:
    '''
    Computes the coefficient of kinship between two individuals based on their haplotype IDs.
    Parameters:
        haplos_i (2D array): M*2 array containing haplotype IDs for individual i.
        haplos_j (2D array): M*2 array containing haplotype IDs for individual j.
        return_arr (bool): If True, also returns an array with the coefficient of kinship between the two individuals at each variant.
    Returns:
        tuple ((coeff_kinship, coeff_kinship_arr)):
        Where:
        - coeff_kinship (float): Coefficient of kinship between the two individuals.
        - coeff_kinship_arr (1D array, optional): If return_arr is True, also returns an array with the coefficient of kinship between the two individuals at each variant.
    '''
    IBD_tensor = get_true_IBD_tensor(haplos_i, haplos_j).astype(np.uint8)
    P = IBD_tensor.shape[1]
    # sums IBD1 statuses across all haplotype pairs
    total_IBD1_arr = IBD_tensor.sum(axis=(1,2))
    # computes kinship coefficient
    coeff_kinship_arr = total_IBD1_arr / (P * P)
    coeff_kinship = coeff_kinship_arr.mean()
    if return_arr:
        return (coeff_kinship, coeff_kinship_arr)
    else:
        return coeff_kinship

def get_coeff_inbreeding(haplos_i:np.ndarray, return_arr: bool = False) -> float:
    '''
    Computes the coefficient of inbreeding for an individual based on their IBD tensor.
    Parameters:
        haplos_i (2D array): M*2 array containing haplotype IDs for individual i.
        return_arr (bool): If True, also returns an array with the coefficient of inbreeding at each variant.
    Returns:
        tuple ((coeff_inbreeding, coeff_inbreeding_arr)):
        Where:
        - coeff_inbreeding (float): Coefficient of inbreeding for the individual.
        - coeff_inbreeding_arr (1D array, optional): If return_arr is True, also returns an array with the coefficient of inbreeding (0 or 1) at each variant.
    '''
    IBD_tensor = get_true_IBD_tensor(haplos_i, haplos_i).astype(np.uint8)
    # inbreeding coefficient is the IBD1 status between the two distinct haplotypes
    coeff_inbreeding_arr = IBD_tensor[:,0,1] # guaranteed to be symmetric, so either off-diagonal works
    coeff_inbreeding = coeff_inbreeding_arr.mean()
    if return_arr:
        return (coeff_inbreeding, coeff_inbreeding_arr)
    else:
        return coeff_inbreeding

def get_coeff_relatedness(haplos_i:np.ndarray, haplos_j:np.ndarray, return_arr: bool = False) -> float:
    '''
    Returns the coefficient of relatedness between two individuals based on their haplotype IDs. Simply twice the coefficient of kinship, see get_coeff_kinship().
    Parameters:
        haplos_i (2D array): M*2 array containing haplotype IDs for individual i.
        haplos_j (2D array): M*2 array containing haplotype IDs for individual j.
        return_arr (bool): If True, also returns an array with the coefficient of relatedness between the two individuals at each variant.
    Returns:
        tuple ((coeff_relatedness, coeff_relatedness_arr)):
        Where:
        - coeff_relatedness (float): Coefficient of relatedness between the two individuals.
        - coeff_relatedness_arr (1D array, optional): If return_arr is True, also returns an array with the coefficient of relatedness between the two individuals at each variant.
    '''
    (coeff_kinship, coeff_kinship_arr) = get_coeff_kinship(haplos_i, haplos_j, return_arr=True)
    coeff_relatedness = 2 * coeff_kinship
    if return_arr:
        coeff_relatedness_arr = 2 * coeff_kinship_arr
        return (coeff_relatedness, coeff_relatedness_arr)
    else:
        return coeff_relatedness

def compute_K_IBD(Haplos: np.ndarray, standardize: bool = False) -> np.ndarray:
    '''
    Computes the kinship matrix based on true IBD between all pairs of individuals in the population.
    Parameters:
        Haplos (3D array): N*M*P array of haplotype IDs. First dimension is individuals, second dimension is variants, third dimension is haplotype number (related to ploidy).
        standardize (bool): If True, standardizes the kinship matrix according to [Young et al. 2018 NatGen]. The mean value in the matrix becomes 0. Default is False.
    Returns:
        K_IBD (2D array): N*N kinship matrix based on true IBD between all pairs of individuals in the population. Element (i,j) is the coefficient of *relatedness* (twice that of kinship) between individuals i and j.
    '''
    if Haplos is None:
        raise ValueError('Haplotype IDs are not being tracked, so true IBD cannot be computed.')
    N = Haplos.shape[0]
    K_IBD = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            coeff_relatedness = get_coeff_relatedness(Haplos[i, :, :], Haplos[j, :, :])
            K_IBD[i, j] = coeff_relatedness
            K_IBD[j, i] = coeff_relatedness
    if standardize:
        K0 = np.mean(K_IBD) # mean relatedness
        print(f'Mean relatedness before standardization: {K0}')
        K_IBD = (K_IBD - K0) / (1 - K0)
    return K_IBD

def greedy_unrelated_subset(relatedness: np.ndarray,
                            max_relatedness: float) -> np.ndarray:
    '''
    Returns a greedy maximal set of individuals whose pairwise relatedness does not
    exceed `max_relatedness`.
    Parameters:
        relatedness (2D array): Square pairwise relatedness matrix, such as a SNP GRM
            or true-IBD relatedness matrix.
        max_relatedness (float): Maximum allowed off-diagonal relatedness between any
            retained pair.
    Returns:
        i_keep (1D int array): Sorted indices of a greedy maximal unrelated subset.
    Notes:
        The routine constructs a graph connecting pairs with relatedness greater than
        `max_relatedness`, then greedily keeps low-degree vertices and removes their
        neighbors. This yields a maximal independent set, though not necessarily the
        globally largest one.
    '''
    relatedness = np.asarray(relatedness, dtype=float)
    if relatedness.ndim != 2 or relatedness.shape[0] != relatedness.shape[1]:
        raise ValueError('`relatedness` must be a square 2D array.')
    if not np.isfinite(relatedness).all():
        raise ValueError('`relatedness` must contain only finite values.')

    N = relatedness.shape[0]
    adjacency = relatedness > max_relatedness
    np.fill_diagonal(adjacency, False)

    active = np.ones(N, dtype=bool)
    keep = []

    while np.any(active):
        active_i = np.flatnonzero(active)
        adj_active = adjacency[np.ix_(active_i, active_i)]
        degrees = adj_active.sum(axis=1)
        weighted_degree = np.where(
            adj_active,
            relatedness[np.ix_(active_i, active_i)],
            0.0,
        ).sum(axis=1)

        order = np.lexsort((active_i, weighted_degree, degrees))
        chosen = int(active_i[order[0]])
        keep.append(chosen)

        neighbors = adjacency[chosen] & active
        active[chosen] = False
        active[neighbors] = False

    return np.array(sorted(keep), dtype=np.int32)

@dataclass
class IBDSegment:
    #i: int = field(init=False) # index of individual i, not necessary
    #j: int = field(init=False) # index of individual j, not necessary
    start: int # start index (inclusive) of IBD segment
    end: int # end index (inclusive) of IBD segment
    i_chr: int # chromosome index of individual i
    j_chr: int # chromosome index of individual j

    @property
    def length(self) -> int:
        return self.end - self.start + 1

def IBD_tensor_to_segments(IBD_tensor: np.ndarray, i_chrs: list = [0,1], j_chrs: list = [0,1]) -> list:
    '''
    Given an IBD tensor (M*P*P array; see pop.get_true_IBD_tensor()) between two individuals, extracts continuous IBD segments for each chromosome pair and stores in a list.
    Parameters:
        IBD_tensor (3D array): M*P*P array where each element at (m, p_i, p_j) indicates whether haplotype p_i of individual i and haplotype p_j of individual j are IBD1 at variant m.
        i_chrs (list): List of haplotype indices (0 to P-1) for individual i to extract segments from. Default is [0,1] (both haplotypes are checked).
        j_chrs (list): List of haplotype indices (0 to P-1) for individual j to extract segments from. Default is [0,1] (both haplotypes are checked).
    '''
    segments = []
    for i_chr in i_chrs:
        for j_chr in j_chrs:
            IBD_vector = IBD_tensor[:, i_chr, j_chr].astype(np.int8)
            IBD_vector = np.concatenate([[0], IBD_vector, [0]])  # padding to catch segments at ends
            IBD_diff = np.diff(IBD_vector)
            IBD_starts = np.where(IBD_diff == 1)[0]
            IBD_ends = np.where(IBD_diff == -1)[0] - 1
            if len(IBD_starts) != len(IBD_ends):
                raise ValueError("Mismatch in number of IBD segment starts and ends. This shouldn't happen.")
            for k in range(len(IBD_starts)):
                segment = IBDSegment(start=IBD_starts[k], end=IBD_ends[k],
                                     i_chr=i_chr, j_chr=j_chr)
                segments.append(segment)

    return segments

def extract_crossover_points(seg: IBDSegment, M: int) -> list:
    '''
    Given an IBD segment, extracts the crossover point(s) that led to the start and end of the segment. This will return 2 indices if the IBD segment is not at the start or end of the chromosome. Crossover points are interpreted as happening at the index of the start of an IBD segment and the index right after the end of an IBD segment.
    Parameters:
        seg (IBDSegment): An IBDSegment object representing an IBD segment.
        M (int): Total number of variants on the chromosome. Used to know if the segment ends at the end of the chromosome.
    Returns:
        tuple ((crossover_points, start_stop, chrs)):
        Where:
        - crossover_points (list): List of crossover point indices (0-based). Will contain 0, 1, or 2 indices depending on whether the segment starts at the beginning or ends at the end of the chromosome.
        - start_stop (list): List indicating whether the corresponding crossover point is at the 'start' (0) or 'end' (1) of the segment.
        - chrs (list): List consisting of a tuple denoting the chromosome index for individual i and individual j for the segment.
    '''
    crossover_points = []
    start_stop = []
    chrs = []
    if seg.start > 0:
        crossover_points.append(seg.start)
        start_stop.append(0)
    if seg.end < (M - 1):
        crossover_points.append(seg.end + 1)
        start_stop.append(1)

    for _ in crossover_points:
        chrs.append( (seg.i_chr, seg.j_chr) )
    return (crossover_points, start_stop, chrs)

def extract_all_crossover_points(segments: list, M: int) -> list:
    '''
    Extracts all crossover points from a list of IBD segments. See extract_crossover_points() for more details.
    Parameters:
        segments (list): List of IBDSegment objects representing IBD segments.
        M (int): Total number of variants on the chromosome. Used to know if the segment ends at the end of the chromosome.
    Returns:
        tuple ((crossover_points, start_stop, chrs)):
        Where:
        - crossover_points (list): List of crossover point indices (0-based), sorted in ascending order by index. A crossover point may appear multiple times if it is shared by multiple segments. 
        - start_stop (list): List indicating whether the corresponding crossover point is at the 'start' (0) or 'end' (1) of the segment.
        - chrs (list): List consisting of a tuple denoting the chromosome index for individual i and individual j for the segment.

    '''
    crossover_points = []
    start_stop = []
    chrs = []
    for seg in segments:
        seg_crossover_points, seg_start_stop, seg_chrs = extract_crossover_points(seg, M)
        crossover_points.extend(seg_crossover_points)
        start_stop.extend(seg_start_stop)
        chrs.extend(seg_chrs)
    
    # sorts crossover points by index, and maintains order for start/stop
    sort_idx = np.argsort(np.array(crossover_points))
    crossover_points = [crossover_points[i] for i in sort_idx]
    start_stop = [start_stop[i] for i in sort_idx]
    chrs = [chrs[i] for i in sort_idx]

    return (crossover_points, start_stop, chrs)

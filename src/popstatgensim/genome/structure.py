"""Genome-structure and haplotype-generation utilities."""

import numpy as np


def draw_binom_haplos(p: np.ndarray, N: int, P: int = 2) -> np.ndarray:
    '''
    Generates 3-dimensional matrix of population haplotypes (allele dosages).
    Parameters:
        p (1D array): Array of length M containing allele frequencies from which haplotypes are drawn.
        N (int): Number of individuals in the population.
        P (int): Ploidy of the population. Default is 2 (diploid).
    Returns:
        H (3D array): N*M*P array of alleles. First dimension is individuals, second dimension is variants, third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
    '''
    M = p.shape[0]
    p = p.reshape(1, M, 1)
    H = np.random.binomial( 1, p = p, size = (N, M, P)).astype(np.uint8)
    return H

def generate_LD_blocks(M: int, N_blocks: int = None,
                       hotspot_lambda: float = None, block_R: float = None):
    '''
    Generates an array of recombination rates that produces LD blocks when simulated. Base recombination rate is kept low, with a few recombination hotspots scattered throughout with much higher recombination rates. Hotspot recombination rates are drawn from an exponential distribution.
    Parameters:
        M (int): Number of variants
        N_blocks (int): Number of LD blocks. If not specified, defaults to whichever is higher between 2 and M // 10 (i.e. LD block every 10 variants on average). If 0 or 1, no recombination hotspots are formed.
        hotspot_lambda (float): Lambda parameter for exponential distribution for which hotspot recombination rates are drawn from. The mean recombination rate is 1/hotspot_lambda, and the variance is 1/hotspot_lambda^2. If not specified, defaults to M, meaning that ~1 crossover event is expected per genome per generation.
        block_R (float): Recombination rate for non-hotspot variants. If not specified, defaults to 1 / (100*M).
    Returns:
        R (1D array) Array of length M specifying recombination rates.
    '''
    if N_blocks is None:
        N_blocks = np.max([2, M // 10])
    N_hotspots = N_blocks - 1
    if block_R is None:
        block_R = (1 / M) / 100
    if hotspot_lambda is None:
        hotspot_lambda = M
    
    R = np.full(M, block_R)
    if N_blocks > 1:
        j_hotspots = np.random.choice(M, size=N_hotspots, replace=False)
        R[j_hotspots] = np.random.exponential(scale = 1 / hotspot_lambda, size = N_hotspots)
    return R

def generate_chromosomes(M: int ,chrs: int = 1, meioses_per_chr: int = 1):
    '''
    Generates array of recombination rates that best approximates a specified number of chromosomes and meioses per chromosome.
    '''
    # gets number of variants per chromosome
    R = np.zeros(M)
    M_left = M
    # splits whole genome into (mostly) even-lengthed chromosomes
    # and sets recombinaton such that each chromosome has an expected meioses_per_chr crossovers per meiosis
    for c in range(chrs):
        M_c = int( np.ceil(M_left / (chrs - c)) )
        start = M - M_left
        stop = M - M_left + M_c
        # sets recombination rates for chromosome
        R[start : stop] = meioses_per_chr / M_c
        if c > 1:
            R[start] = 0.5 # independence between chromosomes
            
        M_left -= M_c

    return np.array(R)

def draw_p_init(M, method: str, params: list) -> np.ndarray:
    '''
    Returns array of initial allele frequencies to simulate genotypes from.
    Parameters:
        method (str): Method of randomly drawing allele frequencies. Options:
            uniform: 'Draws from uniform distribution with lower and upper bounds specified by `params`.'
        params (list): Method-specific parameter values.
    Returns:
        p_init (1D array): Array of length M containing allele frequencies.
    '''
    # uniform sampling is default (and only currently supported method)
    if method == 'uniform':
        p_init = np.random.uniform(params[0], params[1], M)
        return p_init
    else:
        return np.full(M, np.nan)

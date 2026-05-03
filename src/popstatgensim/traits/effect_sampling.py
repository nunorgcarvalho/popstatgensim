"""Trait effect-size sampling and genotype-effect transforms."""

import warnings

import numpy as np


def _normalize_effect_dist(dist: str) -> str:
    '''
    Normalizes effect-size distribution names used by public sampling helpers.
    '''
    dist = str(dist).lower()
    if dist in {'constant', 'point_mass', 'pointmass'}:
        return 'constant'
    if dist == 'normal':
        return dist
    raise ValueError("dist must be either 'normal', 'constant', or 'point_mass'.")


def generate_causal_effects(M: int, M_causal: int = None, var_G: float = 1.0, dist: str = 'normal') -> tuple[np.ndarray, np.ndarray]:
    '''
    Generates variant effect sizes for some trait. Default interpretation is per-standardized-allele effect sizes. Non-causal variant effects are set to 0. Causal effects are drawn from a specified distribution (default is Normal).
    Parameters:
        M (int): Total number of variants (causal and non-causal).
        M_causal (int): Number of causal variants (variants with non-zero effect sizes). Default is all variants.
        var_G (float): Total expected variance contributed by per-standardized-allele genetic effects. Refers to direct genetic effects. Default is 1.0.
        dist (str): Distribution to draw causal effects from. Options are:
            - 'normal': Normal distribution (default).
            - 'constant' or 'point_mass': All causal effect sizes are the same.
    Returns:
        tuple ((causal_effects, j_causal)):
        Where:
        - causal_effects (1D array): Array of length M containing the effect sizes of all variants.
        - j_causal (1D array): Array of length M_causal containing the variant indices of causal variants.
    '''
    dist = _normalize_effect_dist(dist)
    if M_causal is None:
        M_causal = M
    if M_causal < 0 or M_causal > M:
        raise ValueError('M_causal must be between 0 and M.')
    if var_G < 0:
        raise ValueError('var_G must be non-negative.')

    causal_effects = np.zeros(M)
    j_causal = np.random.choice(M, M_causal, replace=False)
    if M_causal > 0:
        if dist == 'normal':
            causal_effects[j_causal] = np.random.normal(0, np.sqrt(var_G/M_causal), M_causal)
        elif dist == 'constant':
            causal_effects[j_causal] = np.sqrt(var_G/M_causal)
    return (causal_effects, j_causal)

def generate_genetic_effects(var_A: float, var_A_par: float, r: float,
                             M: int, M_causal: int = None,
                             dist: str = 'normal',
                             force_var: bool = False,
                             G: np.ndarray = None, G_std: np.ndarray = None,
                             G_par: np.ndarray = None, G_par_std: np.ndarray = None) -> dict:
    '''
    Generates paired direct (`A`) and parental (`A_par`) genetic effects on the same causal variants.
    Effect sizes are drawn jointly from a bivariate normal so that each causal variant has
    per-standardized-allele covariance matrix:
        [[var_A / M_causal,     r * sqrt(var_A * var_A_par) / M_causal],
         [same,                 var_A_par / M_causal]]
    Parameters:
        var_A (float): Expected independent-site variance of the direct genetic component.
        var_A_par (float): Expected independent-site variance of the parental genetic component.
        r (float): Target correlation between the standardized causal effects for A and A_par.
        M (int): Total number of variants.
        M_causal (int): Number of shared causal variants. Default is all variants.
        dist (str): Distribution to draw causal effects from. Options are:
            - 'normal': Bivariate normal distribution (default).
            - 'constant' or 'point_mass': All causal effect sizes are the same
              within each genetic component. In this mode `r` controls only the
              sign of `A_par`, because a point-mass pair cannot realize an
              arbitrary cross-effect correlation.
        force_var (bool): Passed to both returned GeneticEffect objects.
        G (2D array): Optional offspring genotype matrix used to compute G_std for the A effect.
        G_std (1D array): Optional offspring genotype standard deviations for the A effect.
        G_par (2D array): Optional parental genotype-sum matrix used to compute G_par_std for the A_par effect.
        G_par_std (1D array): Optional parental genotype-sum standard deviations for the A_par effect.
            If only G or G_std is provided, the A_par effect uses sqrt(2) times that offspring
            standard deviation as an approximation for the larger variance of G_par.
    Returns:
        effects (dict): Dictionary with keys 'A' and 'A_par', each containing a GeneticEffect object.
    '''
    from .effects import GeneticEffect

    dist = _normalize_effect_dist(dist)

    def _resolve_G_std(G_input: np.ndarray, G_std_input: np.ndarray, name: str) -> np.ndarray:
        if G_input is not None and G_std_input is not None:
            raise ValueError(f'Provide only one of {name} or {name}_std.')
        if G_input is not None:
            return get_G_std_for_effects(G_input, P=int(G_input.max()) if G_input.size > 0 else None)
        if G_std_input is not None:
            G_std_output = np.asarray(G_std_input, dtype=float)
            if G_std_output.ndim != 1:
                raise ValueError(f'{name}_std must be a 1D array.')
            if G_std_output.shape[0] != M:
                raise ValueError(f'Length of {name}_std must match M.')
            return G_std_output
        return None

    if M_causal is None:
        M_causal = M
    if M_causal < 0 or M_causal > M:
        raise ValueError('M_causal must be between 0 and M.')
    if abs(r) > 1:
        raise ValueError('r must be between -1 and 1.')
    if var_A < 0 or var_A_par < 0:
        raise ValueError('var_A and var_A_par must be non-negative.')
    if (var_A == 0 or var_A_par == 0) and not np.isclose(r, 0.0):
        raise ValueError('r must be 0 when either var_A or var_A_par is 0.')

    G_std_A = _resolve_G_std(G, G_std, 'G')
    G_std_A_par = _resolve_G_std(G_par, G_par_std, 'G_par')
    if G_std_A_par is None and G_std_A is not None:
        G_std_A_par = np.sqrt(2.0) * G_std_A

    effects_A = np.zeros(M, dtype=float)
    effects_A_par = np.zeros(M, dtype=float)

    if M_causal > 0:
        j_causal = np.random.choice(M, M_causal, replace=False)
        if dist == 'normal':
            cov = np.array([
                [var_A / M_causal, r * np.sqrt(var_A * var_A_par) / M_causal],
                [r * np.sqrt(var_A * var_A_par) / M_causal, var_A_par / M_causal],
            ])
            draws = np.random.multivariate_normal(mean=np.zeros(2), cov=cov, size=M_causal)
            effects_A[j_causal] = draws[:, 0]
            effects_A_par[j_causal] = draws[:, 1]
        elif dist == 'constant':
            effects_A[j_causal] = np.sqrt(var_A / M_causal)
            effects_A_par[j_causal] = np.sqrt(var_A_par / M_causal)
            if r < 0:
                effects_A_par[j_causal] *= -1.0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        effects = {
            'A': GeneticEffect.from_effects(
                effects=effects_A,
                is_standardized=True,
                name='A',
                force_var=force_var,
                var_indep=var_A,
                G_std=G_std_A,
            ),
            'A_par': GeneticEffect.from_effects(
                effects=effects_A_par,
                is_standardized=True,
                name='A_par',
                force_var=force_var,
                var_indep=var_A_par,
                G_std=G_std_A_par,
            ),
        }
    return effects

def compute_genetic_value(G: np.ndarray, effects: np.ndarray) -> np.ndarray:
    '''
    Computes the genetic value/score given a (standardized) genotype matrix and (per-standardized-allele) effect sizes.
    Parameters:
        G (2D array): N*M NON-standardized genotype matrix.
        effects (1D array): Array of length M containing causal and non-causal genetic effects.
    Returns:
        y_G (1D array): Array of length N containing genetic component values.
    '''
    # dot product
    y_G = G @ effects
    return y_G

def generate_noise_value(N: int, var_Eps: float = 0.0) -> np.ndarray:
    '''
    Generates noise component of trait drawn randomly from Normal distribution.
    Parameters:
        N (int): Number of individuals to generate noise component for.
        var_Eps (float): Variance of the noise component. Default is 0.
    Returns:
        y_Eps (1D array): Array of length N containing noise component values.
    '''
    y_Eps = np.random.normal(loc=0, scale=np.sqrt(var_Eps), size = N)
    return y_Eps

def get_G_std_for_effects(G: np.ndarray, p_min: float = 0.05, P: int = 2 ) -> np.ndarray:
    '''
    Computes the standard deviation of each column in the genotype matrix used for converting between per-allele and per-standardized-allele effects. Has safe handling for monomorphic alleles.
    Parameters:
        G (2D array): N*M NON-standardized genotype matrix.
        p_min (float): If a variant is monomorphic in G, its "true" allele frequency is assumed to be this value. Default is 0.05. Setting this to >0 may prevent downstream issues.
        P (int): Ploidy level. Default is 2.
    Returns:
        G_std (1D array): Array of length M containing the standard deviations of each column in G.
    '''
    # gets observed standard deviation of the genotype matrix
    G_std = G.std(axis=0)
    # if an allele is monomorphic, it uses the standard deviation one would get for a binomial variable with p = p_min
    G_std[G_std == 0] = np.sqrt(P * p_min * (1 - p_min)) # sets standard deviation for monomorphic alleles
    return G_std

def get_standardized_effects(effects: np.ndarray, G_std: np.ndarray, std2allelic: bool = True) -> np.ndarray:
        '''
        Converts between per-allele and per-standardized-allele effects.
        Parameters:
            effects (1D array): M-length array of effects. Can be per-allele or per-standardized-allele.
            G_std (1D array): Array of length M containing genotype standard deviations to scale the effects by. Cannot have zero values. See `get_G_std_for_effects` for how to compute this safely.
            std2allelic (bool): If True (default), converts from per-standardized-allele to per-allele effects. If False, converts from per-allele to per-standardized-allele effects.
        '''
        if std2allelic:
            effects_output = effects / G_std # per-allele effects
        else:
            effects_output = effects * G_std # per-standardized-allele effects
        return effects_output

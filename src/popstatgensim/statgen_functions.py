'''
This file contains functions related to statistical genetics.
The `popsim.py` file contains classes that contain wrapper methods that call these functions, providing the class object's attributes as arguments.
Alternatively, these functions can be called directly with the appropriate arguments.
The functions here contain the documentation for the arguments and return values, which is not repeated in the class methods for the most part.
'''

# imports
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Optional, Tuple, Union

##########################
#### Trait generation ####
##########################

def generate_causal_effects(M: int, M_causal: int = None, var_G: float = 1.0, dist: str = 'normal') -> tuple[np.ndarray, np.ndarray]:
    '''
    Generates variant effect sizes for some trait. Default interpretation is per-standardized-allele effect sizes. Non-causal variant effects are set to 0. Causal effects are drawn from a specified distribution (default is Normal).
    Parameters:
        M (int): Total number of variants (causal and non-causal).
        M_causal (int): Number of causal variants (variants with non-zero effect sizes). Default is all variants.
        var_G (float): Total expected variance contributed by per-standardized-allele genetic effects. Refers to direct genetic effects. Default is 1.0.
        dist (str): Distribution to draw causal effects from. Options are:
            - 'normal': Normal distribution (default).
            - 'constant': All effect sizes are the same.
    Returns:
        tuple ((causal_effects, j_causal)):
        Where:
        - causal_effects (1D array): Array of length M containing the effect sizes of all variants.
        - j_causal (1D array): Array of length M_causal containing the variant indices of causal variants.
    '''
    if M_causal is None:
        M_causal = M
    causal_effects = np.zeros(M)
    j_causal = np.random.choice(M, M_causal, replace=False)
    if dist == 'normal':
        causal_effects[j_causal] = np.random.normal(0, np.sqrt(var_G/M_causal), M_causal)
    elif dist == 'constant':
        causal_effects[j_causal] = np.sqrt(var_G/M_causal)
    return (causal_effects, j_causal)

def generate_genetic_effects(var_A: float, var_A_par: float, r: float,
                             M: int, M_causal: int = None,
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
    from .popsim import GeneticEffect

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
        cov = np.array([
            [var_A / M_causal, r * np.sqrt(var_A * var_A_par) / M_causal],
            [r * np.sqrt(var_A * var_A_par) / M_causal, var_A_par / M_causal],
        ])
        draws = np.random.multivariate_normal(mean=np.zeros(2), cov=cov, size=M_causal)
        effects_A[j_causal] = draws[:, 0]
        effects_A_par[j_causal] = draws[:, 1]

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

#######################
#### Fixed Effects ####
#######################

def scale_binary_FE(x: np.ndarray, variance: float) -> np.ndarray:
    '''
    Mean-centers and rescales a 1D binary/two-level array to have a specified variance.
    Parameters:
        x (1D array): Array containing exactly two unique observed values.
        variance (float): Target variance for the output array. Must be non-negative.
    Returns:
        x_scaled (1D array): Array of the same length as `x`, centered at 0 and with variance `variance`.
    '''
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input `x` must be a 1D array.")
    if variance < 0:
        raise ValueError("Input `variance` must be non-negative.")

    unique_vals = np.unique(x)
    if unique_vals.size != 2:
        raise ValueError("Input `x` must contain exactly two unique observed values.")

    x_centered = x - x.mean()
    x_std = x_centered.std()
    if x_std == 0:
        raise ValueError("Input `x` must have non-zero variance.")

    x_scaled = x_centered * np.sqrt(variance) / x_std
    return x_scaled

########################
#### Random Effects ####
########################

def psd_sqrt(M: np.ndarray, clip: float = 0.0,
             pinv: bool = False, eps: float = 1e-12) -> np.ndarray:
    '''
    Returns a symmetric PSD square root or pseudo-inverse square root of a matrix.
    '''
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError('Matrix square roots require a square 2D array.')

    M = 0.5 * (M + M.T)
    w, U = np.linalg.eigh(M)

    if pinv:
        tau = eps * max(float(np.max(np.abs(w))), 1.0)
        vals = np.zeros_like(w)
        keep = w > tau
        vals[keep] = 1.0 / np.sqrt(w[keep])
    else:
        vals = np.sqrt(np.clip(w, clip, None))

    S = (U * vals) @ U.T
    return 0.5 * (S + S.T)


def nearest_correlation_matrix(X: np.ndarray, eps_eig: float = 1e-12) -> np.ndarray:
    '''
    Projects a matrix to the nearest correlation-like matrix by PSD clipping and
    unit-diagonal rescaling.
    '''
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError('Correlation projection requires a square 2D array.')

    X = 0.5 * (X + X.T)
    w, V = np.linalg.eigh(X)
    w = np.clip(w, eps_eig, None)
    Y = (V * w) @ V.T
    d = np.sqrt(np.clip(np.diag(Y), eps_eig, None))
    Dinv = np.diag(1.0 / d)
    Ccorr = Dinv @ Y @ Dinv
    Ccorr = 0.5 * (Ccorr + Ccorr.T)
    np.fill_diagonal(Ccorr, 1.0)
    return Ccorr


def build_design_matrix_from_groups(groups: np.ndarray, missing_value: int = -1,
                                    dtype: np.dtype = float,
                                    return_labels: bool = False):
    '''
    Builds an N*K design matrix from compact cluster identifiers.
    Parameters:
        groups (1D array): Cluster labels for each individual. Entries equal to
            `missing_value` are treated as unassigned and receive all-zero rows.
        missing_value (int): Sentinel value for unassigned individuals. Default is -1.
        dtype (numpy dtype): Output dtype for the design matrix.
        return_labels (bool): If True, also returns the unique cluster labels.
    Returns:
        Z (2D array): N*K design matrix.
        labels (1D array): Returned only when `return_labels=True`.
    '''
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError('groups must be a 1D array of cluster identifiers.')

    valid = groups != missing_value
    labels = np.unique(groups[valid])
    Z = np.zeros((groups.shape[0], labels.shape[0]), dtype=dtype)
    if labels.shape[0] > 0:
        _, inverse = np.unique(groups[valid], return_inverse=True)
        Z[np.where(valid)[0], inverse] = 1

    if return_labels:
        return (Z, labels)
    return Z


def _standardize_correlation_matrix(C: np.ndarray, name: str,
                                    tol: float = 1e-8) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f'{name} must be a square 2D array.')

    C = 0.5 * (C + C.T)
    if C.shape[0] > 0 and not np.allclose(np.diag(C), 1.0, atol=tol):
        raise ValueError(f'{name} must have ones on the diagonal.')

    eigvals = np.linalg.eigvalsh(C)
    if eigvals.min(initial=0.0) < -tol:
        raise ValueError(f'{name} must be positive semidefinite.')
    return C


def _center_and_scale_random_effect(values: np.ndarray, target_var: float,
                                    name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f'Random effect {name} must be a 1D array.')

    values = values - values.mean()
    current_var = float(values.var())
    if np.isclose(target_var, 0.0):
        return np.zeros_like(values)
    if np.isclose(current_var, 0.0):
        raise ValueError(f'Random effect {name} has zero variance and cannot be rescaled.')
    return values * np.sqrt(target_var / current_var)


def is_identity_matrix(M: np.ndarray, tol: float = 1e-8) -> bool:
    '''
    Returns True if M is numerically close to an identity matrix.
    '''
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    if not np.allclose(np.diag(M), 1.0, atol=tol):
        return False
    off_diag = M - np.diag(np.diag(M))
    return np.allclose(off_diag, 0.0, atol=tol)


def get_group_assignments_from_design(Z: np.ndarray, tol: float = 1e-8) -> Optional[np.ndarray]:
    '''
    If Z is a one-hot membership matrix with at most one cluster per individual,
    returns compact cluster assignments; otherwise returns None.
    '''
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        return None
    if Z.shape[0] == 0:
        return np.array([], dtype=np.int32)

    row_sums = Z.sum(axis=1)
    if not np.allclose(row_sums, np.round(row_sums), atol=tol):
        return None
    if np.any(row_sums < -tol) or np.any(row_sums > 1.0 + tol):
        return None

    if not np.allclose(Z, np.round(Z), atol=tol):
        return None

    assignments = np.full(Z.shape[0], -1, dtype=np.int32)
    assigned = row_sums > tol
    if np.any(assigned):
        assignments[assigned] = np.argmax(Z[assigned], axis=1).astype(np.int32, copy=False)
        rows = np.arange(Z.shape[0], dtype=np.int32)[assigned]
        if not np.allclose(Z[rows, assignments[assigned]], 1.0, atol=tol):
            return None
    return assignments


def apply_identity_cluster_kernel_sqrt(assignments: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Applies K^{1/2} for K = Z Z^T where Z encodes one-hot cluster membership.
    '''
    assignments = np.asarray(assignments, dtype=np.int32)
    y = np.asarray(y, dtype=float)
    if assignments.ndim != 1 or y.ndim != 1 or assignments.shape[0] != y.shape[0]:
        raise ValueError('assignments and y must be 1D arrays of the same length.')

    out = np.zeros_like(y, dtype=float)
    valid = assignments >= 0
    if not np.any(valid):
        return out

    cluster_sizes = np.bincount(assignments[valid])
    cluster_sums = np.bincount(assignments[valid], weights=y[valid], minlength=cluster_sizes.shape[0])
    out[valid] = cluster_sums[assignments[valid]] / np.sqrt(cluster_sizes[assignments[valid]])
    return out


def get_identity_cluster_kernel_trace(assignments: np.ndarray) -> float:
    '''
    Returns trace(K) / N for K = Z Z^T induced by one-hot cluster assignments.
    '''
    assignments = np.asarray(assignments, dtype=np.int32)
    if assignments.ndim != 1:
        raise ValueError('assignments must be a 1D array.')
    if assignments.shape[0] == 0:
        return 0.0
    return float(np.mean(assignments >= 0))


def _calibrate_random_fixed_loading_from_propagated(u_fixed: np.ndarray,
                                                    propagated: np.ndarray,
                                                    trace_random: float,
                                                    target_corr: float,
                                                    fixed_name: str,
                                                    random_name: str) -> float:
    '''
    Returns the latent loading needed for one random effect to attain a desired
    observed correlation with one fixed/replaced effect, given the propagated
    fixed latent field S_random @ y_fixed.
    '''
    if np.isclose(target_corr, 0.0):
        return 0.0

    if np.isclose(u_fixed.var(), 0.0):
        raise ValueError(
            f'Cannot correlate random effect {random_name} with zero-variance fixed effect {fixed_name}.'
        )

    propagated = np.asarray(propagated, dtype=float) - np.mean(propagated)
    propagated_var = float(propagated.var())
    if np.isclose(propagated_var, 0.0):
        warnings.warn(
            f'Fixed effect {fixed_name} has no overlap with the kernel of random effect {random_name}; '
            'the requested cross-effect correlation will be set to 0.',
            stacklevel=2,
        )
        return 0.0

    observed_cov = float(np.mean(u_fixed * propagated))
    max_corr = observed_cov / np.sqrt(float(u_fixed.var()) * propagated_var)
    max_corr = float(np.clip(max_corr, -1.0, 1.0))
    clipped_target = float(np.clip(target_corr, -abs(max_corr), abs(max_corr)))
    if not np.isclose(clipped_target, target_corr):
        warnings.warn(
            f'Requested correlation {target_corr:.3f} between {fixed_name} and {random_name} '
            f'exceeds the feasible magnitude {abs(max_corr):.3f}; clipping to {clipped_target:.3f}.',
            stacklevel=2,
        )

    numerator = clipped_target ** 2 * trace_random
    denominator = (observed_cov ** 2 / float(u_fixed.var())) - clipped_target ** 2 * (
        propagated_var - trace_random
    )

    if denominator <= 0:
        return float(np.sign(clipped_target))
    rho_sq = float(np.clip(numerator / denominator, 0.0, 1.0))
    return float(np.sign(clipped_target) * np.sqrt(rho_sq))


def _get_kappa(Ks: list[np.ndarray], Ss: list[np.ndarray]) -> np.ndarray:
    '''
    Returns the maximum attainable observed cross-effect correlation under the
    latent-kernel construction used below.
    '''
    M = len(Ks)
    trK = np.array([float(np.trace(K_i)) for K_i in Ks], dtype=float)
    kappa = np.eye(M, dtype=float)
    for i in range(M):
        for j in range(i + 1, M):
            denom = np.sqrt(trK[i] * trK[j])
            kij = 0.0 if np.isclose(denom, 0.0) else float(np.trace(Ss[i] @ Ss[j]) / denom)
            kij = float(np.clip(kij, 0.0, 1.0))
            kappa[i, j] = kappa[j, i] = kij
    return kappa


def _validate_random_effect_inputs(Zs: list[np.ndarray], As: list[np.ndarray],
                                   variances: list[float], names: Optional[list[str]],
                                   replace_random: Optional[list[np.ndarray]]):
    M = len(As)
    if M == 0:
        raise ValueError('At least one random effect must be provided.')
    if len(variances) != M:
        raise ValueError('variances must have the same length as As.')

    if Zs is None:
        Zs = [None] * M
    elif len(Zs) != M:
        raise ValueError('Zs must have the same length as As.')
    else:
        Zs = list(Zs)

    if names is None:
        names = [f'RE_{i}' for i in range(M)]
    elif len(names) != M:
        raise ValueError('names must have the same length as As.')
    else:
        names = list(names)

    if len(set(names)) != len(names):
        raise ValueError('Random effect names must be unique.')

    if replace_random is None:
        replace_random = [None] * M
    else:
        replace_random = list(replace_random)
        if len(replace_random) < M:
            replace_random = replace_random + [None] * (M - len(replace_random))
        elif len(replace_random) > M:
            raise ValueError('replace_random must have length at most M.')

    vars_arr = np.asarray(variances, dtype=float)
    if np.any(vars_arr < 0):
        raise ValueError('variances must be non-negative.')

    Zs_valid = []
    As_valid = []
    N = None
    for i in range(M):
        A_i = _standardize_correlation_matrix(As[i], f'As[{i}]')
        if Zs[i] is None:
            Z_i = np.eye(A_i.shape[0], dtype=float)
        else:
            Z_i = np.asarray(Zs[i], dtype=float)
            if Z_i.ndim != 2:
                raise ValueError(f'Zs[{i}] must be a 2D array.')
            if Z_i.shape[1] != A_i.shape[0]:
                raise ValueError(
                    f'Zs[{i}] has {Z_i.shape[1]} columns, but As[{i}] has size {A_i.shape[0]}.'
                )

        if N is None:
            N = Z_i.shape[0]
        elif Z_i.shape[0] != N:
            raise ValueError('All design matrices must have the same number of individuals.')

        if replace_random[i] is not None:
            x_i = np.asarray(replace_random[i], dtype=float)
            if x_i.ndim != 1 or x_i.shape[0] != N:
                raise ValueError(
                    f'replace_random[{i}] must be a length-{N} 1D array.'
                )
            replace_random[i] = x_i

        Zs_valid.append(Z_i)
        As_valid.append(A_i)

    return (Zs_valid, As_valid, vars_arr, names, replace_random, N)


def _calibrate_random_fixed_loading(u_fixed: np.ndarray, y_fixed: np.ndarray,
                                    S_random: np.ndarray, trace_random: float,
                                    target_corr: float,
                                    fixed_name: str, random_name: str) -> float:
    '''
    Returns the latent loading needed for one random effect to attain a desired
    observed correlation with one fixed/replaced effect.
    '''
    propagated = S_random @ y_fixed
    return _calibrate_random_fixed_loading_from_propagated(
        u_fixed=u_fixed,
        propagated=propagated,
        trace_random=trace_random,
        target_corr=target_corr,
        fixed_name=fixed_name,
        random_name=random_name,
    )


def get_random_effects(Zs: list[np.ndarray], As: list[np.ndarray], variances: list[float],
                       C: np.ndarray = None, names: list[str] = None,
                       replace_random: list[np.ndarray] = None,
                       debug: bool = False) -> dict:
    '''
    Generates one or more random effects with user-specified within-effect kernels
    and optional between-effect correlations.
    Parameters:
        Zs (list): List of N*K_i design matrices. If Zs[i] is None, the identity
            matrix is used.
        As (list): List of K_i*K_i cluster-level correlation matrices.
        variances (list): Target empirical variances for each realized effect.
        C (2D array): Target observed correlation matrix between realized effects.
            If None, effects are generated independently.
        names (list): Optional names for the realized effects.
        replace_random (list): Optional realized effect values to plug in directly.
            Each provided vector is centered and rescaled to the corresponding target
            variance before being used.
        debug (bool): If True, includes intermediate kernel information.
    Returns:
        random_effects (dict): Metadata plus realized values. The realized vectors are
            available both in `random_effects['u']` and by name in
            `random_effects['values']`.
    '''
    Zs, As, vars_arr, names, replace_random, N = _validate_random_effect_inputs(
        Zs=Zs,
        As=As,
        variances=variances,
        names=names,
        replace_random=replace_random,
    )
    M = len(names)
    rng = np.random.default_rng()

    fixed_idx = [i for i, values in enumerate(replace_random) if values is not None]
    random_idx = [i for i in range(M) if i not in fixed_idx]

    if C is not None:
        C = _standardize_correlation_matrix(C, 'C')
        if C.shape != (M, M):
            raise ValueError('C must have shape (M, M), where M is the number of random effects.')

    if C is None:
        us = [None] * M
        for i in range(M):
            if replace_random[i] is not None:
                us[i] = _center_and_scale_random_effect(replace_random[i], vars_arr[i], names[i])
                continue

            A_i = As[i]
            if A_i.shape[0] == 0:
                if np.isclose(vars_arr[i], 0.0):
                    us[i] = np.zeros(N, dtype=float)
                    continue
                raise ValueError(f'Random effect {names[i]} has no clusters but positive target variance.')

            L_i = np.linalg.cholesky(0.5 * (A_i + A_i.T) + 1e-12 * np.eye(A_i.shape[0]))
            cluster_scores = L_i @ rng.standard_normal(A_i.shape[0])
            values = Zs[i] @ cluster_scores
            us[i] = _center_and_scale_random_effect(values, vars_arr[i], names[i])

        values_by_name = {name: values for name, values in zip(names, us)}
        return {
            'name': names,
            'names': names,
            'var': vars_arr.tolist(),
            'variances': vars_arr.copy(),
            'corr': None,
            'Z': Zs,
            'A': As,
            'u': us,
            'values': values_by_name,
        }

    Ks = [Z_i @ A_i @ Z_i.T for Z_i, A_i in zip(Zs, As)]
    Ss = [psd_sqrt(K_i, clip=0.0) for K_i in Ks]
    trK = np.array([float(np.trace(K_i)) / N for K_i in Ks], dtype=float)

    for i in range(M):
        if trK[i] <= 1e-12 and not np.isclose(vars_arr[i], 0.0):
            raise ValueError(
                f'Random effect {names[i]} has a zero kernel and cannot realize positive variance.'
            )

    kappa = _get_kappa(Ks, Ss)
    us = [None] * M

    Y_fixed = np.zeros((N, len(fixed_idx)), dtype=float)
    for col, i in enumerate(fixed_idx):
        u_fixed = _center_and_scale_random_effect(replace_random[i], vars_arr[i], names[i])
        us[i] = u_fixed

        if np.isclose(vars_arr[i], 0.0):
            continue

        S_pinv = psd_sqrt(Ks[i], pinv=True)
        y_fixed = S_pinv @ (u_fixed / np.sqrt(vars_arr[i]))
        y_fixed = y_fixed - y_fixed.mean()
        y_var = float(y_fixed.var())
        if np.isclose(y_var, 0.0):
            raise ValueError(
                f'Replaced effect {names[i]} is inconsistent with its kernel and cannot be used for correlation.'
            )
        Y_fixed[:, col] = y_fixed / np.sqrt(y_var)

    R = np.zeros((len(random_idx), len(fixed_idx)), dtype=float)
    for row, i in enumerate(random_idx):
        for col, j in enumerate(fixed_idx):
            R[row, col] = _calibrate_random_fixed_loading(
                u_fixed=us[j],
                y_fixed=Y_fixed[:, col],
                S_random=Ss[i],
                trace_random=trK[i],
                target_corr=float(C[i, j]),
                fixed_name=names[j],
                random_name=names[i],
            )

    C_latent = np.eye(len(random_idx), dtype=float)
    for a, i in enumerate(random_idx):
        for b, j in enumerate(random_idx):
            if a >= b:
                continue
            kij = kappa[i, j]
            if kij <= 1e-12:
                clipped = 0.0
                raw_latent_target = 0.0
            else:
                raw_latent_target = float(C[i, j] / kij)
                clipped = float(np.clip(raw_latent_target, -1.0, 1.0))
            if kij <= 1e-12 and not np.isclose(C[i, j], 0.0):
                warnings.warn(
                    f'Random effects {names[i]} and {names[j]} have no kernel overlap; '
                    f'the requested correlation {C[i, j]:.3f} will be set to 0.',
                    stacklevel=2,
                )
            elif not np.isclose(clipped, raw_latent_target):
                warnings.warn(
                    f'Requested correlation between {names[i]} and {names[j]} is not feasible '
                    f'under their kernels; clipping to approximately {clipped * kij:.3f}.',
                    stacklevel=2,
                )
            C_latent[a, b] = C_latent[b, a] = clipped
    if len(random_idx) > 0:
        C_latent = nearest_correlation_matrix(C_latent)

    G_fixed = (Y_fixed.T @ Y_fixed) / N if fixed_idx else np.zeros((0, 0), dtype=float)
    residual_target = C_latent - R @ G_fixed @ R.T

    if len(random_idx) > 0:
        residual_diag = np.clip(np.diag(residual_target), 0.0, None)
        scales = np.sqrt(residual_diag)
        with np.errstate(divide='ignore', invalid='ignore'):
            Dinv = np.diag(np.where(scales > 0, 1.0 / scales, 0.0))
        residual_corr = Dinv @ residual_target @ Dinv if len(random_idx) > 0 else residual_target
        residual_corr = nearest_correlation_matrix(residual_corr) if len(random_idx) > 0 else residual_corr
        residual_cov = np.diag(scales) @ residual_corr @ np.diag(scales)
        L_random = np.linalg.cholesky(
            0.5 * (residual_cov + residual_cov.T) + 1e-12 * np.eye(len(random_idx))
        )
    else:
        residual_cov = np.zeros((0, 0), dtype=float)
        L_random = np.zeros((0, 0), dtype=float)

    latent_noise = rng.standard_normal(size=(N, len(random_idx))) if random_idx else np.zeros((N, 0))
    Y_random = Y_fixed @ R.T + latent_noise @ L_random.T

    for row, i in enumerate(random_idx):
        raw_values = Ss[i] @ Y_random[:, row]
        us[i] = _center_and_scale_random_effect(raw_values, vars_arr[i], names[i])

    values_by_name = {name: values for name, values in zip(names, us)}
    random_effects = {
        'name': names,
        'names': names,
        'var': vars_arr.tolist(),
        'variances': vars_arr.copy(),
        'corr': C.copy(),
        'Z': Zs,
        'A': As,
        'K': Ks,
        'u': us,
        'values': values_by_name,
    }

    if debug:
        component_matrix = np.column_stack(us)
        random_effects['debug'] = {
            'K': Ks,
            'S': Ss,
            'kappa': kappa,
            'C_input': C.copy(),
            'C_latent_random': C_latent,
            'latent_fixed': Y_fixed,
            'fixed_loading': R,
            'latent_residual_cov': residual_cov,
            'C_observed': np.corrcoef(component_matrix.T),
        }
    return random_effects

#################################
#### Heritability Estimation ####
#################################

#### Haseman-Elston Methods ####
def plot_HE_regression(A: np.ndarray, y: np.ndarray, bins: int = 5) -> None:
    '''
    Plots the Haseman-Elston regression line and the data points.
    Parameters:
        A (2D array): N*N matrix of pairwise genetic relatedness coefficients.
        y (1D array): N-length array of trait values. Should be standardized.
        bins (int): Number of bins to use for grouping relatedness coefficients. 95% confidence interval for each bin's mean is shown. Default is 5 bins. If 0, no binning is performed.
    '''
    relatedness = []
    sq_diff = []

    y = (y - y.mean()) / y.std() # standardizes trait values
    N = A.shape[0]
    # computes the squared difference between pairwise trait values
    # and the pairwise genetic relatedness coefficients
    for j in range(N):
        for k in range(N):
            if j >= k: # since matrix is symmetrical along diagonal
                continue
            relatedness.append( A[j,k] )
            sq_diff.append( (y[j] - y[k])**2 )
    
    relatedness = np.array(relatedness)
    sq_diff = np.array(sq_diff)
    # binning
    if bins > 1:
        # defines bin edges
        bin_edges = np.linspace(np.min(relatedness), np.max(relatedness), bins + 1)
        # assigns each x_val to a bin index
        x_bin_i = np.digitize(relatedness, bin_edges) - 1 # -1 to make indices 0-based
        x_bin_i[x_bin_i == bins] = bins - 1  # ensure the last bin is inclusive
        # computes mean x and y for each bin
        x_val = np.array([np.mean(relatedness[x_bin_i == i]) for i in range(bins)])
        y_val = np.array([np.mean(sq_diff[x_bin_i == i]) for i in range(bins)])
        # computes 95CI around each point
        y_val_se = np.array([np.std(sq_diff[x_bin_i == i]) / np.sqrt(np.sum(x_bin_i == i)) for i in range(bins)])
    else:
        x_val = relatedness
        y_val = sq_diff

    # plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(x_val, y_val, alpha=0.5)
    if bins > 1:
        plt.errorbar(x_val, y_val, yerr=1.96*y_val_se, fmt='o', color='black', capsize=5, label='95% CI')
    plt.xlabel('Genetic relatedness coefficient')
    plt.ylabel('Squared difference in trait values')
    plt.title('Haseman-Elston Regression')
    plt.show()

#### REstricted Maximum Likelihood (REML) Methods ####
from .reml import run_HEreg as run_HEreg
from .reml import run_HEreg as run_HE_regression
from .reml import run_REML as run_REML

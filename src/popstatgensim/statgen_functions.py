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
from typing import Tuple, Union

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

# The following function was written (almost) entirely by ChatGPT 5
def psd_sqrt(M, clip: float = 0.0, pinv: bool = False, eps: float = 1e-12):
    """
    Symmetric PSD square root via eigendecomposition.
    Returns S such that S @ S.T ≈ M (up to numerical error).
    'clip' floors eigenvalues (e.g., tiny negatives due to roundoff) to this value for the sqrt.
    'pinv' returns the pseudo inverse square root instead.
    """
    M = 0.5 * (M + M.T)  # symmetrize
    w, U = np.linalg.eigh(M)

    if not pinv:
        w_clipped = np.clip(w, clip, None)
        S = (U * np.sqrt(w_clipped)) @ U.T
        return 0.5 * (S + S.T)
    else:
        # relative threshold to decide which eigs to keep
        tau = eps * max(w.max(), 1.0)
        mask = w > tau
        invsqrt = np.zeros_like(w)
        invsqrt[mask] = 1.0 / np.sqrt(w[mask])
        S_pinv = (U * invsqrt) @ U.T
        return 0.5 * (S_pinv + S_pinv.T)

# The following function was written (almost) entirely by ChatGPT 5
def nearest_correlation_matrix(X, eps_eig=1e-12):
    """
    One-shot Higham-style projection to the nearest correlation-like matrix:
      1) symmetrize
      2) project to PSD by clipping eigenvalues
      3) renormalize to unit diagonal (convert covariance -> correlation)
    Ensures symmetric, PSD, and ones on diagonal.
    """
    X = 0.5 * (X + X.T)
    w, V = np.linalg.eigh(X)
    w = np.clip(w, eps_eig, None)  # make PD for a clean Cholesky downstream
    Y = (V * w) @ V.T
    d = np.sqrt(np.diag(Y))
    Dinv = np.diag(1.0 / d)
    Ccorr = Dinv @ Y @ Dinv
    Ccorr = 0.5 * (Ccorr + Ccorr.T)
    np.fill_diagonal(Ccorr, 1.0)
    return Ccorr

# The following function was written (almost) entirely by ChatGPT 5
def _get_kappa(Ks: list[np.ndarray],Ss: list[np.ndarray]) -> np.ndarray:
    # We want the *observed* across-individual Pearson correlation between u_i and u_j
    # to match the input C[i,j]. Under the LMC construction (below), its expectation is
    #   E[corr(u_i, u_j)] ≈ C_input[i,j] * kappa[i,j]
    # where: kappa[i,j] = tr(S_i S_j) / sqrt(tr(S_i S_i) tr(S_j S_j))
    # and: tr(S_i S_i) = tr(K_i) because S_i S_i^T = K_i.
    M = len(Ks)
    trKi = np.array([np.trace(Ks[i]) for i in range(M)], dtype=float)
    kappa = np.eye(M, dtype=float)
    for i in range(M):
        for j in range(i+1, M):
            kij = float(np.trace(Ss[i] @ Ss[j]) / np.sqrt(trKi[i] * trKi[j]))
            # numerical guard into [0,1]
            kij = min(max(kij, 0.0), 1.0)
            kappa[i, j] = kappa[j, i] = kij
    return kappa

# much of the code related to correlated random effects was written (almost) entirely by ChatGPT 5
def get_random_effects(Zs: list[np.ndarray], As: list[np.ndarray], variances: list[float],
                       C: np.ndarray = None, names: list[str] = None,
                       replace_random: list[np.ndarray] = None, debug: bool = False) -> dict:
    '''
    Computes random effects of clusters for a mixed model.
    Parameters:
        Zs (list): List of design matrices for random effects. Each matrix should be N*N_i, where N is the number of individuals and N_i is the number of clusters for that random effect. If None, assumes identity matrix of size N*N (each individual is its own cluster).
        As (list): List of correlation matrices for each random effect. Each matrix should be N_i*N_i.
        variances (list): List of variances for each random effect. For component i, random effects are drawn from a normal distribution with mean 0 and covariance given by variances[i] * As[i].
        C (2D array): Correlation matrix between random effects. Should be M*M, where M is the number of random effects. Default is None, meaning random effects are independent. For random effects to be correlated, the design matrices Zs must be the same for all random effects.
        names (list): List of names for each random effect. Default is None, meaning names are not used (instead the index is used).
        replace_random (list): List of length M containing 1-D arrays of length N. If an element is not None, it replaces the randomly generated random effects for that component with the provided values. Correlations between effects are maintained. Note that the algorithm assumes that the respective correlation matrix A provided describes the covariance structure of the provided values. Furthermore, the scores are scaled such that the final values have the variance provided in the `variances` list. Be warned that including more than one replaced random effect may struggle to reproduce the desired correlations between effects. If an element is None, that random effect is generated by the function. Default is None, meaning all random effects are generated randomly. 
        debug (bool): If True, output contains another dictionary inside it with information relevant for correlated random effects. Default is False.
    Returns:
        random_effects (dict): Dictionary of relevant pieces of random effects, where each value in the dictionary is a list of the same length as the number of random effects.
    '''
    M = len(As) # number of components
    # names components if not given
    if names is None:
        names = [f"RE_{i}" for i in range(M)]
    
    # checks if A or Z matrices are None
    for i in range(M):
        if As[i] is None:
            raise ValueError("As cannot contain None values. Each random effect must have a correlation matrix.")
        if Zs[i] is None:
            Zs[i] = np.identity(As[i].shape[0])

    # checks if user provided random effects to replace
    scores = [None] * M
    i_random = list(range(M)) # indices of random effects
    i_fixed = [] # indices of fixed effects
    if replace_random is not None:
        # this just ensures that the list is of length M, even if user provided a shorter list
        for i in range(M):
            if replace_random[i] is not None:
                scores[i] = replace_random[i]
                i_fixed.append(i)
                i_random.remove(i)

    rng = np.random.default_rng()
    # independent random effects
    if C is None:
        us = []
        # generates random effects for each cluster for each component
        for i in range(M):
            N_i = As[i].shape[0]
            z_i = rng.standard_normal(N_i) # ~ N(0, I_{N_i}), "z-score"
            L_i = np.linalg.cholesky(0.5*(As[i]+As[i].T) + 1e-12*np.eye(N_i)) # As[i] = L L^T
            u_i = np.sqrt(variances[i]) * (Zs[i] @ L_i @ z_i) # u_i ~ N(0, var_i Z_i A_i Z_i^T)
            us.append(u_i)

    # correlated random effects
    else:
        # checks if C is valid
        if C.shape[0] != M or C.shape[1] != M:
            raise ValueError("Correlation matrix must be of shape M*M, where M is the number of random effects.")
        
        ## step 1: build individual-level kernels & their square roots ##
        N = Zs[0].shape[0]
        vars_arr = np.asarray(variances, dtype=float)

        # Individual-level covariance *kernels* (correlations if As are correlations and Z is 0/1)
        #   K_i = Z_i A_i Z_i^T  (shape N x N)
        Ks = [Zs[i] @ As[i] @ Zs[i].T for i in range(M)]

        # Square roots S_i = K_i^{1/2}  (shape N x N)
        # These are the "feature maps" that carry the within-effect structure for effect i.
        Ss = [psd_sqrt(Ks[i], clip=0.0) for i in range(M)]

        kappa = _get_kappa(Ks, Ss)  # shape M x M
        # check if any requested (random) correlations are too high with LMC
        for a, i in enumerate(i_random):
            for b, j in enumerate(i_random):
                if a < b and abs(C[i,j]) > kappa[i,j] + 1e-12:
                    print(f"|C[{i},{j}]|={abs(C[i,j]):.3f} exceeds kappa={kappa[i,j]:.3f};"
                          f" will cap at ~{np.sign(C[i,j])*kappa[i,j]:.3f}")


        # Draw M latent standard normal fields over individuals (each is length-N)
        Z_latent = rng.standard_normal(size=(N, M))  # columns z_q
        us = [None] * M # stores random effects (including those that become fixed)
        R = np.zeros((M, M), dtype=float)

        # replaces columns of random effects with provided values if given
        for i in i_fixed:
            # centers provided scores if they're not zero-centered
            scores_i = scores[i] - scores[i].mean()
            S_p = psd_sqrt(Ks[i], pinv=True) # K_i^{+1/2}

            if np.std(scores_i) == 0:
                raise ValueError(f"Provided random effect for component {i} has zero variance.")
            # gets equivalent z-score, although this isn't necessarily var=1, but
            # later when generating u, the variance is correct
            z_scores_i = (S_p @ scores_i) / np.std(scores_i) 
            Z_latent[:, i] = z_scores_i # replaces z-score
            # stores fixed effect
            us[i] = np.sqrt(vars_arr[i]) * (Ss[i] @ Z_latent[:, i])

            nu = float((Z_latent[:, i] @ Z_latent[:, i]) / N) # empirical variance
            # determines the correlation between fixed and real effects
            for j in i_random:
                vj = Ss[j] @ Z_latent[:,i] # S_j z_fixed
                A  = float(us[i] @ vj / (np.linalg.norm(us[i]) + 1e-15))
                B  = float(vj @ vj)
                T  = float(np.trace(Ks[j]))
                c_star = float(C[i, j]) # desired observed corr with fixed effect
                c_max  = 0.0 if B == 0 else A / np.sqrt(B)
                c_tgt  = float(np.clip(c_star, -abs(c_max), abs(c_max)))
                denom  = A*A - c_tgt*c_tgt*(B - nu*T)
                if denom <= 0:
                    rho = np.sign(c_tgt) * 1.0
                else:
                    rho2 = np.clip((c_tgt*c_tgt)*T / denom, 0.0, 1.0)
                    rho  = np.sign(c_tgt) * np.sqrt(rho2)
                R[j, i] = rho
        # subsets to only (random x fixed) effects
        R = R[np.ix_(i_random, i_fixed)]

        # Target observed correlation is C; we need to "pre-whiten" it by kappa so that
        #   observed ≈ C_calibrated ∘ kappa  (∘ = Hadamard on the *M x M* effect space),
        # i.e., C_calibrated[i,j] = C[i,j] / kappa[i,j] for i != j.
        # only applicable to non-fixed effects
        C_cal = C.copy()
        for i in i_random:
            for j in i_random:
                if i == j:
                    C_cal[i, j] = 1.0
                else:
                    if kappa[i, j] > 0:
                        C_cal[i, j] = np.clip(C[i, j] / kappa[i, j], -1.0, 1.0)
                    else:
                        # If kappa=0, the two effects share no common modes; the observed corr must be ~0.
                        C_cal[i, j] = 0.0
        # subsets to only random effects
        C_cal = C_cal[np.ix_(i_random, i_random)]
        # Project to the nearest valid correlation matrix (symmetric, PSD, diag=1)
        C_cal = nearest_correlation_matrix(C_cal, eps_eig=1e-12)

        # Build Z_f (N x k) and its Gram G_f (k x k)
        Z_f = np.column_stack([Z_latent[:, i] for i in i_fixed]) if i_fixed else np.zeros((N,0))
        # (columns should already be mean ~0 from your centering; optionally enforce)
        if Z_f.size:
            Z_f = Z_f - Z_f.mean(axis=0, keepdims=True)
        G_f = (Z_f.T @ Z_f) / N  # fixed-latent covariance across individuals

        # R is (len(i_random) x len(i_fixed))
        # Target Y-level correlation among random effects after κ-cal:
        #   Cov(Y_random) target = C_cal
        # Contribution from fixed cols = R @ G_f @ R^T
        Res = C_cal - R @ G_f @ R.T

        # Ensure correct diagonals & PSD via correlation projection
        diag_left = np.clip(np.diag(Res), 0.0, None)
        s = np.sqrt(diag_left)
        with np.errstate(divide='ignore', invalid='ignore'):
            Dinv = np.diag(np.where(s > 0, 1.0/s, 0.0))
        Q = Dinv @ Res @ Dinv
        Q = nearest_correlation_matrix(Q, eps_eig=1e-12)
        Res_psd = np.diag(s) @ Q @ np.diag(s)

        L_random = np.linalg.cholesky(0.5*(Res_psd + Res_psd.T) + 1e-12*np.eye(len(i_random)))

        # Makes L lower-triangular by combining fixed and random effects
        L = np.zeros((M,M), dtype=float)
        L[np.ix_(i_fixed, i_fixed)] = np.eye(len(i_fixed))
        L[np.ix_(i_random, i_random)] = L_random
        L[np.ix_(i_random, i_fixed)] = R

        # Mix them across effects: Y = Z_latent @ L.T, so column i is y_i = sum_q L[i,q] z_q
        Y = Z_latent @ L.T  # shape (N, M)

        # Build the M random effects: u_i = sqrt(var_i) * S_i @ y_i
        us = []
        for i in range(M):
            u_i = np.sqrt(vars_arr[i]) * (Ss[i] @ Y[:, i])
            us.append(u_i) 
        
        if debug:
            debug_info = {
                'K': Ks,
                'S': Ss,
                'kappa': kappa,
                'C_calibrated': C_cal,
                'C_input': C,
                'C_observed': np.corrcoef(np.vstack(us)),
                'Z_latent': Z_latent,
                'Y': Y,
                'L': L
            }

    # creates a dictionary of random effects
    random_effects = {
        'name': names,
        'var': variances,
        'corr': C,
        'Z': Zs,
        'A': As,
        'u': us
    }
    if debug and C is not None:
        random_effects['debug'] = debug_info
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

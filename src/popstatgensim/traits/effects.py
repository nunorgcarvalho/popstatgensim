"""Effect classes used to build trait components."""

from __future__ import annotations

import copy
import warnings
from typing import Optional, Tuple, Union

import numpy as np

from . import effect_sampling as sampling
from . import random_effects as random_effects_utils


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

        effects_standardized, j_causal = sampling.generate_causal_effects(
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
            self.effects_per_allele = sampling.get_standardized_effects(
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
            G_std = sampling.get_G_std_for_effects(G, P=int(G.max()) if G.size > 0 else None)
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
            self.effects_per_allele = sampling.get_standardized_effects(
                self.effects_standardized, self.G_std, std2allelic=True
            )

        self.effects_standardized = sampling.get_standardized_effects(
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
        values = sampling.compute_genetic_value(G, self.effects_per_allele)

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

        Z = random_effects_utils.build_design_matrix_from_groups(groups)
        A = np.eye(Z.shape[1], dtype=float) if self.A is None else np.asarray(self.A, dtype=float)
        return (Z, A)

    def _generate_component_identity_cluster_fast(self, reference_values: np.ndarray,
                                                  Z: np.ndarray, A: np.ndarray) -> Optional[np.ndarray]:
        '''
        Fast path for identity cluster relationship matrices, which avoids dense
        N x N kernel eigendecompositions.
        '''
        if A is None or not random_effects_utils.is_identity_matrix(A):
            return None

        if Z is None:
            assignments = np.arange(reference_values.shape[0], dtype=np.int32)
        else:
            assignments = random_effects_utils.get_group_assignments_from_design(Z)
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
        propagated = random_effects_utils.apply_identity_cluster_kernel_sqrt(assignments, y_fixed)
        trace_random = random_effects_utils.get_identity_cluster_kernel_trace(assignments)
        rho = random_effects_utils._calibrate_random_fixed_loading_from_propagated(
            u_fixed=u_fixed,
            propagated=propagated,
            trace_random=trace_random,
            target_corr=self.r,
            fixed_name=self._reference_label(),
            random_name=self.name,
        )

        latent_noise = np.random.normal(size=reference_values.shape[0])
        latent_values = rho * y_fixed + np.sqrt(max(1.0 - rho * rho, 0.0)) * latent_noise
        raw_values = random_effects_utils.apply_identity_cluster_kernel_sqrt(assignments, latent_values)
        return random_effects_utils._center_and_scale_random_effect(raw_values, self.var, self.name)

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
            random_effects = random_effects_utils.get_random_effects(
                Zs=[Z],
                As=[A],
                variances=[self.var],
                names=[self.name],
            )
            return random_effects['values'][self.name]

        reference_name = f'{self.name}__reference'
        random_effects = random_effects_utils.get_random_effects(
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
        values = sampling.generate_noise_value(int(inputs['N']), self.var)
        if not self.force_var:
            return values

        current_var = values.var()
        if np.isclose(current_var, 0.0):
            if np.isclose(self.var, 0.0):
                return np.zeros_like(values)
            raise ValueError('Cannot rescale zero-variance noise component.')
        return values * np.sqrt(self.var / current_var)

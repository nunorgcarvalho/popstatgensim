"""Trait class and trait-level composition helpers."""

from __future__ import annotations

import copy
import warnings
from typing import Union

import numpy as np

from . import effect_sampling as sampling
from .effects import Effect, FixedEffect, GeneticEffect, NoiseEffect, RandomEffect
from .pgs import simulate_pgs_standardized_weights


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
                self.inputs['G_std'] = sampling.get_G_std_for_effects(G, P=int(G.max()) if G.size > 0 else None)
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
                self.inputs['G_par_std'] = sampling.get_G_std_for_effects(G_par, P=int(G_par.max()) if G_par.size > 0 else None)
            N_candidates.append(G_par.shape[0])

        fixed_input_names = {
            effect.input_name for effect in self.effects.values()
            if (
                isinstance(effect, FixedEffect)
                and effect.relation == 'self'
                and effect.uses_external_input(inputs=self.inputs, pop=self.pop)
            )
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
            elif isinstance(effect, RandomEffect):
                self.var_initial[name] = effect.var
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
            elif not isinstance(effect, (FixedEffect, RandomEffect)):
                raise ValueError(
                    f'effects[{name}] must be a GeneticEffect, FixedEffect, '
                    'RandomEffect, or NoiseEffect object.'
                )

            for input_name in effect.required_inputs:
                if input_name not in self.inputs:
                    raise ValueError(f"Missing required trait input '{input_name}' for effect {name}.")

            if isinstance(effect, FixedEffect):
                is_trait = effect._infer_is_trait(inputs=self.inputs, pop=self.pop, require_resolution=False)
                if effect.relation != 'self' and not is_trait:
                    raise ValueError(
                        f"Relation-backed fixed effect '{name}' currently requires a trait-backed source."
                    )

                if is_trait and self.pop is not None:
                    if effect._missing_past_source_population(pop=self.pop):
                        continue
                    source_pop = effect._get_source_population(
                        pop=self.pop,
                        current_trait_name=current_name,
                    )
                    if effect.input_name not in source_pop.traits:
                        raise ValueError(
                            f"Trait-backed fixed effect '{name}' requires source trait "
                            f"'{effect.input_name}', but it was not found."
                        )
                    if effect.input_component is not None:
                        source_trait = source_pop.traits[effect.input_name]
                        if effect.input_component not in source_trait.y_:
                            raise ValueError(
                                f"Trait-backed fixed effect '{name}' requires component "
                                f"'{effect.input_component}' in trait '{effect.input_name}', but it was not found."
                            )
                    if source_pop is self.pop and current_name in trait_names:
                        if effect.input_name == current_name:
                            raise ValueError(
                                f"Trait-backed fixed effect '{name}' in trait '{current_name}' "
                                'cannot depend on the same current-generation trait values.'
                            )
                        if trait_names.index(effect.input_name) >= trait_names.index(current_name):
                            raise ValueError(
                                f"Trait-backed fixed effect '{name}' in trait '{current_name}' "
                                f"must depend on an earlier trait; got '{effect.input_name}'."
                            )

            if isinstance(effect, RandomEffect):
                if effect.has_dynamic_definition() and self.pop is None:
                    raise ValueError(
                        f"Random effect '{name}' requires a Population object "
                        'to resolve its dynamic kernel definition.'
                    )

                if isinstance(effect.source, str) and effect.source_kind == 'relation' and self.pop is not None:
                    if effect.source not in self.pop.relations:
                        raise ValueError(
                            f"Random effect '{name}' references missing relation '{effect.source}'."
                        )

                if isinstance(effect.source, str) and effect.source_kind == 'trait':
                    if self.pop is None:
                        raise ValueError(
                            f"Random effect '{name}' references another trait in its kernel and "
                            'therefore requires a Population object.'
                        )
                    if effect.source not in self.pop.traits:
                        raise ValueError(
                            f"Random effect '{name}' references missing source trait '{effect.source}'."
                        )
                    if current_name is not None and effect.source == current_name:
                        raise ValueError(
                            f"Random effect '{name}' cannot use its own trait '{current_name}' as a kernel source."
                        )
                    if current_name in trait_names:
                        if trait_names.index(effect.source) >= trait_names.index(current_name):
                            raise ValueError(
                                f"Random effect '{name}' in trait '{current_name}' "
                                f"must use an earlier source trait; got '{effect.source}'."
                            )

                if not effect.has_reference():
                    continue

                if effect.reference_trait is None:
                    if effect.reference_component is None:
                        raise ValueError(
                            f"Random effect '{name}' must reference a component "
                            'when no reference_trait is provided.'
                        )
                    if effect.reference_component not in effect_order:
                        raise ValueError(
                            f"Random effect '{name}' references unknown same-trait "
                            f"component '{effect.reference_component}'."
                        )
                    if effect_order.index(effect.reference_component) >= idx:
                        raise ValueError(
                            f"Random effect '{name}' must come after its referenced "
                            f"same-trait component '{effect.reference_component}'."
                        )
                else:
                    if current_name is not None and effect.reference_trait == current_name:
                        raise ValueError(
                            f"Random effect '{name}' should use reference_trait=None "
                            'for same-trait component dependencies.'
                        )
                    if self.pop is None:
                        raise ValueError(
                            f"Random effect '{name}' references another trait and "
                            'therefore requires a Population object.'
                        )
                    if effect.reference_trait not in self.pop.traits:
                        raise ValueError(
                            f"Random effect '{name}' references missing trait "
                            f"'{effect.reference_trait}'."
                        )
                    reference_trait = self.pop.traits[effect.reference_trait]
                    if effect.reference_component is not None and effect.reference_component not in reference_trait.y_:
                        raise ValueError(
                            f"Random effect '{name}' references missing component "
                            f"'{effect.reference_component}' in trait '{effect.reference_trait}'."
                        )
                    if current_name in trait_names:
                        if trait_names.index(effect.reference_trait) >= trait_names.index(current_name):
                            raise ValueError(
                                f"Random effect '{name}' in trait '{current_name}' "
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

        component_names = list(self.y_.keys())
        if 'Eps' in self.y_:
            component_names = [name for name in component_names if name != 'Eps'] + ['Eps']
            self.y_ = {name: self.y_[name] for name in component_names}

        component_sum = np.zeros(N, dtype=float)
        for name in component_names:
            values = self.y_[name]
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

    def required_per_generation_update_effects(self) -> list[str]:
        '''
        Returns effect names that require per-generation trait updates during
        forward simulation to remain internally consistent.
        '''
        required = []
        for name, effect in self.effects.items():
            if effect.requires_per_generation_trait_updates(inputs=self.inputs, pop=self.pop):
                required.append(name)
        return required

    def requires_per_generation_updates(self) -> bool:
        '''
        Whether this trait contains any effects that require per-generation
        trait updates during forward simulation.
        '''
        return len(self.required_per_generation_update_effects()) > 0


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
        self.inputs['_current_trait_name'] = self.name
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
            self.inputs.pop('_current_trait_name', None)

        self.y = np.zeros(int(self.inputs['N']), dtype=float) if total is None else total
        self.type = 'composite'
        self._update_empirical_variances()
        self._update_initial_variances()

    def _restore_state(self, state: dict):
        self.pop = state['pop']
        self.name = state['name']
        self.y = state['y']
        self.y_ = state['y_']
        self.var = state['var']
        self.var_initial = state['var_initial']
        self.effects = state['effects']
        self.inputs = state['inputs']
        self.type = state['type']

    def _snapshot_state(self) -> dict:
        return {
            'pop': self.pop,
            'name': self.name,
            'y': self.y.copy(),
            'y_': {name: values.copy() for name, values in self.y_.items()},
            'var': copy.deepcopy(self.var),
            'var_initial': copy.deepcopy(self.var_initial),
            'effects': copy.deepcopy(self.effects),
            'inputs': copy.deepcopy(self.inputs),
            'type': self.type,
        }

    def _recompute_total_from_components(self):
        N = int(self.inputs['N'])
        total = np.zeros(N, dtype=float)
        for values in self.y_.values():
            total += values
        self.y = total
        self._update_empirical_variances()
        self._update_initial_variances()

    def _same_trait_dependents(self, component_name: str) -> list[str]:
        dependents = []
        for name, effect in self.effects.items():
            if not isinstance(effect, RandomEffect):
                continue
            if effect.reference_trait is None and effect.reference_component == component_name:
                dependents.append(name)
        return dependents

    def _remove_effect_in_place(self, name: str):
        self.effects.pop(name, None)
        self.y_.pop(name, None)
        self.var.pop(name, None)
        self.var_initial.pop(name, None)
        self._recompute_total_from_components()

    def _drop_random_effects_for_population_reshape(self, reason: str,
                                                    warn: bool = True) -> list[str]:
        removed = []
        pending = [
            name for name, effect in self.effects.items()
            if isinstance(effect, RandomEffect) and effect.should_drop_on_population_reshape()
        ]
        while pending:
            name = pending.pop(0)
            if name in removed or name not in self.effects:
                continue
            removed.append(name)
            for dependent in self._same_trait_dependents(name):
                if dependent not in removed and dependent not in pending:
                    pending.append(dependent)

        if removed and warn:
            warnings.warn(
                f"Trait '{self.name}' removed random effect(s) {removed} after {reason} because "
                'their kernels were defined from fixed matrices or arrays and cannot update safely '
                'to the new population state.',
                stacklevel=2,
            )

        for name in removed:
            self._remove_effect_in_place(name)
        return removed

    def _refresh_dynamic_random_effects(self) -> list[str]:
        refreshed = []
        self.inputs['_trait_components'] = self.y_
        try:
            for name, effect in self.effects.items():
                if not isinstance(effect, RandomEffect):
                    continue
                if not effect.has_dynamic_definition():
                    continue
                effect.refresh_from_inputs(self.inputs)
                self.y_[name] = effect.generate_component(self.inputs, pop=self.pop)
                refreshed.append(name)
        finally:
            self.inputs.pop('_trait_components', None)

        if refreshed:
            self._recompute_total_from_components()
        return refreshed

    def add_effect(self, effect: Effect, inputs: dict = None,
                   copy_inputs: bool = True, **kwargs):
        '''
        Adds one effect to an existing composite trait and updates the realized
        trait values without regenerating unaffected components.
        '''
        if self.type in {'fixed', 'permanent'}:
            raise ValueError(f"Cannot add effects to a {self.type} trait.")
        if not isinstance(effect, Effect):
            raise ValueError('effect must be an Effect object.')
        if effect.name in self.effects:
            raise ValueError(f"Trait already contains an effect named '{effect.name}'.")

        state = self._snapshot_state()
        try:
            self.effects[effect.name] = copy.deepcopy(effect)
            if inputs is not None or kwargs:
                self.update_inputs(inputs=inputs, copy_inputs=copy_inputs, **kwargs)

            self._validate_effect_definitions()
            self.inputs['_trait_components'] = self.y_
            try:
                effect_obj = self.effects[effect.name]
                effect_obj.refresh_from_inputs(self.inputs)
                self.y_[effect.name] = effect_obj.generate_component(self.inputs, pop=self.pop)
            finally:
                self.inputs.pop('_trait_components', None)

            self._recompute_total_from_components()
            self.validate()
        except Exception:
            self._restore_state(state)
            raise

    def remove_effect(self, name: str):
        '''
        Removes one effect from an existing composite trait and updates the
        realized trait values.
        '''
        if self.type in {'fixed', 'permanent'}:
            raise ValueError(f"Cannot remove effects from a {self.type} trait.")
        if name not in self.effects:
            raise ValueError(f"Trait does not contain an effect named '{name}'.")

        dependents = self._same_trait_dependents(name)
        if dependents:
            raise ValueError(
                f"Cannot remove effect '{name}' because the following effect(s) depend on it: {dependents}."
            )

        self._remove_effect_in_place(name)
        self.validate()

    def add_popstrat(self, variance: float, force_var: bool = True,
                     name: str = 'popstrat'):
        '''
        Convenience wrapper that adds a categorical subpopulation random effect
        using the population's permanent `subpop` trait.
        '''
        if self.pop is None:
            raise ValueError('add_popstrat requires the trait to belong to a Population.')
        if 'subpop' not in self.pop.traits:
            raise ValueError(
                "Population is missing the 'subpop' trait. "
                'Call SuperPopulation.add_subpop_trait() before add_popstrat().'
            )

        self.add_effect(
            RandomEffect(
                name=name,
                var=variance,
                force_var=force_var,
                source='subpop',
                source_kind='trait',
                type='categorical',
            )
        )

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
                values_unscaled = sampling.compute_genetic_value(G_current, effect.effects_per_allele)
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

    def simulate_PGS_weights(
        self,
        R2: float = None,
        N: int = None,
        h2_method: str = 'additive_effects',
        rng: np.random.Generator = None,
        seed: int = None,
    ) -> np.ndarray:
        r'''
        Simulates estimated standardized polygenic score weights for the additive
        genetic component.

        The returned weights are defined variant-wise as
        \hat{\beta}_j = \beta_j + \epsilon_j,
        where \beta_j is the true standardized additive effect of variant j from
        ``Trait.effects['A'].effects_standardized`` and
        \epsilon_j \sim \mathcal{N}(0, V_{\epsilon}).

        Two definitions of V_{\epsilon} are supported, depending on whether the
        user supplies ``R2`` or ``N`` (mutually exclusive):

        V_{\epsilon} = (h^2 / M) (h^2 / R^2 - 1)
        V_{\epsilon} = 1 / N

        Here ``h2`` is obtained from ``Trait.get_h2(method=h2_method)``
        and ``M`` is the number of variants. The R^2-based expression follows
        from

        R^2 = Corr(\hat{A}, Y)^2 = (V_A)^2 / (V_A + M / N)

        under the assumptions that
        1. all variants are independent from each other,
        2. variants are only correlated with phenotype through the additive
           genetic component (the ``'A'`` component),
        3. estimation noise is independent of the variant.

        Reference
        ---------
        Daetwyler et al. 2008, PLoS ONE.

        Parameters
        ----------
        R2 : float, optional
            Expected prediction R^2 of the polygenic score. Mutually exclusive
            with N.
        N : int, optional
            GWAS sample size used in the large-sample approximation
            V_{\epsilon} = 1 / N. Mutually exclusive with R2.
        h2_method : str, optional
            Method passed through to ``Trait.get_h2(method=...)`` when obtaining
            the heritability used in the noise-variance formula. Default is
            ``'additive_effects'``.
        rng : np.random.Generator, optional
            Random number generator used to draw estimation noise.
        seed : int, optional
            Seed used to initialize a new random generator when rng is not
            provided.

        Returns
        -------
        np.ndarray
            Estimated standardized PGS weights of length M.
        '''
        if 'A' not in self.effects:
            raise ValueError("Trait must contain an additive genetic effect 'A'.")

        beta = self.effects['A'].effects_standardized
        if beta is None:
            raise ValueError(
                "Trait.effects['A'] must have standardized additive effects stored."
            )

        h2 = self.get_h2(method=h2_method)
        return simulate_pgs_standardized_weights(
            beta=beta,
            h2=h2,
            R2=R2,
            N=N,
            rng=rng,
            seed=seed,
        )

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
        trait_new._drop_random_effects_for_population_reshape(reason='joining populations')

        trait_new.y_ = {}
        total = None
        trait_new.inputs['_trait_components'] = trait_new.y_
        trait_new.inputs['_current_trait_name'] = trait_new.name
        try:
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
        finally:
            trait_new.inputs.pop('_trait_components', None)
            trait_new.inputs.pop('_current_trait_name', None)

        trait_new.y = np.zeros(int(trait_new.inputs['N']), dtype=float) if total is None else total
        trait_new._update_empirical_variances()
        trait_new._update_initial_variances()
        return trait_new

    def index_trait(self, i_keep: np.ndarray, G: np.ndarray = None,
                    G_already_indexed: bool = False,
                    pop: Population = None) -> 'Trait':
        '''
        Returns a Trait object that contains only the specified individuals.
        '''
        trait_new = self.__class__.__new__(self.__class__)
        trait_new._initialize_empty()
        trait_new.pop = self.pop if pop is None else pop
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

        if trait_new.type == 'composite':
            trait_new._drop_random_effects_for_population_reshape(reason='subsetting a population')
            trait_new._refresh_dynamic_random_effects()
            trait_new.validate()
        else:
            trait_new._update_initial_variances()
        return trait_new

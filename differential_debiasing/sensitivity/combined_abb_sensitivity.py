"""
Combined A-BB sensitivity estimation using both static and dynamic bias measurements
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List, Callable

from .base import SensitivityEstimator
from .abb_sensitivity import ABBSensitivity
from .factor_analysis import FactorAnalysisSensitivity
from .empirical import EmpiricalSensitivity
from .domain_specific import DomainSpecificSensitivity
from .psychometric_reliability import PsychometricReliabilitySensitivity
from .schematic_adherence import SchematicAdherenceSensitivity
from .combined import CombinedSensitivity
from ..neighbors import BaseNeighborGenerator, HammingNeighborGenerator, FormattingNeighborGenerator, OrderNeighborGenerator


class CombinedABBSensitivity(SensitivityEstimator):
    """
    Combined A-BB sensitivity estimation using both static and dynamic bias measurements.
    
    This estimator provides system-level bias measurement by combining:
    1. Static measurements: Analysis of existing score patterns (psychometric, factor analysis, etc.)
    2. Dynamic measurements: Active probing with neighbor sampling (A-BB mechanism)
    
    The combination gives a comprehensive estimate of judge sensitivity that captures
    both systematic bias patterns and contextual bias triggers.
    """
    
    def __init__(self, 
                 static_estimators: Optional[List[Union[str, SensitivityEstimator]]] = None,
                 dynamic_generators: Optional[List[Union[str, BaseNeighborGenerator]]] = None,
                 combination_strategy: str = "conservative",
                 static_weights: Optional[List[float]] = None,
                 dynamic_weights: Optional[List[float]] = None,
                 judge_function: Optional[Callable] = None,
                 num_neighbors: int = 10,
                 **kwargs):
        """
        Initialize combined A-BB sensitivity estimator.
        
        Parameters:
        -----------
        static_estimators : List[str or SensitivityEstimator], optional
            Static bias measurement methods. Default: ["psychometric_reliability", "factor_analysis"]
        dynamic_generators : List[str or BaseNeighborGenerator], optional  
            Dynamic neighbor generators for A-BB. Default: ["hamming", "formatting"]
        combination_strategy : str
            How to combine measurements. Options:
            - "conservative": max(all_estimates) - most protective
            - "rms": sqrt(sum(estimate²)) - independent bias sources
            - "weighted": weighted average - balanced approach
            - "adaptive": choose strategy based on measurement agreement
        static_weights : List[float], optional
            Weights for static estimators (for weighted strategy)
        dynamic_weights : List[float], optional
            Weights for dynamic estimators (for weighted strategy)  
        judge_function : Callable, optional
            Judge function for dynamic measurements
        num_neighbors : int
            Number of neighbors per dynamic generator
        **kwargs : dict
            Additional parameters passed to sub-estimators
        """
        super().__init__(**kwargs)
        
        # Default static estimators
        if static_estimators is None:
            static_estimators = ["psychometric_reliability", "factor_analysis"]
        
        # Default dynamic generators  
        if dynamic_generators is None:
            dynamic_generators = ["hamming", "formatting"]
        
        self.static_estimators = self._create_static_estimators(static_estimators, **kwargs)
        self.dynamic_generators = self._create_dynamic_generators(dynamic_generators, **kwargs)
        self.combination_strategy = combination_strategy
        self.static_weights = static_weights
        self.dynamic_weights = dynamic_weights
        self.judge_function = judge_function
        self.num_neighbors = num_neighbors
        
        # Set up random number generator
        self.rng = np.random.RandomState(kwargs.get('random_seed'))
        
        # State tracking
        self._static_sensitivities = {}
        self._dynamic_sensitivities = {}
        self._combined_sensitivity = None
        self._measurement_details = {}
        
    def _create_static_estimators(self, estimators: List[Union[str, SensitivityEstimator]], **kwargs) -> Dict[str, SensitivityEstimator]:
        """Create static sensitivity estimators from list."""
        static_map = {
            "factor_analysis": FactorAnalysisSensitivity,
            "empirical": EmpiricalSensitivity,
            "domain_specific": DomainSpecificSensitivity,
            "psychometric_reliability": PsychometricReliabilitySensitivity,
            "schematic_adherence": SchematicAdherenceSensitivity,
            "combined": CombinedSensitivity,
        }
        
        result = {}
        for i, estimator in enumerate(estimators):
            if isinstance(estimator, SensitivityEstimator):
                result[f"static_{i}"] = estimator
            elif isinstance(estimator, str):
                if estimator not in static_map:
                    available = list(static_map.keys())
                    raise ValueError(f"Unknown static estimator '{estimator}'. Available: {available}")
                result[estimator] = static_map[estimator](**kwargs)
            else:
                raise ValueError(f"Invalid static estimator type: {type(estimator)}")
        
        return result
    
    def _create_dynamic_generators(self, generators: List[Union[str, BaseNeighborGenerator]], **kwargs) -> Dict[str, BaseNeighborGenerator]:
        """Create dynamic neighbor generators from list."""
        generator_map = {
            "hamming": HammingNeighborGenerator,
            "formatting": FormattingNeighborGenerator,
            "order": OrderNeighborGenerator,
        }
        
        # Add context-aware generators if available
        try:
            from ..neighbors.context_aware_neighbors import create_context_aware_generator
            context_aware_types = [
                "question_paraphrasing", "answer_perturbation", 
                "model_obfuscation", "contextual_variation"
            ]
            # Note: context-aware generators need judge_function, handled below
        except ImportError:
            context_aware_types = []
        
        result = {}
        for i, generator in enumerate(generators):
            if isinstance(generator, BaseNeighborGenerator):
                result[f"dynamic_{i}"] = generator
            elif isinstance(generator, str):
                # Check if it's a context-aware generator type
                if generator in context_aware_types:
                    if self.judge_function is None:
                        raise ValueError(f"Context-aware generator '{generator}' requires judge_function")
                    from ..neighbors.context_aware_neighbors import create_context_aware_generator
                    result[generator] = create_context_aware_generator(generator, self.judge_function, **kwargs)
                elif generator in generator_map:
                    result[generator] = generator_map[generator](**kwargs)
                else:
                    available = list(generator_map.keys()) + context_aware_types
                    raise ValueError(f"Unknown neighbor generator '{generator}'. Available: {available}")
            else:
                raise ValueError(f"Invalid generator type: {type(generator)}")
        
        return result
    
    def fit(self, judgments: Union[np.ndarray, pd.DataFrame, Dict], **kwargs) -> 'CombinedABBSensitivity':
        """
        Fit combined sensitivity by running all static and dynamic measurements.
        
        Parameters:
        -----------
        judgments : array-like, DataFrame, or Dict
            Judgment data/context for measurement
        **kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        CombinedABBSensitivity : Self for method chaining
        """
        # Handle judge function override
        if 'judge_function' in kwargs:
            self.judge_function = kwargs['judge_function']
        
        # Run static measurements
        self._fit_static_estimators(judgments, **kwargs)
        
        # Run dynamic measurements (if judge function available)
        if self.judge_function is not None:
            self._fit_dynamic_estimators(judgments, **kwargs)
        else:
            print("Warning: No judge function provided. Dynamic measurements skipped.")
        
        self._fitted = True
        return self
    
    def _fit_static_estimators(self, judgments: Union[np.ndarray, pd.DataFrame], **kwargs):
        """Fit all static sensitivity estimators."""
        self._static_sensitivities = {}
        
        for name, estimator in self.static_estimators.items():
            try:
                estimator.fit(judgments, **kwargs)
                # We'll get sensitivity in estimate() call
                self._static_sensitivities[name] = None  # Placeholder
            except Exception as e:
                print(f"Warning: Static estimator '{name}' failed: {e}")
                self._static_sensitivities[name] = None
    
    def _fit_dynamic_estimators(self, context: Union[Dict, pd.DataFrame], **kwargs):
        """Fit all dynamic neighbor generators."""
        self._dynamic_sensitivities = {}
        
        for name, generator in self.dynamic_generators.items():
            try:
                # Create A-BB estimator for this generator
                abb_estimator = ABBSensitivity(
                    judge_function=self.judge_function,
                    neighbor_generator=generator,
                    num_neighbors=self.num_neighbors,
                    random_seed=self.rng.get_state()[1][0]  # Get seed from our RNG
                )
                
                # Fit and store
                abb_estimator.fit(context, **kwargs)
                self._dynamic_sensitivities[name] = abb_estimator
                
            except Exception as e:
                print(f"Warning: Dynamic generator '{name}' failed: {e}")
                self._dynamic_sensitivities[name] = None
    
    def estimate(self, score_range: Optional[float] = None) -> float:
        """
        Estimate combined bias sensitivity from all measurements.
        
        Parameters:
        -----------
        score_range : float, optional
            Range of judgment scores for validation
            
        Returns:
        --------
        float : Combined sensitivity estimate
        """
        self._validate_fitted()
        
        # Collect all sensitivity estimates
        static_estimates = []
        dynamic_estimates = []
        
        # Determine proper score range if not provided
        if score_range is None:
            # Arena-Hard uses 1-5 Likert scale, so range is 4.0
            score_range = 4.0
            print(f"Using default Arena-Hard score range: {score_range}")
        
        # Get static estimates
        for name, estimator in self.static_estimators.items():
            if estimator is not None and name in self._static_sensitivities:
                try:
                    estimate = estimator.estimate(score_range)
                    if estimate > 0:
                        static_estimates.append(estimate)
                        self._measurement_details[f"static_{name}"] = estimate
                        print(f"Static {name}: {estimate:.4f} (range: {score_range})")
                except Exception as e:
                    print(f"Warning: Could not get estimate from static '{name}': {e}")
        
        # Get dynamic estimates
        for name, abb_estimator in self._dynamic_sensitivities.items():
            if abb_estimator is not None:
                try:
                    estimate = abb_estimator.estimate(score_range)
                    if estimate > 0:
                        dynamic_estimates.append(estimate)
                        self._measurement_details[f"dynamic_{name}"] = estimate
                        print(f"Dynamic {name}: {estimate:.4f} (range: {score_range})")
                except Exception as e:
                    print(f"Warning: Could not get estimate from dynamic '{name}': {e}")
        
        # Combine estimates
        all_estimates = static_estimates + dynamic_estimates
        if not all_estimates:
            raise ValueError("No valid sensitivity estimates obtained from any method")
        
        self._combined_sensitivity = self._combine_estimates(
            static_estimates, dynamic_estimates, all_estimates
        )
        
        # Validate against score range
        if score_range is not None and self._combined_sensitivity > score_range:
            self._combined_sensitivity = score_range
        
        self._bias_sensitivity = self._combined_sensitivity
        return self._combined_sensitivity
    
    def _combine_estimates(self, static_estimates: List[float], 
                          dynamic_estimates: List[float], 
                          all_estimates: List[float]) -> float:
        """Combine sensitivity estimates using specified strategy."""
        
        if self.combination_strategy == "conservative":
            # Use maximum estimate (most protective)
            return float(np.max(all_estimates))
        
        elif self.combination_strategy == "rms":
            # Root-mean-square combination (independent bias sources)
            return float(np.sqrt(np.mean(np.array(all_estimates) ** 2)))
        
        elif self.combination_strategy == "weighted":
            # Weighted average
            weights = self._get_weights(len(static_estimates), len(dynamic_estimates))
            if len(weights) != len(all_estimates):
                # Fall back to equal weights
                weights = np.ones(len(all_estimates)) / len(all_estimates)
            
            return float(np.sum(np.array(all_estimates) * np.array(weights)))
        
        elif self.combination_strategy == "adaptive":
            # Choose strategy based on measurement agreement
            return self._adaptive_combination(static_estimates, dynamic_estimates, all_estimates)
        
        else:
            raise ValueError(f"Unknown combination strategy: {self.combination_strategy}")
    
    def _get_weights(self, num_static: int, num_dynamic: int) -> List[float]:
        """Get weights for weighted combination."""
        weights = []
        
        # Static weights
        if self.static_weights is not None:
            weights.extend(self.static_weights[:num_static])
        else:
            # Equal weight for static estimators
            static_weight = 0.5 / max(num_static, 1) if num_static > 0 else 0
            weights.extend([static_weight] * num_static)
        
        # Dynamic weights
        if self.dynamic_weights is not None:
            weights.extend(self.dynamic_weights[:num_dynamic])
        else:
            # Equal weight for dynamic estimators
            dynamic_weight = 0.5 / max(num_dynamic, 1) if num_dynamic > 0 else 0
            weights.extend([dynamic_weight] * num_dynamic)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        return weights
    
    def _adaptive_combination(self, static_estimates: List[float], 
                             dynamic_estimates: List[float], 
                             all_estimates: List[float]) -> float:
        """Adaptively choose combination strategy based on measurement agreement."""
        
        if len(all_estimates) < 2:
            return all_estimates[0] if all_estimates else 0.0
        
        # Measure agreement (coefficient of variation)
        mean_estimate = np.mean(all_estimates)
        std_estimate = np.std(all_estimates)
        cv = std_estimate / mean_estimate if mean_estimate > 0 else float('inf')
        
        # If measurements agree well (low CV), use weighted average
        # If measurements disagree (high CV), use conservative approach
        if cv < 0.3:  # Low disagreement
            return self._combine_estimates(static_estimates, dynamic_estimates, all_estimates)
        else:  # High disagreement - be conservative
            return float(np.max(all_estimates))
    
    def get_measurement_breakdown(self) -> Dict[str, Any]:
        """
        Get detailed breakdown of all measurements.
        
        Returns:
        --------
        Dict[str, Any] : Detailed measurement information
        """
        return {
            "static_measurements": {name: self._measurement_details.get(f"static_{name}")
                                  for name in self.static_estimators.keys()},
            "dynamic_measurements": {name: self._measurement_details.get(f"dynamic_{name}")
                                   for name in self.dynamic_generators.keys()},
            "combination_strategy": self.combination_strategy,
            "combined_sensitivity": self._combined_sensitivity,
            "measurement_details": self._measurement_details.copy()
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostic information.
        
        Returns:
        --------
        Dict[str, Any] : Diagnostic information
        """
        diagnostics = super().get_diagnostics()
        
        if self._fitted:
            # Add measurement breakdown
            diagnostics.update(self.get_measurement_breakdown())
            
            # Add sub-estimator diagnostics
            static_diagnostics = {}
            for name, estimator in self.static_estimators.items():
                if estimator is not None:
                    try:
                        static_diagnostics[name] = estimator.get_diagnostics()
                    except:
                        static_diagnostics[name] = "Failed to get diagnostics"
            
            dynamic_diagnostics = {}
            for name, abb_estimator in self._dynamic_sensitivities.items():
                if abb_estimator is not None:
                    try:
                        dynamic_diagnostics[name] = abb_estimator.get_diagnostics()
                    except:
                        dynamic_diagnostics[name] = "Failed to get diagnostics"
            
            diagnostics.update({
                "static_estimator_diagnostics": static_diagnostics,
                "dynamic_estimator_diagnostics": dynamic_diagnostics,
                "num_successful_static": len([v for v in self._measurement_details.values() 
                                            if v is not None and "static_" in str(v)]),
                "num_successful_dynamic": len([v for v in self._measurement_details.values() 
                                             if v is not None and "dynamic_" in str(v)]),
                "has_judge_function": self.judge_function is not None
            })
        
        return diagnostics
    
    def validate_abb_constraint(self, tau: float, delta: float) -> Dict[str, Any]:
        """
        Validate A-BB constraint using combined sensitivity.
        
        Parameters:
        -----------
        tau : float
            Bias protection parameter
        delta : float
            Failure probability
            
        Returns:
        --------
        Dict[str, Any] : Validation results
        """
        if not self._fitted or self._combined_sensitivity is None:
            raise ValueError("Must fit estimator and estimate sensitivity before validating constraint")
        
        constraint_threshold = self._combined_sensitivity * np.sqrt(2.0 / delta)
        constraint_satisfied = tau > constraint_threshold
        
        return {
            "constraint_satisfied": constraint_satisfied,
            "tau": tau,
            "delta": delta,
            "combined_sensitivity": self._combined_sensitivity,
            "constraint_threshold": constraint_threshold,
            "margin": tau - constraint_threshold,
            "constraint_formula": "τ > Δ*₂_combined(f,D) * sqrt(2/δ)",
            "measurement_breakdown": self.get_measurement_breakdown()
        }
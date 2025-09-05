"""
Unified Conformal Prediction + Bias-Bounded Evaluation framework.

This module implements the ConformedBiasBoundedPredictor, which combines
conformal prediction's uncertainty quantification with BBE's bias protection
to create a robust evaluation framework that adapts noise injection based on
prediction uncertainty.
"""

import torch
import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List, Tuple, Callable
import warnings
from dataclasses import dataclass

try:
    import torchcp
    TORCHCP_AVAILABLE = True
except ImportError:
    TORCHCP_AVAILABLE = False
    warnings.warn("TorchCP not available. Install with: pip install torchcp")

from .conformal_sensitivity import ConformalSensitivityEstimator
from .multi_judge_conformal import MultiJudgeConformalPredictor
from .bias_bounded_calibration import BiaseBoundedCalibrationCreator, CalibrationSetConfig
from ..core.utils import calculate_noise_parameter, calculate_abb_noise_parameter
from .base import SensitivityEstimator


@dataclass
class CPBBEConfig:
    """Configuration for Conformal + BBE integration."""
    # Conformal prediction settings
    conformal_alpha: float = 0.1  # 90% coverage
    conformal_method: str = "split_cp"
    calibration_ratio: float = 0.2
    
    # BBE settings
    bbe_tau: float = 1.0
    bbe_delta: float = 0.1
    use_average_case: bool = True
    
    # Integration settings
    adaptation_strategy: str = "uncertainty_weighted"  # "uncertainty_weighted", "threshold_based", "hybrid"
    uncertainty_threshold: float = 0.5  # For threshold-based adaptation
    noise_adaptation_factor: float = 0.5  # How much to reduce noise for uncertain predictions
    
    # Multi-judge settings
    enable_multi_judge: bool = True
    judge_aggregation: str = "weighted_ensemble"
    judge_weight_strategy: str = "reliability_based"
    
    # Validation settings
    validate_exchangeability: bool = True
    min_coverage_requirement: float = 0.85  # Minimum coverage for validation


class ConformedBiasBoundedPredictor(SensitivityEstimator):
    """
    Unified predictor combining conformal prediction uncertainty with bias-bounded evaluation.
    
    This class implements adaptive noise injection where the amount of BBE noise is
    calibrated based on conformal prediction uncertainty estimates. High-uncertainty
    predictions receive less added noise (they're already uncertain), while confident
    predictions receive more noise to mask potential bias.
    """
    
    def __init__(self,
                 config: Optional[CPBBEConfig] = None,
                 base_model: Optional[torch.nn.Module] = None,
                 judge_functions: Optional[Dict[str, Callable]] = None,
                 device: str = "auto",
                 random_seed: Optional[int] = None,
                 **kwargs):
        """
        Initialize unified CP+BBE predictor.
        
        Parameters:
        -----------
        config : CPBBEConfig, optional
            Configuration for CP+BBE integration
        base_model : torch.nn.Module, optional
            Base model for conformal prediction
        judge_functions : Dict[str, Callable], optional
            Dictionary of judge functions for multi-judge scenarios
        device : str
            PyTorch device
        random_seed : int, optional
            Random seed for reproducibility
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        
        if not TORCHCP_AVAILABLE:
            raise ImportError("TorchCP is required but not available. Install with: pip install torchcp")
        
        self.config = config or CPBBEConfig()
        self.base_model = base_model
        self.judge_functions = judge_functions or {}
        self.random_seed = random_seed
        
        # Set up device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Set random seeds
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        self.rng = np.random.RandomState(random_seed)
        
        # Initialize components
        self.conformal_estimator = None
        self.multi_judge_predictor = None
        self.calibration_creator = None
        
        # State
        self._fitted = False
        self._uncertainty_scores = None
        self._adaptive_noise_parameters = None
        self._coverage_validation = None
    
    def fit(self, 
            judgments: Union[np.ndarray, pd.DataFrame, Dict], 
            judge_metadata: Optional[Dict] = None,
            **kwargs) -> 'ConformedBiasBoundedPredictor':
        """
        Fit the unified CP+BBE predictor.
        
        Parameters:
        -----------
        judgments : array-like, DataFrame, or Dict
            Judge evaluation data
        judge_metadata : Dict, optional
            Metadata about judges (model types, parameters, etc.)
        **kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        ConformedBiasBoundedPredictor : Self for method chaining
        """
        # Initialize components
        self._initialize_components()
        
        # Handle multi-judge vs single-judge scenarios
        if self._is_multi_judge_data(judgments):
            self._fit_multi_judge(judgments, judge_metadata, **kwargs)
        else:
            self._fit_single_judge(judgments, **kwargs)
        
        # Compute adaptive noise parameters
        self._compute_adaptive_noise_parameters()
        
        # Validate coverage if required
        if self.config.validate_exchangeability:
            self._validate_coverage()
        
        self._fitted = True
        return self
    
    def _initialize_components(self):
        """Initialize CP and BBE components."""
        # Conformal sensitivity estimator
        self.conformal_estimator = ConformalSensitivityEstimator(
            base_model=self.base_model,
            conformal_method=self.config.conformal_method,
            alpha=self.config.conformal_alpha,
            calibration_ratio=self.config.calibration_ratio,
            device=self.device,
            random_seed=self.random_seed
        )
        
        # Multi-judge conformal predictor
        if self.config.enable_multi_judge:
            self.multi_judge_predictor = MultiJudgeConformalPredictor(
                aggregation_method=self.config.judge_aggregation,
                conformal_method=self.config.conformal_method,
                alpha=self.config.conformal_alpha,
                judge_weight_strategy=self.config.judge_weight_strategy,
                calibration_ratio=self.config.calibration_ratio,
                device=self.device,
                random_seed=self.random_seed
            )
        
        # Bias-bounded calibration creator
        calibration_config = CalibrationSetConfig(
            tau=self.config.bbe_tau,
            delta=self.config.bbe_delta
        )
        self.calibration_creator = BiaseBoundedCalibrationCreator(
            config=calibration_config,
            random_seed=self.random_seed,
            device=self.device
        )
    
    def _is_multi_judge_data(self, judgments: Union[np.ndarray, pd.DataFrame, Dict]) -> bool:
        """Determine if data contains multiple judges."""
        if isinstance(judgments, dict) and len(judgments) > 1:
            return True
        elif isinstance(judgments, pd.DataFrame):
            score_columns = [col for col in judgments.columns if 'score' in col.lower()]
            return len(score_columns) > 1
        return False
    
    def _fit_multi_judge(self, judgments: Union[pd.DataFrame, Dict], judge_metadata: Optional[Dict], **kwargs):
        """Fit multi-judge scenario."""
        # Prepare judge predictions dictionary
        if isinstance(judgments, pd.DataFrame):
            score_columns = [col for col in judgments.columns if 'score' in col.lower()]
            judge_predictions = {col: judgments[col].values for col in score_columns}
        else:
            judge_predictions = {k: np.asarray(v) for k, v in judgments.items()}
        
        # Fit multi-judge conformal predictor
        self.multi_judge_predictor.fit(judge_predictions, judge_metadata)
        
        # Get aggregated predictions and uncertainties
        aggregated_predictions, uncertainties = self.multi_judge_predictor.predict()
        self._uncertainty_scores = uncertainties
        
        # Fit conformal estimator on aggregated predictions
        self.conformal_estimator.fit(aggregated_predictions, **kwargs)
        
        # Store judge information
        self._judge_weights = self.multi_judge_predictor.get_judge_weights()
        self._judge_names = list(judge_predictions.keys())
    
    def _fit_single_judge(self, judgments: Union[np.ndarray, pd.DataFrame], **kwargs):
        """Fit single-judge scenario."""
        # Fit conformal estimator directly
        self.conformal_estimator.fit(judgments, **kwargs)
        
        # Get uncertainty scores
        self._uncertainty_scores = self.conformal_estimator.get_uncertainty_scores()
        
        # Single judge scenario
        self._judge_weights = {"single_judge": 1.0}
        self._judge_names = ["single_judge"]
    
    def _compute_adaptive_noise_parameters(self):
        """Compute adaptive noise parameters based on uncertainty."""
        if self._uncertainty_scores is None:
            raise ValueError("Uncertainty scores not computed")
        
        # Normalize uncertainty scores
        uncertainty_normalized = (self._uncertainty_scores - np.min(self._uncertainty_scores)) / \
                               (np.max(self._uncertainty_scores) - np.min(self._uncertainty_scores) + 1e-8)
        
        # Compute base noise parameter
        base_sensitivity = self.conformal_estimator.estimate()
        
        if self.config.adaptation_strategy == "uncertainty_weighted":
            # Inverse relationship: high uncertainty -> low additional noise
            adaptation_factors = 1.0 - (uncertainty_normalized * self.config.noise_adaptation_factor)
            adaptation_factors = np.clip(adaptation_factors, 0.1, 1.0)  # Don't go below 10%
            
        elif self.config.adaptation_strategy == "threshold_based":
            # Binary adaptation based on threshold
            high_uncertainty_mask = uncertainty_normalized > self.config.uncertainty_threshold
            adaptation_factors = np.where(high_uncertainty_mask, 
                                        1.0 - self.config.noise_adaptation_factor,
                                        1.0)
        
        elif self.config.adaptation_strategy == "hybrid":
            # Combination of weighted and threshold approaches
            weighted_factors = 1.0 - (uncertainty_normalized * self.config.noise_adaptation_factor)
            threshold_mask = uncertainty_normalized > self.config.uncertainty_threshold
            
            # Use weighted for high uncertainty, full noise for low uncertainty
            adaptation_factors = np.where(threshold_mask, weighted_factors, 1.0)
            adaptation_factors = np.clip(adaptation_factors, 0.1, 1.0)
        
        else:
            raise ValueError(f"Unknown adaptation strategy: {self.config.adaptation_strategy}")
        
        # Compute adaptive noise parameters
        adaptive_noise_stds = []
        for adaptation_factor in adaptation_factors:
            adapted_sensitivity = base_sensitivity * adaptation_factor
            
            noise_std = calculate_noise_parameter(
                bias_sensitivity=adapted_sensitivity,
                tau=self.config.bbe_tau,
                delta=self.config.bbe_delta,
                use_average_case=self.config.use_average_case,
                n_samples=len(self._uncertainty_scores)
            )
            
            adaptive_noise_stds.append(noise_std)
        
        self._adaptive_noise_parameters = {
            'base_sensitivity': base_sensitivity,
            'uncertainty_scores': self._uncertainty_scores,
            'uncertainty_normalized': uncertainty_normalized,
            'adaptation_factors': adaptation_factors,
            'adaptive_noise_stds': np.array(adaptive_noise_stds),
            'mean_noise_std': np.mean(adaptive_noise_stds),
            'noise_reduction_achieved': 1.0 - np.mean(adaptation_factors)
        }
    
    def _validate_coverage(self):
        """Validate coverage properties of the unified predictor."""
        validation_results = {}
        
        # Get prediction intervals
        prediction_intervals = self.conformal_estimator.get_prediction_intervals()
        
        if isinstance(prediction_intervals, tuple):
            # Regression case
            lower, upper = prediction_intervals
            interval_widths = upper - lower
            
            # Check empirical coverage (simplified)
            # In practice, you'd use held-out test data
            validation_results['interval_widths'] = {
                'mean': float(np.mean(interval_widths)),
                'std': float(np.std(interval_widths)),
                'min': float(np.min(interval_widths)),
                'max': float(np.max(interval_widths))
            }
        else:
            # Classification case
            set_sizes = [len(pred_set) for pred_set in prediction_intervals]
            validation_results['set_sizes'] = {
                'mean': float(np.mean(set_sizes)),
                'std': float(np.std(set_sizes)),
                'min': float(np.min(set_sizes)),
                'max': float(np.max(set_sizes))
            }
        
        # Validate exchangeability of bias-bounded calibration
        if hasattr(self.calibration_creator, 'validate_exchangeability'):
            original_scores = self._extract_original_scores()
            exchangeability_results = self.calibration_creator.validate_exchangeability(original_scores)
            validation_results['exchangeability'] = exchangeability_results
        
        # Check uncertainty-noise relationship
        correlation = np.corrcoef(self._uncertainty_scores, 
                                self._adaptive_noise_parameters['adaptive_noise_stds'])[0, 1]
        validation_results['uncertainty_noise_correlation'] = float(correlation)
        
        # Overall validation score
        validation_score = 1.0  # Start optimistic
        
        if 'exchangeability' in validation_results:
            if not validation_results['exchangeability'].get('exchangeable', False):
                validation_score -= 0.3
        
        if abs(correlation) < 0.3:  # Expect some negative correlation
            validation_score -= 0.2
        
        validation_results['overall_validation_score'] = validation_score
        validation_results['passes_validation'] = validation_score >= 0.7
        
        self._coverage_validation = validation_results
    
    def _extract_original_scores(self) -> np.ndarray:
        """Extract original scores for validation."""
        # This is a simplified extraction - in practice, you'd store original data
        if hasattr(self.conformal_estimator, '_base_predictions'):
            return self.conformal_estimator._base_predictions.flatten()
        else:
            # Fallback: use uncertainty scores as proxy
            return self._uncertainty_scores
    
    def estimate(self, score_range: Optional[float] = None) -> float:
        """
        Estimate adaptive bias sensitivity.
        
        Parameters:
        -----------
        score_range : float, optional
            Range of judgment scores for validation
            
        Returns:
        --------
        float : Uncertainty-adapted sensitivity estimate
        """
        self._validate_fitted()
        
        if self._adaptive_noise_parameters is None:
            raise ValueError("Adaptive noise parameters not computed")
        
        # Return mean adapted sensitivity
        base_sensitivity = self._adaptive_noise_parameters['base_sensitivity']
        mean_adaptation = np.mean(self._adaptive_noise_parameters['adaptation_factors'])
        adapted_sensitivity = base_sensitivity * mean_adaptation
        
        # Validate against score range if provided
        if score_range is not None and adapted_sensitivity > score_range:
            print(f"⚠️ Adapted sensitivity {adapted_sensitivity:.4f} exceeds score range {score_range:.4f}")
            adapted_sensitivity = min(adapted_sensitivity, score_range)
        
        return float(adapted_sensitivity)
    
    def predict_with_adaptive_noise(self, 
                                  scores: np.ndarray,
                                  return_intervals: bool = True) -> Dict[str, np.ndarray]:
        """
        Apply adaptive noise injection to new scores.
        
        Parameters:
        -----------
        scores : np.ndarray
            Scores to be debiased
        return_intervals : bool
            Whether to return conformal prediction intervals
            
        Returns:
        --------
        Dict[str, np.ndarray] : Dictionary containing debiased scores and optional intervals
        """
        self._validate_fitted()
        
        # Get uncertainty estimates for new scores
        if len(scores) == len(self._uncertainty_scores):
            # Assume same length means same samples
            uncertainty_estimates = self._uncertainty_scores
            adaptive_noise_stds = self._adaptive_noise_parameters['adaptive_noise_stds']
        else:
            # Estimate uncertainties for new scores (simplified approach)
            # In practice, you'd rerun conformal prediction
            mean_uncertainty = np.mean(self._uncertainty_scores)
            uncertainty_estimates = np.full(len(scores), mean_uncertainty)
            mean_noise_std = self._adaptive_noise_parameters['mean_noise_std']
            adaptive_noise_stds = np.full(len(scores), mean_noise_std)
        
        # Generate adaptive noise
        adaptive_noise = np.array([
            self.rng.normal(0, noise_std) for noise_std in adaptive_noise_stds
        ])
        
        # Apply noise
        debiased_scores = scores + adaptive_noise
        
        result = {
            'original_scores': scores,
            'debiased_scores': debiased_scores,
            'uncertainty_estimates': uncertainty_estimates,
            'adaptive_noise': adaptive_noise,
            'noise_stds': adaptive_noise_stds
        }
        
        if return_intervals:
            # Get conformal prediction intervals (simplified)
            prediction_intervals = self.conformal_estimator.get_prediction_intervals()
            if isinstance(prediction_intervals, tuple):
                result['lower_bounds'] = prediction_intervals[0]
                result['upper_bounds'] = prediction_intervals[1]
            else:
                result['prediction_sets'] = prediction_intervals
        
        return result
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics about the unified predictor."""
        diagnostics = super().get_diagnostics()
        
        if self._fitted:
            diagnostics.update({
                "framework": "Conformal+BBE",
                "conformal_alpha": self.config.conformal_alpha,
                "conformal_coverage": 1 - self.config.conformal_alpha,
                "bbe_tau": self.config.bbe_tau,
                "bbe_delta": self.config.bbe_delta,
                "adaptation_strategy": self.config.adaptation_strategy,
                "multi_judge_enabled": self.config.enable_multi_judge,
                "num_judges": len(self._judge_names),
                "judge_names": self._judge_names.copy(),
            })
            
            if hasattr(self, '_judge_weights'):
                diagnostics["judge_weights"] = self._judge_weights.copy()
            
            if self._adaptive_noise_parameters:
                diagnostics.update({
                    "base_sensitivity": self._adaptive_noise_parameters['base_sensitivity'],
                    "mean_uncertainty": float(np.mean(self._uncertainty_scores)),
                    "std_uncertainty": float(np.std(self._uncertainty_scores)),
                    "mean_noise_std": self._adaptive_noise_parameters['mean_noise_std'],
                    "noise_reduction_achieved": self._adaptive_noise_parameters['noise_reduction_achieved'],
                    "adaptation_factor_range": [
                        float(np.min(self._adaptive_noise_parameters['adaptation_factors'])),
                        float(np.max(self._adaptive_noise_parameters['adaptation_factors']))
                    ]
                })
            
            if self._coverage_validation:
                diagnostics["validation"] = self._coverage_validation.copy()
            
            # Component diagnostics
            if self.conformal_estimator:
                diagnostics["conformal_estimator"] = self.conformal_estimator.get_diagnostics()
            
            if self.multi_judge_predictor and self.multi_judge_predictor.fitted_:
                diagnostics["multi_judge_predictor"] = self.multi_judge_predictor.get_diagnostics()
        
        return diagnostics
    
    def compare_with_baseline(self, baseline_method: str = "fixed_noise") -> Dict[str, Any]:
        """
        Compare adaptive approach with baseline methods.
        
        Parameters:
        -----------
        baseline_method : str
            Baseline to compare against: "fixed_noise", "no_noise", "uniform_cp"
            
        Returns:
        --------
        Dict[str, Any] : Comparison results
        """
        self._validate_fitted()
        
        if baseline_method == "fixed_noise":
            # Fixed noise BBE
            base_sensitivity = self._adaptive_noise_parameters['base_sensitivity']
            fixed_noise_std = calculate_noise_parameter(
                bias_sensitivity=base_sensitivity,
                tau=self.config.bbe_tau,
                delta=self.config.bbe_delta,
                use_average_case=self.config.use_average_case,
                n_samples=len(self._uncertainty_scores)
            )
            
            baseline_noise_stds = np.full(len(self._uncertainty_scores), fixed_noise_std)
            
        elif baseline_method == "no_noise":
            baseline_noise_stds = np.zeros(len(self._uncertainty_scores))
            
        elif baseline_method == "uniform_cp":
            # Uniform conformal prediction without BBE
            uniform_uncertainty = np.mean(self._uncertainty_scores)
            baseline_noise_stds = np.full(len(self._uncertainty_scores), uniform_uncertainty * 0.1)
        
        else:
            raise ValueError(f"Unknown baseline method: {baseline_method}")
        
        # Compare noise levels
        adaptive_noise_stds = self._adaptive_noise_parameters['adaptive_noise_stds']
        
        comparison = {
            "baseline_method": baseline_method,
            "adaptive_mean_noise": float(np.mean(adaptive_noise_stds)),
            "baseline_mean_noise": float(np.mean(baseline_noise_stds)),
            "noise_reduction": float(1 - np.mean(adaptive_noise_stds) / np.mean(baseline_noise_stds)),
            "adaptive_std_noise": float(np.std(adaptive_noise_stds)),
            "baseline_std_noise": float(np.std(baseline_noise_stds)),
            "correlation_with_uncertainty": float(np.corrcoef(self._uncertainty_scores, adaptive_noise_stds)[0, 1]),
            "efficiency_gain": float(np.mean(baseline_noise_stds) / np.mean(adaptive_noise_stds))
        }
        
        return comparison
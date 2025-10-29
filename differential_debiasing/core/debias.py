"""
Core differential debiasing implementation
"""

import numpy as np
import pandas as pd
import warnings
from typing import Union, Optional, Dict, Any, List
from .utils import (
    normalize_judgments,
    denormalize_judgments,
    calculate_abb_noise_parameter,
    validate_input_array,
    clip_to_range,
    check_bias_parameters,
    calculate_effective_alpha,
    build_abb_precheck,
)
from .sensitivity_profiles import SensitivityProfile, SensitivityProfileManager, get_formatting_sensitivity
from ..sensitivity.base import SensitivityEstimator
from ..sensitivity.psychometric_reliability import PsychometricReliabilitySensitivity
from ..sensitivity.schematic_adherence import SchematicAdherenceSensitivity
from ..sensitivity.abb_sensitivity import ABBSensitivity
from ..sensitivity.combined_abb_sensitivity import CombinedABBSensitivity
from ..sensitivity.profile_enhanced_abb import ProfileEnhancedABBSensitivity
from ..sensitivity.fixed import FixedSensitivityEstimator


class DifferentialDebias:
    """
    Main class for applying differential debiasing to LLM judge evaluations.
    
    This class implements the bias-bounded mechanism from "Bias-Bounded LLM Judge 
    Mechanisms: A Differential Privacy Approach", providing formal guarantees that
    systematic bias patterns are indistinguishable from calibrated noise.
    """
    
    def __init__(self, 
                 tau: float = 0.5,
                 delta: float = 0.05,
                 sensitivity_estimator: Optional[Union[str, SensitivityEstimator]] = "fixed",
                 sensitivity_profile: Optional[Union[SensitivityProfile, str, Dict[str, float]]] = None,
                 profile_dir: Optional[str] = None,
                 use_average_case: bool = True,
                 random_seed: Optional[int] = None,
                 dimensionality: Optional[int] = None,
                 # Shrinkage / contraction options
                 shrink_alpha: Optional[float] = None,
                 target_tau: Optional[float] = None,
                 shrink_center: str = "mean",
                 **estimator_kwargs):
        """
        Initialize differential debiasing mechanism.
        
        Parameters:
        -----------
        tau : float
            Bias protection parameter. Lower values provide stronger protection.
            τ = 0.5 means bias patterns can be at most 1.65× stronger than random.
        delta : float
            Failure probability. Probability that bias protection fails.
        sensitivity_estimator : str or SensitivityEstimator
            Method to estimate bias sensitivity. Supported options include:
            - "psychometric_reliability": Psychometric reliability analysis (Cronbach's α + CLR + HTMT)
            - "schematic_adherence": Schematic adherence analysis (linear/polynomial regression)
            - "abb": A-BB mechanism with neighbor sampling and RMS sensitivity
            - "combined_abb": Combined static + dynamic A-BB measurements
            - "profile_enhanced_abb": Combined A-BB with cached profile values
            - "fixed": Use a fixed, externally provided sensitivity value (default)
            - Custom SensitivityEstimator instance
        sensitivity_profile : SensitivityProfile, str, or Dict[str, float], optional
            Pre-computed sensitivity values to use instead of/alongside estimator.
            Can be:
            - SensitivityProfile instance
            - Judge name (str) to load profile from profile_dir
            - Dict with keys like 'formatting_sensitivity', 'hamming_sensitivity', etc.
        profile_dir : str, optional
            Directory containing sensitivity profile files (default: "sensitivity_profiles")
        use_average_case : bool
            Whether to use average-case bounds for datasets (recommended)
            Note: Ignored when using A-BB mechanism (abb estimator)
        random_seed : int, optional
            Random seed for reproducible noise generation
        dimensionality : int, optional
            Dimensionality of judgment space for A-BB mechanism. 
            If None, will be inferred from data. Required for A-BB estimator.
        **estimator_kwargs : dict
            Additional arguments passed to sensitivity estimator
        """
        # Validate bias parameters
        check_bias_parameters(tau, delta)
        
        self.tau = tau
        self.delta = delta
        self.use_average_case = use_average_case
        self.random_seed = random_seed
        self.dimensionality = dimensionality
        # Shrinkage parameters
        self.shrink_alpha = shrink_alpha
        self.target_tau = target_tau
        self.shrink_center = shrink_center
        self._effective_alpha: Optional[float] = None
        # Optional shrinkage centers (computed lazily when applicable)
        self._holdout_mu_norm: Optional[float] = None
        self._ema_mu_norm: Optional[float] = None
        
        # Set up random number generator
        self.rng = np.random.RandomState(random_seed)
        
        # Handle sensitivity profiles
        self.sensitivity_profile = self._load_sensitivity_profile(
            sensitivity_profile, profile_dir or "sensitivity_profiles"
        )
        
        # Initialize sensitivity estimator
        # Pass profile information to supported estimators
        if self.sensitivity_profile and sensitivity_estimator in ['profile_enhanced_abb', 'combined_abb']:
            estimator_kwargs['sensitivity_profile'] = self.sensitivity_profile
        
        if isinstance(sensitivity_estimator, str) and sensitivity_estimator == "fixed":
            estimator_kwargs.setdefault("fixed_sensitivity_value", 1.0)
        
        self.sensitivity_estimator = self._create_sensitivity_estimator(
            sensitivity_estimator, **estimator_kwargs
        )
        
        # State tracking
        self._fitted = False
        self._bias_sensitivity = None
        self._sigma = None
        self._score_min = None
        self._score_max = None
        self._original_range = None
        
    def _create_sensitivity_estimator(self, 
                                    estimator: Union[str, SensitivityEstimator],
                                    **kwargs) -> SensitivityEstimator:
        """Create sensitivity estimator from string or instance."""
        if isinstance(estimator, SensitivityEstimator):
            return estimator
        
        estimator_map = {
            "psychometric_reliability": PsychometricReliabilitySensitivity,
            "schematic_adherence": SchematicAdherenceSensitivity,
            "abb": ABBSensitivity,
            "combined_abb": CombinedABBSensitivity,
            "profile_enhanced_abb": ProfileEnhancedABBSensitivity,
            "fixed": FixedSensitivityEstimator,
        }
        
        if estimator not in estimator_map:
            available = list(estimator_map.keys())
            raise ValueError(f"Unknown sensitivity estimator '{estimator}'. "
                           f"Available: {available}")
        
        return estimator_map[estimator](**kwargs)
    
    def _load_sensitivity_profile(self, 
                                profile: Optional[Union[SensitivityProfile, str, Dict[str, float]]],
                                profile_dir: str) -> Optional[SensitivityProfile]:
        """Load sensitivity profile from various input formats."""
        if profile is None:
            return None
        
        if isinstance(profile, SensitivityProfile):
            return profile
        
        if isinstance(profile, str):
            # Load profile by judge name
            manager = SensitivityProfileManager(profile_dir)
            loaded_profile = manager.load_profile(profile)
            if loaded_profile is None:
                warnings.warn(f"Could not load sensitivity profile for judge '{profile}' from {profile_dir}")
            return loaded_profile
        
        if isinstance(profile, dict):
            # Create profile from dict
            profile_data = {
                'judge_name': 'custom',
                'profile_version': '1.0',
                'created_date': None,
                'formatting_sensitivity': profile.copy(),
                'measurement_metadata': {'source': 'manual_dict'}
            }
            return SensitivityProfile(profile_data)
        
        raise ValueError(f"Unsupported sensitivity_profile type: {type(profile)}")
    
    def get_sensitivity_from_profile(self, sensitivity_type: str) -> Optional[float]:
        """Get a specific sensitivity value from the loaded profile."""
        if self.sensitivity_profile is None:
            return None
        
        if sensitivity_type == 'formatting':
            return self.sensitivity_profile.get_formatting_sensitivity()
        
        # Future: Add other sensitivity types (hamming, etc.)
        return None
    
    def fit(self, judgments: Union[np.ndarray, pd.DataFrame, List], **kwargs) -> 'DifferentialDebias':
        """
        Fit the debiasing mechanism to judgment data.
        
        This step estimates bias sensitivity and calculates noise parameters.
        
        Parameters:
        -----------
        judgments : array-like or DataFrame
            Judgment data to analyze for bias sensitivity
        **kwargs : dict
            Additional arguments passed to sensitivity estimator
            
        Returns:
        --------
        DifferentialDebias : Self for method chaining
        """
        # Validate input but preserve DataFrame for estimators that need it
        if isinstance(judgments, pd.DataFrame):
            # Keep as DataFrame for new estimators that need factor columns
            if judgments.empty:
                raise ValueError("judgments DataFrame cannot be empty")
        else:
            # Convert to numpy array for traditional estimators
            judgments = validate_input_array(judgments, "judgments")
        
        # Store original score range for later use
        if isinstance(judgments, pd.DataFrame):
            # For DataFrames, try to infer score range from target column
            score_cols = [col for col in judgments.columns if 'score' in col.lower()]
            if score_cols:
                scores = judgments[score_cols[0]].dropna()
                if len(scores) > 0:
                    self._score_min = float(scores.min())
                    self._score_max = float(scores.max())
            else:
                # Fallback to 1-10 scale
                self._score_min, self._score_max = 1.0, 10.0
        else:
            # For arrays, use actual range
            self._score_min = float(judgments.min())
            self._score_max = float(judgments.max())
        
        self._original_range = self._score_max - self._score_min
        
        if self._original_range <= 0:
            # Handle identical judgments by adding minimal range
            self._original_range = 0.1  # Minimal range for noise application
            self._score_max = self._score_min + 0.1
        
        # Fit sensitivity estimator
        self.sensitivity_estimator.fit(judgments, **kwargs)
        
        # Estimate bias sensitivity
        self._bias_sensitivity = self.sensitivity_estimator.estimate(self._original_range)

        if self._bias_sensitivity <= 0:
            raise ValueError("Bias sensitivity must be positive")

        # Determine effective shrinkage alpha from target_tau if requested
        self._effective_alpha = calculate_effective_alpha(
            bias_sensitivity=float(self._bias_sensitivity),
            score_range=float(self._original_range),
            tau=float(self.tau),
            delta=float(self.delta),
            shrink_alpha=self.shrink_alpha,
            target_tau=self.target_tau,
        )
        
        # Calculate noise parameter (will be applied per-transform)
        self._fitted = True
        
        return self
    
    def transform(self, 
                 judgments: Union[np.ndarray, List],
                 score_min: Optional[float] = None,
                 score_max: Optional[float] = None) -> np.ndarray:
        """
        Apply differential debiasing to judgment scores.
        
        Parameters:
        -----------
        judgments : array-like
            Judgment scores to debias
        score_min : float, optional
            Minimum score value. If None, uses value from fit()
        score_max : float, optional  
            Maximum score value. If None, uses value from fit()
            
        Returns:
        --------
        np.ndarray : Debiased judgment scores
        """
        if not self._fitted:
            raise ValueError("Must call fit() before transform()")
        
        # Validate input
        judgments = validate_input_array(judgments, "judgments")
        
        # Use score range from fit() if not provided
        if score_min is None:
            score_min = self._score_min
        if score_max is None:
            score_max = self._score_max
            
        if score_min >= score_max:
            raise ValueError("score_max must be greater than score_min")
        
        # Normalize judgments to [0, 1]
        normalized_judgments, _, _ = normalize_judgments(judgments, score_min, score_max)
        
        # Calculate noise parameter using the A-BB formulation; default dimensionality falls back to sample size
        if self.dimensionality is None:
            self.dimensionality = len(judgments)

        rms_sensitivity = (self._bias_sensitivity / (score_max - score_min)) * float(self._effective_alpha or 1.0)
        sigma = calculate_abb_noise_parameter(
            rms_sensitivity=rms_sensitivity,
            tau=self.tau,
            delta=self.delta,
            dimensionality=self.dimensionality,
        )

        # Generate multivariate Gaussian noise (diagonal covariance -> i.i.d. normals)
        noise = self.rng.normal(loc=0.0, scale=sigma, size=len(judgments))
        # Apply shrinkage mapping in normalized space before adding noise (per Proposition)
        alpha = float(self._effective_alpha or 1.0)
        base_normalized = normalized_judgments
        if alpha < 1.0:
            mode = (self.shrink_center or "mean").lower()
            if mode == "median":
                mu = float(np.median(base_normalized))
            elif mode == "zero":
                mu = 0.0
            elif mode == "one":
                mu = 1.0
            elif mode == "holdout_mean" and self._holdout_mu_norm is not None:
                mu = float(self._holdout_mu_norm)
            elif mode == "profile_mean" and self._holdout_mu_norm is not None:
                mu = float(self._holdout_mu_norm)
            elif mode.startswith("ema") and self._ema_mu_norm is not None:
                mu = float(self._ema_mu_norm)
            else:
                mu = float(np.mean(base_normalized))
            base_normalized = alpha * base_normalized + (1.0 - alpha) * mu
            base_normalized = clip_to_range(base_normalized, 0.0, 1.0)

        # Add noise and clip
        debiased_normalized = base_normalized + noise
        debiased_normalized = clip_to_range(debiased_normalized, 0.0, 1.0)
        
        # Denormalize back to original scale
        debiased_judgments = denormalize_judgments(debiased_normalized, score_min, score_max)
        
        return debiased_judgments
    
    def fit_transform(self, 
                     judgments: Union[np.ndarray, pd.DataFrame, List],
                     **kwargs) -> np.ndarray:
        """
        Fit debiasing mechanism and transform judgments in one step.
        
        Parameters:
        -----------
        judgments : array-like or DataFrame
            Judgment data to fit and transform
        **kwargs : dict
            Additional arguments passed to fit()
            
        Returns:
        --------
        np.ndarray : Debiased judgment scores
        """
        # For fit_transform, extract scores if DataFrame is provided
        if isinstance(judgments, pd.DataFrame):
            self.fit(judgments, **kwargs)
            
            # Extract target scores for transformation
            score_cols = [col for col in judgments.columns if 'score' in col.lower()]
            if score_cols:
                target_scores = judgments[score_cols[0]].dropna().values
            else:
                raise ValueError("No score column found in DataFrame")
        else:
            self.fit(judgments, **kwargs)
            target_scores = judgments
        
        return self.transform(target_scores)
    
    def get_bias_bounds(self, n_samples: int) -> Dict[str, float]:
        """
        Get bias protection bounds for a given sample size.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples in dataset
            
        Returns:
        --------
        Dict[str, float] : Bias protection information
        """
        if not self._fitted:
            raise ValueError("Must call fit() before getting bias bounds")
        
        # Calculate sigma for this sample size using the A-BB formulation
        dimensionality = self.dimensionality if self.dimensionality is not None else n_samples
        sigma = calculate_abb_noise_parameter(
            rms_sensitivity=(self._bias_sensitivity / self._original_range) * float(self._effective_alpha or 1.0),
            tau=self.tau,
            delta=self.delta,
            dimensionality=dimensionality,
        )
        
        # Convert back to original scale
        sigma_original_scale = sigma * self._original_range
        
        return {
            "tau": self.tau,
            "delta": self.delta,
            "max_bias_ratio": np.exp(self.tau),
            "bias_sensitivity": self._bias_sensitivity,
            "noise_std": sigma_original_scale,
            "n_samples": n_samples,
            "use_average_case": self.use_average_case,
            "protection_guarantee": f"Bias patterns stronger than {self.tau:.2f} occur with probability < {self.delta:.3f}"
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostic information.
        
        Returns:
        --------
        Dict[str, Any] : Diagnostic information
        """
        diagnostics = {
            "fitted": self._fitted,
            "tau": self.tau,
            "delta": self.delta,
            "use_average_case": self.use_average_case,
            "random_seed": self.random_seed,
            "dimensionality": self.dimensionality,
            "is_abb_mechanism": isinstance(self.sensitivity_estimator, (ABBSensitivity, CombinedABBSensitivity, FixedSensitivityEstimator)),
        }
        
        if self._fitted:
            diagnostics.update({
                "bias_sensitivity": self._bias_sensitivity,
                "score_range": (self._score_min, self._score_max),
                "original_range": self._original_range,
                "sensitivity_estimator": self.sensitivity_estimator.get_diagnostics()
            })
            
            # Add A-BB specific diagnostics (normalized check to match mechanism math)
            if isinstance(self.sensitivity_estimator, (ABBSensitivity, CombinedABBSensitivity, FixedSensitivityEstimator)):
                precheck = build_abb_precheck(
                    tau=self.tau,
                    delta=self.delta,
                    sensitivity=float(self._bias_sensitivity),
                    score_range=float(self._original_range) if self._original_range is not None else None,
                    alpha=float(self._effective_alpha or 1.0),
                )
                diagnostics["abb_constraint_validation"] = precheck
                diagnostics["shrinkage"] = {
                    "enabled": bool((self._effective_alpha or 1.0) < 1.0),
                    "alpha": float(self._effective_alpha or 1.0),
                    "center": self.shrink_center,
                    "target_tau": float(self.target_tau) if self.target_tau is not None else None,
                    "suggested_projection_radius_normalized": (
                        float(self.target_tau / self._original_range) / 2.0 if self.target_tau is not None and self._original_range else None
                    ),
                }
        
        return diagnostics
    
    def validate_effectiveness(self, 
                             original_judgments: np.ndarray,
                             debiased_judgments: np.ndarray) -> Dict[str, float]:
        """
        Validate the effectiveness of debiasing.
        
        Parameters:
        -----------
        original_judgments : np.ndarray
            Original judgment scores
        debiased_judgments : np.ndarray
            Debiased judgment scores
            
        Returns:
        --------
        Dict[str, float] : Validation metrics
        """
        original_judgments = validate_input_array(original_judgments, "original_judgments")
        debiased_judgments = validate_input_array(debiased_judgments, "debiased_judgments")
        
        if len(original_judgments) != len(debiased_judgments):
            raise ValueError("Original and debiased judgments must have same length")
        
        # Calculate validation metrics
        correlation = float(np.corrcoef(original_judgments, debiased_judgments)[0, 1])
        mean_diff = float(np.mean(np.abs(original_judgments - debiased_judgments)))
        variance_ratio = float(np.var(debiased_judgments) / np.var(original_judgments))
        
        return {
            "correlation": correlation,
            "mean_absolute_difference": mean_diff,
            "variance_ratio": variance_ratio,
            "signal_preservation": correlation,  # Higher is better
            "noise_level": mean_diff / np.std(original_judgments),  # Relative to signal
        }

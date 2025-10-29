"""
Bias-bounded calibration sets for robust conformal prediction.

This module implements methods for creating calibration sets that are robust to
systematic bias using BBE noise injection. It ensures that conformal prediction
intervals remain valid even in the presence of judge manipulation or systematic
evaluation bias.
"""

import torch
import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List, Tuple, Callable
import warnings
from dataclasses import dataclass

from ..core.utils import calculate_abb_noise_parameter


@dataclass
class CalibrationSetConfig:
    """Configuration for bias-bounded calibration set creation."""
    noise_method: str = "gaussian"  # "gaussian", "laplace", "adaptive"
    tau: float = 1.0  # Bias protection parameter
    delta: float = 0.1  # Failure probability
    sensitivity_estimate: float = 1.0  # Bias sensitivity estimate
    use_average_case: bool = True  # Use average-case bounds
    dimensionality: Optional[int] = None  # For A-BB mechanism
    min_calibration_size: int = 50  # Minimum calibration set size
    max_noise_ratio: float = 0.5  # Maximum noise-to-signal ratio


class BiaseBoundedCalibrationCreator:
    """
    Creates robust calibration sets for conformal prediction using BBE noise injection.
    
    This class implements several strategies for creating calibration data that remains
    valid under systematic bias, ensuring that conformal prediction coverage guarantees
    hold even when judges are manipulated or exhibit systematic patterns.
    """
    
    def __init__(self,
                 config: Optional[CalibrationSetConfig] = None,
                 random_seed: Optional[int] = None,
                 device: str = "auto"):
        """
        Initialize bias-bounded calibration creator.
        
        Parameters:
        -----------
        config : CalibrationSetConfig, optional
            Configuration for calibration set creation
        random_seed : int, optional
            Random seed for reproducibility
        device : str
            PyTorch device for computations
        """
        self.config = config or CalibrationSetConfig()
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
    
    def create_robust_calibration_set(self,
                                    original_data: Union[pd.DataFrame, np.ndarray, Dict],
                                    bias_sensitivity: Optional[float] = None,
                                    method: str = "noise_injection") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Create a bias-bounded calibration set.
        
        Parameters:
        -----------
        original_data : DataFrame, array, or Dict
            Original evaluation data
        bias_sensitivity : float, optional
            Estimated bias sensitivity. If None, uses config default.
        method : str
            Method for creating robust calibration set:
            - "noise_injection": Add calibrated noise to scores
            - "adversarial_sampling": Sample adversarial examples
            - "debiased_bootstrap": Bootstrap with bias correction
            - "consensus_filtering": Filter to high-consensus examples
            
        Returns:
        --------
        Tuple : (calibration_features, calibration_targets, metadata)
        """
        if method == "noise_injection":
            return self._create_noise_injected_calibration(original_data, bias_sensitivity)
        elif method == "adversarial_sampling":
            return self._create_adversarial_calibration(original_data, bias_sensitivity)
        elif method == "debiased_bootstrap":
            return self._create_debiased_bootstrap_calibration(original_data, bias_sensitivity)
        elif method == "consensus_filtering":
            return self._create_consensus_filtered_calibration(original_data, bias_sensitivity)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def _create_noise_injected_calibration(self,
                                         original_data: Union[pd.DataFrame, np.ndarray, Dict],
                                         bias_sensitivity: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Create calibration set by injecting BBE-calibrated noise."""
        # Extract scores from data
        scores = self._extract_scores(original_data)
        features = self._extract_features(original_data)
        
        # Use provided sensitivity or config default
        sensitivity = bias_sensitivity or self.config.sensitivity_estimate
        
        # Calculate noise parameter using the A-BB mechanism (fallback dimensionality = number of scores)
        dimensionality = self.config.dimensionality or len(scores)
        try:
            noise_std = calculate_abb_noise_parameter(
                rms_sensitivity=sensitivity,
                tau=self.config.tau,
                delta=self.config.delta,
                dimensionality=dimensionality,
            )
        except ValueError as exc:
            warnings.warn(f"Failed to compute A-BB noise parameter: {exc}. Using zero noise instead.")
            noise_std = 0.0
        
        # Apply noise-to-signal ratio limit
        signal_std = np.std(scores)
        max_noise_std = self.config.max_noise_ratio * signal_std
        if noise_std > max_noise_std:
            warnings.warn(f"Noise std {noise_std:.4f} exceeds {self.config.max_noise_ratio*100}% of signal std. "
                         f"Capping at {max_noise_std:.4f}")
            noise_std = max_noise_std
        
        # Generate noise
        if self.config.noise_method == "gaussian":
            noise = self.rng.normal(0, noise_std, len(scores))
        elif self.config.noise_method == "laplace":
            # Laplace noise with equivalent privacy
            laplace_scale = noise_std / np.sqrt(2)
            noise = self.rng.laplace(0, laplace_scale, len(scores))
        elif self.config.noise_method == "adaptive":
            # Adaptive noise based on local density
            noise = self._generate_adaptive_noise(scores, noise_std)
        else:
            raise ValueError(f"Unknown noise method: {self.config.noise_method}")
        
        # Apply noise to scores
        noisy_scores = scores + noise
        
        # Ensure minimum calibration size
        if len(noisy_scores) < self.config.min_calibration_size:
            # Bootstrap to minimum size
            bootstrap_indices = self.rng.choice(len(noisy_scores), 
                                              size=self.config.min_calibration_size, 
                                              replace=True)
            noisy_scores = noisy_scores[bootstrap_indices]
            if features is not None:
                features = features[bootstrap_indices]
        
        # Prepare output
        calibration_targets = noisy_scores
        calibration_features = features if features is not None else np.arange(len(noisy_scores)).reshape(-1, 1)
        
        metadata = {
            "method": "noise_injection",
            "noise_method": self.config.noise_method,
            "noise_std": noise_std,
            "original_std": signal_std,
            "noise_to_signal_ratio": noise_std / signal_std,
            "tau": self.config.tau,
            "delta": self.config.delta,
            "sensitivity": sensitivity,
            "calibration_size": len(calibration_targets),
            "bias_bounded": True
        }
        
        return calibration_features, calibration_targets, metadata
    
    def _create_adversarial_calibration(self,
                                      original_data: Union[pd.DataFrame, np.ndarray, Dict],
                                      bias_sensitivity: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Create calibration set using adversarial sampling."""
        scores = self._extract_scores(original_data)
        features = self._extract_features(original_data)
        sensitivity = bias_sensitivity or self.config.sensitivity_estimate
        
        # Create adversarial perturbations that simulate worst-case bias
        adversarial_scores = []
        adversarial_features = []
        
        n_adversarial = max(self.config.min_calibration_size // 2, len(scores) // 4)
        
        for _ in range(n_adversarial):
            # Sample base score and feature
            base_idx = self.rng.randint(len(scores))
            base_score = scores[base_idx]
            base_feature = features[base_idx] if features is not None else np.array([base_idx])
            
            # Create adversarial perturbation within sensitivity bounds
            adversarial_perturbation = self.rng.uniform(-sensitivity, sensitivity)
            adversarial_score = base_score + adversarial_perturbation
            
            adversarial_scores.append(adversarial_score)
            adversarial_features.append(base_feature)
        
        # Combine with original data
        all_scores = np.concatenate([scores, adversarial_scores])
        if features is not None:
            all_features = np.vstack([features, np.array(adversarial_features)])
        else:
            all_features = np.arange(len(all_scores)).reshape(-1, 1)
        
        metadata = {
            "method": "adversarial_sampling",
            "n_adversarial": n_adversarial,
            "sensitivity": sensitivity,
            "calibration_size": len(all_scores),
            "bias_bounded": True
        }
        
        return all_features, all_scores, metadata
    
    def _create_debiased_bootstrap_calibration(self,
                                             original_data: Union[pd.DataFrame, np.ndarray, Dict],
                                             bias_sensitivity: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Create calibration set using bias-corrected bootstrap."""
        scores = self._extract_scores(original_data)
        features = self._extract_features(original_data)
        sensitivity = bias_sensitivity or self.config.sensitivity_estimate
        
        # Estimate and correct for systematic bias
        # Simple approach: center scores and add controlled variation
        centered_scores = scores - np.mean(scores)
        
        # Bootstrap with bias correction
        n_bootstrap = max(self.config.min_calibration_size, len(scores))
        bootstrap_scores = []
        bootstrap_features = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            sample_indices = self.rng.choice(len(scores), size=len(scores), replace=True)
            sample_scores = centered_scores[sample_indices]
            
            # Add bias correction noise
            bias_correction = self.rng.normal(0, sensitivity * 0.5, len(sample_scores))
            corrected_scores = sample_scores + bias_correction
            
            # Take mean to create single calibration point
            bootstrap_scores.append(np.mean(corrected_scores))
            
            if features is not None:
                bootstrap_features.append(np.mean(features[sample_indices], axis=0))
            else:
                bootstrap_features.append([len(bootstrap_scores) - 1])
        
        calibration_targets = np.array(bootstrap_scores)
        calibration_features = np.array(bootstrap_features) if features is not None else np.arange(len(bootstrap_scores)).reshape(-1, 1)
        
        metadata = {
            "method": "debiased_bootstrap",
            "n_bootstrap": n_bootstrap,
            "sensitivity": sensitivity,
            "calibration_size": len(calibration_targets),
            "bias_bounded": True
        }
        
        return calibration_features, calibration_targets, metadata
    
    def _create_consensus_filtered_calibration(self,
                                             original_data: Union[pd.DataFrame, np.ndarray, Dict],
                                             bias_sensitivity: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Create calibration set by filtering to high-consensus examples."""
        # This method assumes multi-judge data
        if isinstance(original_data, pd.DataFrame):
            score_columns = [col for col in original_data.columns if 'score' in col.lower()]
            if len(score_columns) < 2:
                warnings.warn("Consensus filtering requires multiple judges. Falling back to noise injection.")
                return self._create_noise_injected_calibration(original_data, bias_sensitivity)
            
            judge_scores = original_data[score_columns].values
        elif isinstance(original_data, dict) and len(original_data) > 1:
            judge_names = sorted(original_data.keys())
            judge_scores = np.column_stack([original_data[judge] for judge in judge_names])
        else:
            warnings.warn("Consensus filtering requires multiple judges. Falling back to noise injection.")
            return self._create_noise_injected_calibration(original_data, bias_sensitivity)
        
        # Calculate consensus metrics
        judge_means = np.mean(judge_scores, axis=1)
        judge_stds = np.std(judge_scores, axis=1)
        
        # Filter to low-disagreement examples
        disagreement_threshold = np.percentile(judge_stds, 70)  # Keep 30% most consensual
        consensus_mask = judge_stds <= disagreement_threshold
        
        if np.sum(consensus_mask) < self.config.min_calibration_size:
            # Relax threshold if too few consensus examples
            disagreement_threshold = np.percentile(judge_stds, 90)
            consensus_mask = judge_stds <= disagreement_threshold
        
        consensus_scores = judge_means[consensus_mask]
        consensus_stds = judge_stds[consensus_mask]
        
        # Features include both mean score and disagreement level
        consensus_features = np.column_stack([consensus_scores, consensus_stds])
        
        # Bootstrap if needed for minimum size
        if len(consensus_scores) < self.config.min_calibration_size:
            bootstrap_indices = self.rng.choice(len(consensus_scores),
                                              size=self.config.min_calibration_size,
                                              replace=True)
            consensus_scores = consensus_scores[bootstrap_indices]
            consensus_features = consensus_features[bootstrap_indices]
        
        metadata = {
            "method": "consensus_filtering",
            "disagreement_threshold": disagreement_threshold,
            "consensus_fraction": np.mean(consensus_mask),
            "n_judges": judge_scores.shape[1],
            "calibration_size": len(consensus_scores),
            "bias_bounded": True
        }
        
        return consensus_features, consensus_scores, metadata
    
    def _generate_adaptive_noise(self, scores: np.ndarray, base_noise_std: float) -> np.ndarray:
        """Generate adaptive noise based on local score density."""
        # Estimate local density using kernel density estimation
        from scipy.stats import gaussian_kde
        
        try:
            kde = gaussian_kde(scores)
            densities = kde(scores)
            
            # Inverse relationship: lower density -> higher noise
            # Normalize densities
            normalized_densities = (densities - np.min(densities)) / (np.max(densities) - np.min(densities) + 1e-8)
            
            # Adaptive noise: higher noise in low-density regions
            noise_multipliers = 1.0 + (1.0 - normalized_densities)  # Range [1, 2]
            adaptive_stds = base_noise_std * noise_multipliers
            
            # Generate noise with adaptive standard deviations
            noise = np.array([self.rng.normal(0, std) for std in adaptive_stds])
            
        except Exception as e:
            warnings.warn(f"Adaptive noise generation failed: {e}. Using uniform noise.")
            noise = self.rng.normal(0, base_noise_std, len(scores))
        
        return noise
    
    def _extract_scores(self, data: Union[pd.DataFrame, np.ndarray, Dict]) -> np.ndarray:
        """Extract scores from various data formats."""
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return data
            elif data.ndim == 2:
                # If multiple columns, take mean as consensus score
                return np.mean(data, axis=1)
            else:
                raise ValueError(f"Unsupported array shape: {data.shape}")
        
        elif isinstance(data, pd.DataFrame):
            score_columns = [col for col in data.columns if 'score' in col.lower()]
            if not score_columns:
                raise ValueError("No score columns found in DataFrame")
            
            if len(score_columns) == 1:
                return data[score_columns[0]].values
            else:
                # Multiple score columns - take mean
                return data[score_columns].mean(axis=1).values
        
        elif isinstance(data, dict):
            if len(data) == 1:
                return np.asarray(list(data.values())[0])
            else:
                # Multiple judges - take mean
                judge_scores = np.column_stack(list(data.values()))
                return np.mean(judge_scores, axis=1)
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _extract_features(self, data: Union[pd.DataFrame, np.ndarray, Dict]) -> Optional[np.ndarray]:
        """Extract features from data if available."""
        if isinstance(data, pd.DataFrame):
            # Use non-score columns as features
            feature_columns = [col for col in data.columns if 'score' not in col.lower()]
            if feature_columns:
                return data[feature_columns].values
        
        # If no features available, return None (will use indices)
        return None
    
    def validate_exchangeability(self,
                               calibration_data: np.ndarray,
                               test_data: Optional[np.ndarray] = None,
                               method: str = "ks_test") -> Dict[str, Any]:
        """
        Validate that calibration data maintains exchangeability after bias-bounded transformation.
        
        Parameters:
        -----------
        calibration_data : np.ndarray
            Bias-bounded calibration data
        test_data : np.ndarray, optional
            Test data to compare against
        method : str
            Statistical test method: "ks_test", "anderson_darling", "permutation"
            
        Returns:
        --------
        Dict[str, Any] : Validation results
        """
        if method == "ks_test":
            return self._ks_test_exchangeability(calibration_data, test_data)
        elif method == "anderson_darling":
            return self._anderson_darling_exchangeability(calibration_data, test_data)
        elif method == "permutation":
            return self._permutation_test_exchangeability(calibration_data, test_data)
        else:
            raise ValueError(f"Unknown validation method: {method}")
    
    def _ks_test_exchangeability(self, cal_data: np.ndarray, test_data: Optional[np.ndarray]) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test for exchangeability."""
        from scipy.stats import ks_2samp
        
        if test_data is None:
            # Test internal exchangeability by splitting calibration data
            n_split = len(cal_data) // 2
            indices = self.rng.permutation(len(cal_data))
            split1 = cal_data[indices[:n_split]]
            split2 = cal_data[indices[n_split:2*n_split]]
        else:
            split1 = cal_data
            split2 = test_data
        
        # Perform KS test
        statistic, p_value = ks_2samp(split1, split2)
        
        # Exchangeability is supported if we fail to reject null hypothesis (p > 0.05)
        exchangeable = p_value > 0.05
        
        return {
            "method": "ks_test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "exchangeable": exchangeable,
            "interpretation": "Distributions are similar" if exchangeable else "Distributions differ significantly"
        }
    
    def _anderson_darling_exchangeability(self, cal_data: np.ndarray, test_data: Optional[np.ndarray]) -> Dict[str, Any]:
        """Anderson-Darling test for exchangeability."""
        from scipy.stats import anderson_ksamp
        
        if test_data is None:
            # Split calibration data
            n_split = len(cal_data) // 2
            indices = self.rng.permutation(len(cal_data))
            samples = [cal_data[indices[:n_split]], cal_data[indices[n_split:2*n_split]]]
        else:
            samples = [cal_data, test_data]
        
        # Perform Anderson-Darling test
        result = anderson_ksamp(samples)
        
        # Exchangeability supported if p > 0.05
        exchangeable = result.pvalue > 0.05
        
        return {
            "method": "anderson_darling",
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "exchangeable": exchangeable,
            "critical_values": result.critical_values.tolist() if hasattr(result, 'critical_values') else None
        }
    
    def _permutation_test_exchangeability(self, cal_data: np.ndarray, test_data: Optional[np.ndarray]) -> Dict[str, Any]:
        """Permutation test for exchangeability."""
        if test_data is None:
            # Use all calibration data for permutation test
            test_statistic = np.var(cal_data)  # Use variance as test statistic
            
            # Generate null distribution via permutation
            n_permutations = 1000
            null_statistics = []
            
            for _ in range(n_permutations):
                permuted_data = self.rng.permutation(cal_data)
                null_statistics.append(np.var(permuted_data))
            
            # Compute p-value
            null_statistics = np.array(null_statistics)
            p_value = np.mean(np.abs(null_statistics - np.mean(null_statistics)) >= 
                            np.abs(test_statistic - np.mean(null_statistics)))
            
        else:
            # Use difference in means as test statistic
            test_statistic = np.abs(np.mean(cal_data) - np.mean(test_data))
            
            # Combined data for permutation
            combined_data = np.concatenate([cal_data, test_data])
            n_cal = len(cal_data)
            
            null_statistics = []
            n_permutations = 1000
            
            for _ in range(n_permutations):
                permuted_combined = self.rng.permutation(combined_data)
                perm_cal = permuted_combined[:n_cal]
                perm_test = permuted_combined[n_cal:]
                null_statistics.append(np.abs(np.mean(perm_cal) - np.mean(perm_test)))
            
            # Compute p-value
            null_statistics = np.array(null_statistics)
            p_value = np.mean(null_statistics >= test_statistic)
        
        exchangeable = p_value > 0.05
        
        return {
            "method": "permutation_test",
            "test_statistic": float(test_statistic),
            "p_value": float(p_value),
            "n_permutations": n_permutations,
            "exchangeable": exchangeable
        }

"""
Conformal Prediction-based sensitivity estimation for bias-bounded evaluation.

This module implements sensitivity estimators that use conformal prediction (CP)
to quantify uncertainty and improve bias sensitivity measurements. It integrates
TorchCP for state-of-the-art conformal prediction capabilities.
"""

import torch
import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List, Callable, Tuple
import warnings

try:
    import torchcp
    from torchcp.classification import SplitCP, RAPS, APS
    from torchcp.regression import SplitCP as RegressionSplitCP
    TORCHCP_AVAILABLE = True
except ImportError:
    TORCHCP_AVAILABLE = False
    warnings.warn("TorchCP not available. Install with: pip install torchcp")

from .base import SensitivityEstimator


class ConformalSensitivityEstimator(SensitivityEstimator):
    """
    Sensitivity estimator that uses conformal prediction to quantify uncertainty
    and improve bias sensitivity measurements.
    
    This estimator addresses a key limitation in traditional sensitivity estimation:
    it incorporates prediction uncertainty to provide more robust bias estimates.
    Instead of treating all predictions equally, it weights sensitivity calculations
    by prediction uncertainty from conformal prediction intervals.
    """
    
    def __init__(self,
                 base_model: Optional[torch.nn.Module] = None,
                 conformal_method: str = "split_cp",
                 alpha: float = 0.1,
                 calibration_ratio: float = 0.2,
                 aggregation_strategy: str = "uncertainty_weighted",
                 device: str = "auto",
                 random_seed: Optional[int] = None,
                 **kwargs):
        """
        Initialize conformal sensitivity estimator.
        
        Parameters:
        -----------
        base_model : torch.nn.Module, optional
            Base PyTorch model for predictions. If None, will expect pre-computed scores.
        conformal_method : str
            Conformal prediction method. Options: "split_cp", "raps", "aps"
        alpha : float
            Significance level for conformal prediction intervals (default: 0.1 for 90% coverage)
        calibration_ratio : float
            Proportion of data to use for conformal calibration
        aggregation_strategy : str
            How to aggregate multiple judge predictions. Options: "uncertainty_weighted", "ensemble_cp", "median"
        device : str
            Device for PyTorch operations ("cpu", "cuda", or "auto")
        random_seed : int, optional
            Random seed for reproducibility
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        
        if not TORCHCP_AVAILABLE:
            raise ImportError("TorchCP is required but not available. Install with: pip install torchcp")
        
        self.base_model = base_model
        self.conformal_method = conformal_method
        self.alpha = alpha
        self.calibration_ratio = calibration_ratio
        self.aggregation_strategy = aggregation_strategy
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
        
        # Initialize conformal predictor
        self.conformal_predictor = None
        self._fitted = False
        
        # State for sensitivity calculation
        self._prediction_intervals = None
        self._uncertainty_scores = None
        self._base_predictions = None
        
    def _create_conformal_predictor(self, num_classes: Optional[int] = None):
        """Create conformal predictor based on specified method."""
        if self.conformal_method == "split_cp":
            if num_classes is not None:
                # Classification
                self.conformal_predictor = SplitCP(score_function="softmax")
            else:
                # Regression
                self.conformal_predictor = RegressionSplitCP()
        elif self.conformal_method == "raps":
            if num_classes is None:
                raise ValueError("RAPS method requires classification (num_classes must be specified)")
            self.conformal_predictor = RAPS()
        elif self.conformal_method == "aps":
            if num_classes is None:
                raise ValueError("APS method requires classification (num_classes must be specified)")
            self.conformal_predictor = APS()
        else:
            raise ValueError(f"Unknown conformal method: {self.conformal_method}")
    
    def fit(self, judgments: Union[np.ndarray, pd.DataFrame, Dict], **kwargs) -> 'ConformalSensitivityEstimator':
        """
        Fit the conformal sensitivity estimator.
        
        This method calibrates conformal prediction intervals and computes
        uncertainty-weighted sensitivity estimates.
        
        Parameters:
        -----------
        judgments : array-like, DataFrame, or Dict
            Judge evaluation data. Can be:
            - DataFrame: With columns for scores and potentially judge IDs
            - Dict: Multiple judges' scores
            - numpy array: Single judge scores
        **kwargs : dict
            Additional fitting parameters:
            - judge_functions: List of judge functions for multi-judge scenarios
            - calibration_data: Explicit calibration data
            - num_classes: Number of classes for classification tasks
            
        Returns:
        --------
        ConformalSensitivityEstimator : Self for method chaining
        """
        # Prepare data for conformal prediction
        if isinstance(judgments, pd.DataFrame):
            scores, multiple_judges = self._prepare_dataframe_data(judgments)
        elif isinstance(judgments, dict):
            scores, multiple_judges = self._prepare_dict_data(judgments)
        else:
            scores = np.asarray(judgments)
            multiple_judges = False
        
        # Determine if this is classification or regression
        num_classes = self._infer_num_classes(scores)
        
        # Create conformal predictor
        self._create_conformal_predictor(num_classes)
        
        if multiple_judges:
            self._fit_multi_judge(scores, **kwargs)
        else:
            self._fit_single_judge(scores, **kwargs)
        
        self._fitted = True
        return self
    
    def _prepare_dataframe_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, bool]:
        """Prepare DataFrame for conformal prediction."""
        # Look for score columns
        score_columns = [col for col in df.columns if 'score' in col.lower()]
        
        if not score_columns:
            raise ValueError("No score columns found in DataFrame")
        
        if len(score_columns) == 1:
            # Single judge
            return df[score_columns[0]].values, False
        else:
            # Multiple judges/scores
            scores = df[score_columns].values
            return scores, True
    
    def _prepare_dict_data(self, data_dict: Dict) -> Tuple[np.ndarray, bool]:
        """Prepare dictionary data for conformal prediction."""
        if len(data_dict) == 1:
            # Single judge
            scores = list(data_dict.values())[0]
            return np.asarray(scores), False
        else:
            # Multiple judges
            judge_names = sorted(data_dict.keys())
            scores = np.column_stack([data_dict[judge] for judge in judge_names])
            return scores, True
    
    def _infer_num_classes(self, scores: np.ndarray) -> Optional[int]:
        """Infer number of classes from score data."""
        if scores.ndim == 1:
            unique_scores = np.unique(scores)
            # If we have a small number of integer values, treat as classification
            if len(unique_scores) <= 10 and all(isinstance(x, (int, np.integer)) for x in unique_scores):
                return len(unique_scores)
        elif scores.ndim == 2:
            # Multi-dimensional scores - check if they look like class probabilities
            if np.allclose(scores.sum(axis=1), 1.0) and np.all(scores >= 0):
                return scores.shape[1]
        
        return None  # Treat as regression
    
    def _fit_single_judge(self, scores: np.ndarray, **kwargs):
        """Fit conformal predictor for single judge scenario."""
        # Split data for calibration
        n_total = len(scores)
        n_cal = int(n_total * self.calibration_ratio)
        
        # Random split
        indices = np.random.permutation(n_total)
        cal_indices = indices[:n_cal]
        train_indices = indices[n_cal:]
        
        # For single judge, we treat the task as predicting score reliability
        # We use the scores themselves as both features and targets for uncertainty estimation
        cal_scores = scores[cal_indices]
        train_scores = scores[train_indices]
        
        # Create dummy features (we'll improve this in multi-judge scenarios)
        cal_features = np.arange(len(cal_scores)).reshape(-1, 1)
        train_features = np.arange(len(train_scores)).reshape(-1, 1)
        
        # Convert to tensors
        cal_features_tensor = torch.FloatTensor(cal_features).to(self.device)
        cal_scores_tensor = torch.FloatTensor(cal_scores).to(self.device)
        train_features_tensor = torch.FloatTensor(train_features).to(self.device)
        train_scores_tensor = torch.FloatTensor(train_scores).to(self.device)
        
        # Calibrate conformal predictor
        if self.base_model is None:
            # Create simple model for uncertainty estimation
            self.base_model = self._create_simple_model(cal_features.shape[1])
            self.base_model.to(self.device)
        
        # Get model predictions for calibration
        with torch.no_grad():
            cal_predictions = self.base_model(cal_features_tensor)
        
        # Calibrate conformal predictor
        self.conformal_predictor.calibrate(cal_predictions, cal_scores_tensor, alpha=self.alpha)
        
        # Store base predictions and compute uncertainty scores
        with torch.no_grad():
            all_predictions = self.base_model(torch.FloatTensor(np.arange(len(scores)).reshape(-1, 1)).to(self.device))
        
        # Get prediction intervals
        prediction_intervals = self.conformal_predictor.predict(all_predictions)
        
        # Compute uncertainty scores (interval width)
        if isinstance(prediction_intervals, tuple):
            # Regression intervals: (lower, upper)
            interval_widths = prediction_intervals[1] - prediction_intervals[0]
        else:
            # Classification sets: size of prediction set
            interval_widths = torch.tensor([len(pred_set) for pred_set in prediction_intervals])
        
        self._prediction_intervals = prediction_intervals
        self._uncertainty_scores = interval_widths.cpu().numpy()
        self._base_predictions = all_predictions.cpu().numpy()
    
    def _fit_multi_judge(self, scores: np.ndarray, **kwargs):
        """Fit conformal predictor for multi-judge scenario."""
        # Multi-judge conformal prediction using ensemble aggregation
        n_judges = scores.shape[1]
        n_total = scores.shape[0]
        n_cal = int(n_total * self.calibration_ratio)
        
        # Split data
        indices = np.random.permutation(n_total)
        cal_indices = indices[:n_cal]
        train_indices = indices[n_cal:]
        
        cal_scores = scores[cal_indices]
        train_scores = scores[train_indices]
        
        # For multi-judge, use judge disagreement as features
        judge_features = self._compute_judge_features(scores)
        
        cal_features = judge_features[cal_indices]
        train_features = judge_features[train_indices]
        
        # Convert to tensors
        cal_features_tensor = torch.FloatTensor(cal_features).to(self.device)
        cal_target = torch.FloatTensor(np.mean(cal_scores, axis=1)).to(self.device)  # Average as target
        
        # Create model if needed
        if self.base_model is None:
            self.base_model = self._create_simple_model(judge_features.shape[1])
            self.base_model.to(self.device)
        
        # Get predictions and calibrate
        with torch.no_grad():
            cal_predictions = self.base_model(cal_features_tensor)
        
        self.conformal_predictor.calibrate(cal_predictions, cal_target, alpha=self.alpha)
        
        # Get all predictions and intervals
        all_features_tensor = torch.FloatTensor(judge_features).to(self.device)
        with torch.no_grad():
            all_predictions = self.base_model(all_features_tensor)
        
        prediction_intervals = self.conformal_predictor.predict(all_predictions)
        
        # Compute uncertainty scores
        if isinstance(prediction_intervals, tuple):
            interval_widths = prediction_intervals[1] - prediction_intervals[0]
        else:
            interval_widths = torch.tensor([len(pred_set) for pred_set in prediction_intervals])
        
        self._prediction_intervals = prediction_intervals
        self._uncertainty_scores = interval_widths.cpu().numpy()
        self._base_predictions = all_predictions.cpu().numpy()
        
        # Store judge disagreement as additional uncertainty measure
        judge_std = np.std(scores, axis=1)
        self._judge_disagreement = judge_std
    
    def _compute_judge_features(self, scores: np.ndarray) -> np.ndarray:
        """Compute features from multi-judge scores for uncertainty estimation."""
        features = []
        
        # Basic statistics
        features.append(np.mean(scores, axis=1))  # Average score
        features.append(np.std(scores, axis=1))   # Judge disagreement
        features.append(np.min(scores, axis=1))   # Min score
        features.append(np.max(scores, axis=1))   # Max score
        
        # Pairwise correlations (if more than 2 judges)
        if scores.shape[1] > 2:
            for i in range(scores.shape[1]):
                for j in range(i+1, scores.shape[1]):
                    # Pearson correlation between judges i and j per sample
                    corr_feature = np.corrcoef(scores[:, i], scores[:, j])[0, 1]
                    features.append(np.full(len(scores), corr_feature))
        
        return np.column_stack(features)
    
    def _create_simple_model(self, input_dim: int) -> torch.nn.Module:
        """Create simple neural network for uncertainty estimation."""
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    
    def estimate(self, score_range: Optional[float] = None) -> float:
        """
        Estimate bias sensitivity using conformal prediction uncertainty.
        
        Parameters:
        -----------
        score_range : float, optional
            Range of judgment scores for validation
            
        Returns:
        --------
        float : Uncertainty-weighted sensitivity estimate
        """
        self._validate_fitted()
        
        if self._uncertainty_scores is None:
            raise ValueError("No uncertainty scores computed. Check fit() call.")
        
        # Compute sensitivity as weighted average of uncertainty scores
        if self.aggregation_strategy == "uncertainty_weighted":
            # Weight by uncertainty - more uncertain predictions contribute more to sensitivity
            weights = self._uncertainty_scores / np.sum(self._uncertainty_scores)
            sensitivity = np.sum(weights * self._uncertainty_scores)
        elif self.aggregation_strategy == "ensemble_cp":
            # Use ensemble-style aggregation
            sensitivity = np.mean(self._uncertainty_scores)
        elif self.aggregation_strategy == "median":
            # Robust median-based estimate
            sensitivity = np.median(self._uncertainty_scores)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
        
        # Add judge disagreement if available (multi-judge case)
        if hasattr(self, '_judge_disagreement'):
            disagreement_weight = 0.3  # Hyperparameter for combining CP uncertainty with judge disagreement
            judge_uncertainty = np.mean(self._judge_disagreement)
            sensitivity = (1 - disagreement_weight) * sensitivity + disagreement_weight * judge_uncertainty
        
        # Validate against score range if provided
        if score_range is not None and sensitivity > score_range:
            print(f"⚠️ Conformal sensitivity {sensitivity:.4f} exceeds score range {score_range:.4f}")
            sensitivity = min(sensitivity, score_range)
        
        return float(sensitivity)
    
    def get_prediction_intervals(self) -> Union[Tuple[np.ndarray, np.ndarray], List]:
        """
        Get conformal prediction intervals.
        
        Returns:
        --------
        Union[Tuple, List] : Prediction intervals or sets
        """
        self._validate_fitted()
        
        if isinstance(self._prediction_intervals, tuple):
            # Regression intervals
            return (self._prediction_intervals[0].cpu().numpy(), 
                   self._prediction_intervals[1].cpu().numpy())
        else:
            # Classification sets
            return self._prediction_intervals
    
    def get_uncertainty_scores(self) -> np.ndarray:
        """
        Get uncertainty scores (interval widths or set sizes).
        
        Returns:
        --------
        np.ndarray : Uncertainty scores for each prediction
        """
        self._validate_fitted()
        return self._uncertainty_scores.copy()
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about conformal sensitivity estimation."""
        diagnostics = super().get_diagnostics()
        
        if self._fitted:
            diagnostics.update({
                "conformal_method": self.conformal_method,
                "alpha": self.alpha,
                "coverage_level": 1 - self.alpha,
                "calibration_ratio": self.calibration_ratio,
                "aggregation_strategy": self.aggregation_strategy,
                "mean_uncertainty": float(np.mean(self._uncertainty_scores)) if self._uncertainty_scores is not None else None,
                "std_uncertainty": float(np.std(self._uncertainty_scores)) if self._uncertainty_scores is not None else None,
                "has_multi_judge": hasattr(self, '_judge_disagreement'),
                "device": str(self.device),
                "base_model_params": sum(p.numel() for p in self.base_model.parameters()) if self.base_model else 0
            })
            
            if hasattr(self, '_judge_disagreement'):
                diagnostics.update({
                    "mean_judge_disagreement": float(np.mean(self._judge_disagreement)),
                    "std_judge_disagreement": float(np.std(self._judge_disagreement))
                })
        
        return diagnostics
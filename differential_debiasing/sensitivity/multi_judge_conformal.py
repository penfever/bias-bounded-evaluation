"""
Multi-judge conformal prediction aggregation for LLM evaluation without ground truth.

This module implements conformal prediction methods specifically designed for
scenarios with multiple LLM judges but no single source of ground truth. It uses
weighted aggregation and ensemble techniques to create robust consensus predictions
with valid uncertainty quantification.
"""

import torch
import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List, Callable, Tuple
import warnings
from dataclasses import dataclass

try:
    import torchcp
    from torchcp.classification import RAPS
    try:
        from torchcp.classification import SplitCP  # torchcp<=1.1
    except ImportError:
        from torchcp.classification import SplitPredictor as SplitCP  # torchcp>=1.2
    try:
        from torchcp.regression import SplitCP as RegressionSplitCP  # torchcp<=1.1
    except ImportError:
        from torchcp.regression import SplitPredictor as RegressionSplitCP  # torchcp>=1.2
    TORCHCP_AVAILABLE = True
except ImportError:
    TORCHCP_AVAILABLE = False
    warnings.warn("TorchCP not available. Install with: pip install torchcp")


@dataclass
class JudgeWeight:
    """Container for judge weighting information."""
    judge_id: str
    reliability_weight: float
    disagreement_penalty: float
    historical_accuracy: Optional[float] = None
    
    def get_effective_weight(self) -> float:
        """Compute effective weight combining reliability and disagreement."""
        return self.reliability_weight * (1 - self.disagreement_penalty)


class MultiJudgeConformalPredictor:
    """
    Conformal prediction aggregator for multiple LLM judges without ground truth.
    
    This class implements several strategies for aggregating predictions from multiple
    judges using conformal prediction principles, when there is no single ground truth
    to calibrate against.
    """
    
    def __init__(self,
                 aggregation_method: str = "weighted_ensemble",
                 conformal_method: str = "split_cp",
                 alpha: float = 0.1,
                 judge_weight_strategy: str = "reliability_based",
                 consensus_threshold: float = 0.7,
                 disagreement_tolerance: float = 0.3,
                 calibration_ratio: float = 0.2,
                 device: str = "auto",
                 random_seed: Optional[int] = None,
                 **kwargs):
        """
        Initialize multi-judge conformal predictor.
        
        Parameters:
        -----------
        aggregation_method : str
            Method for aggregating judge predictions. Options:
            - "weighted_ensemble": Weighted average with reliability-based weights
            - "majority_vote": Majority voting with conformal intervals
            - "consensus_only": Only use predictions where judges agree
            - "hierarchical": Hierarchical aggregation with judge clustering
        conformal_method : str
            Underlying conformal prediction method
        alpha : float
            Significance level for conformal prediction intervals
        judge_weight_strategy : str
            Strategy for computing judge weights. Options:
            - "reliability_based": Use judge self-consistency
            - "disagreement_based": Weight by disagreement patterns
            - "adaptive": Adapt weights based on prediction difficulty
            - "equal": Equal weights for all judges
        consensus_threshold : float
            Minimum agreement threshold for consensus-based methods
        disagreement_tolerance : float
            Maximum allowed disagreement for including a prediction
        calibration_ratio : float
            Proportion of data to use for conformal calibration
        device : str
            PyTorch device
        random_seed : int, optional
            Random seed for reproducibility
        """
        if not TORCHCP_AVAILABLE:
            raise ImportError("TorchCP is required but not available. Install with: pip install torchcp")
        
        self.aggregation_method = aggregation_method
        self.conformal_method = conformal_method
        self.alpha = alpha
        self.judge_weight_strategy = judge_weight_strategy
        self.consensus_threshold = consensus_threshold
        self.disagreement_tolerance = disagreement_tolerance
        self.calibration_ratio = calibration_ratio
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
        
        # State
        self.judge_weights_ = {}
        self.conformal_predictors_ = {}
        self.fitted_ = False
        self.judge_names_ = []
        
    def fit(self, judge_predictions: Dict[str, np.ndarray], 
            judge_metadata: Optional[Dict[str, Dict]] = None,
            **kwargs) -> 'MultiJudgeConformalPredictor':
        """
        Fit multi-judge conformal predictor.
        
        Parameters:
        -----------
        judge_predictions : Dict[str, np.ndarray]
            Dictionary mapping judge names to their prediction arrays
        judge_metadata : Dict[str, Dict], optional
            Additional metadata for each judge (e.g., model type, temperature)
        **kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        MultiJudgeConformalPredictor : Self for method chaining
        """
        self.judge_names_ = list(judge_predictions.keys())
        
        # Validate input
        self._validate_judge_predictions(judge_predictions)
        
        # Compute judge weights
        self.judge_weights_ = self._compute_judge_weights(judge_predictions, judge_metadata)
        
        # Fit conformal predictors
        if self.aggregation_method == "weighted_ensemble":
            self._fit_weighted_ensemble(judge_predictions, **kwargs)
        elif self.aggregation_method == "majority_vote":
            self._fit_majority_vote(judge_predictions, **kwargs)
        elif self.aggregation_method == "consensus_only":
            self._fit_consensus_only(judge_predictions, **kwargs)
        elif self.aggregation_method == "hierarchical":
            self._fit_hierarchical(judge_predictions, **kwargs)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        self.fitted_ = True
        return self
    
    def _validate_judge_predictions(self, judge_predictions: Dict[str, np.ndarray]):
        """Validate judge prediction inputs."""
        if not judge_predictions:
            raise ValueError("No judge predictions provided")
        
        # Check that all judges have same number of predictions
        lengths = [len(preds) for preds in judge_predictions.values()]
        if len(set(lengths)) > 1:
            raise ValueError(f"Judge predictions have different lengths: {lengths}")
        
        # Check for missing values
        for judge_name, preds in judge_predictions.items():
            if np.any(np.isnan(preds)):
                warnings.warn(f"Judge {judge_name} has NaN predictions")
    
    def _compute_judge_weights(self, judge_predictions: Dict[str, np.ndarray],
                             judge_metadata: Optional[Dict[str, Dict]] = None) -> Dict[str, JudgeWeight]:
        """Compute weights for each judge based on the specified strategy."""
        judge_weights = {}
        
        if self.judge_weight_strategy == "equal":
            # Equal weights
            n_judges = len(judge_predictions)
            for judge_name in judge_predictions:
                judge_weights[judge_name] = JudgeWeight(
                    judge_id=judge_name,
                    reliability_weight=1.0 / n_judges,
                    disagreement_penalty=0.0
                )
        
        elif self.judge_weight_strategy == "reliability_based":
            judge_weights = self._compute_reliability_weights(judge_predictions)
        
        elif self.judge_weight_strategy == "disagreement_based":
            judge_weights = self._compute_disagreement_weights(judge_predictions)
        
        elif self.judge_weight_strategy == "adaptive":
            judge_weights = self._compute_adaptive_weights(judge_predictions)
        
        else:
            raise ValueError(f"Unknown judge weight strategy: {self.judge_weight_strategy}")
        
        return judge_weights
    
    def _compute_reliability_weights(self, judge_predictions: Dict[str, np.ndarray]) -> Dict[str, JudgeWeight]:
        """Compute weights based on judge self-consistency and agreement with others."""
        judge_weights = {}
        judge_names = list(judge_predictions.keys())
        n_judges = len(judge_names)
        
        # Convert to matrix for easier computation
        pred_matrix = np.column_stack([judge_predictions[name] for name in judge_names])
        
        for i, judge_name in enumerate(judge_names):
            judge_preds = pred_matrix[:, i]
            other_preds = np.delete(pred_matrix, i, axis=1)
            
            # Self-consistency: lower variance indicates higher reliability
            self_consistency = 1.0 / (1.0 + np.var(judge_preds))
            
            # Agreement with others: higher correlation indicates higher reliability
            if other_preds.shape[1] > 0:
                correlations = []
                for j in range(other_preds.shape[1]):
                    corr = np.corrcoef(judge_preds, other_preds[:, j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))  # Use absolute correlation
                
                agreement = np.mean(correlations) if correlations else 0.0
            else:
                agreement = 0.0
            
            # Combined reliability score
            reliability = 0.6 * self_consistency + 0.4 * agreement
            
            # Disagreement penalty based on deviation from group average
            group_avg = np.mean(pred_matrix, axis=1)
            disagreement = np.mean(np.abs(judge_preds - group_avg))
            disagreement_penalty = min(disagreement / np.std(group_avg), 0.5)  # Cap at 50%
            
            judge_weights[judge_name] = JudgeWeight(
                judge_id=judge_name,
                reliability_weight=reliability,
                disagreement_penalty=disagreement_penalty
            )
        
        # Normalize weights
        total_effective_weight = sum(w.get_effective_weight() for w in judge_weights.values())
        for weight in judge_weights.values():
            weight.reliability_weight = weight.get_effective_weight() / total_effective_weight
            weight.disagreement_penalty = 0.0  # Reset since we normalized
        
        return judge_weights
    
    def _compute_disagreement_weights(self, judge_predictions: Dict[str, np.ndarray]) -> Dict[str, JudgeWeight]:
        """Compute weights based on disagreement patterns."""
        judge_names = list(judge_predictions.keys())
        pred_matrix = np.column_stack([judge_predictions[name] for name in judge_names])
        
        judge_weights = {}
        
        for i, judge_name in enumerate(judge_names):
            judge_preds = pred_matrix[:, i]
            
            # Weight inversely proportional to disagreement with group
            group_median = np.median(pred_matrix, axis=1)
            disagreement = np.median(np.abs(judge_preds - group_median))
            
            # Convert disagreement to weight (lower disagreement = higher weight)
            weight = 1.0 / (1.0 + disagreement)
            
            judge_weights[judge_name] = JudgeWeight(
                judge_id=judge_name,
                reliability_weight=weight,
                disagreement_penalty=0.0
            )
        
        # Normalize weights
        total_weight = sum(w.reliability_weight for w in judge_weights.values())
        for weight in judge_weights.values():
            weight.reliability_weight /= total_weight
        
        return judge_weights
    
    def _compute_adaptive_weights(self, judge_predictions: Dict[str, np.ndarray]) -> Dict[str, JudgeWeight]:
        """Compute weights that adapt based on prediction difficulty."""
        judge_names = list(judge_predictions.keys())
        pred_matrix = np.column_stack([judge_predictions[name] for name in judge_names])
        
        judge_weights = {}
        
        # Identify "difficult" predictions based on judge disagreement
        judge_std = np.std(pred_matrix, axis=1)
        difficulty_threshold = np.median(judge_std)
        
        for i, judge_name in enumerate(judge_names):
            judge_preds = pred_matrix[:, i]
            
            # For difficult predictions, weight by consistency with ensemble
            # For easy predictions, weight equally
            ensemble_mean = np.mean(pred_matrix, axis=1)
            consistency = np.exp(-np.abs(judge_preds - ensemble_mean))
            
            # Adaptive weight: use consistency for difficult cases, equal for easy
            easy_mask = judge_std <= difficulty_threshold
            adaptive_weight = np.where(easy_mask, 1.0, consistency)
            weight = np.mean(adaptive_weight)
            
            judge_weights[judge_name] = JudgeWeight(
                judge_id=judge_name,
                reliability_weight=weight,
                disagreement_penalty=0.0
            )
        
        # Normalize weights
        total_weight = sum(w.reliability_weight for w in judge_weights.values())
        for weight in judge_weights.values():
            weight.reliability_weight /= total_weight
        
        return judge_weights
    
    def _fit_weighted_ensemble(self, judge_predictions: Dict[str, np.ndarray], **kwargs):
        """Fit weighted ensemble conformal predictor."""
        # Create weighted average predictions
        judge_names = list(judge_predictions.keys())
        pred_matrix = np.column_stack([judge_predictions[name] for name in judge_names])
        
        weights = np.array([self.judge_weights_[name].get_effective_weight() for name in judge_names])
        weighted_predictions = np.dot(pred_matrix, weights)
        
        # Split for calibration
        n_total = len(weighted_predictions)
        n_cal = int(n_total * self.calibration_ratio)
        
        indices = np.random.permutation(n_total)
        cal_indices = indices[:n_cal]
        
        # Create conformal predictor
        if self._is_classification(pred_matrix):
            self.conformal_predictors_['ensemble'] = SplitCP()
        else:
            self.conformal_predictors_['ensemble'] = RegressionSplitCP()
        
        # For calibration, we use the weighted predictions as both input and target
        # This is a limitation - in practice you'd want actual model features
        cal_features = torch.FloatTensor(weighted_predictions[cal_indices].reshape(-1, 1)).to(self.device)
        cal_targets = torch.FloatTensor(weighted_predictions[cal_indices]).to(self.device)
        
        # Simple identity model for this example
        identity_model = torch.nn.Identity().to(self.device)
        cal_predictions = identity_model(cal_features).squeeze()
        
        # Calibrate
        self.conformal_predictors_['ensemble'].calibrate(cal_predictions, cal_targets, alpha=self.alpha)
        
        # Store for prediction
        self._weighted_predictions = weighted_predictions
        self._identity_model = identity_model
    
    def _fit_majority_vote(self, judge_predictions: Dict[str, np.ndarray], **kwargs):
        """Fit majority vote conformal predictor."""
        # Implementation for majority voting with conformal prediction
        judge_names = list(judge_predictions.keys())
        pred_matrix = np.column_stack([judge_predictions[name] for name in judge_names])
        
        # For each prediction, take majority vote (or median for continuous)
        if self._is_classification(pred_matrix):
            # Discrete majority vote
            majority_predictions = []
            for i in range(len(pred_matrix)):
                values, counts = np.unique(pred_matrix[i], return_counts=True)
                majority_predictions.append(values[np.argmax(counts)])
            majority_predictions = np.array(majority_predictions)
        else:
            # Continuous median
            majority_predictions = np.median(pred_matrix, axis=1)
        
        # Create and calibrate conformal predictor (similar to weighted ensemble)
        n_total = len(majority_predictions)
        n_cal = int(n_total * self.calibration_ratio)
        
        indices = np.random.permutation(n_total)
        cal_indices = indices[:n_cal]
        
        if self._is_classification(pred_matrix):
            self.conformal_predictors_['majority'] = SplitCP()
        else:
            self.conformal_predictors_['majority'] = RegressionSplitCP()
        
        cal_features = torch.FloatTensor(majority_predictions[cal_indices].reshape(-1, 1)).to(self.device)
        cal_targets = torch.FloatTensor(majority_predictions[cal_indices]).to(self.device)
        
        identity_model = torch.nn.Identity().to(self.device)
        cal_predictions = identity_model(cal_features).squeeze()
        
        self.conformal_predictors_['majority'].calibrate(cal_predictions, cal_targets, alpha=self.alpha)
        
        self._majority_predictions = majority_predictions
        self._identity_model = identity_model
    
    def _fit_consensus_only(self, judge_predictions: Dict[str, np.ndarray], **kwargs):
        """Fit conformal predictor using only high-consensus predictions."""
        judge_names = list(judge_predictions.keys())
        pred_matrix = np.column_stack([judge_predictions[name] for name in judge_names])
        
        # Filter to high-consensus predictions
        if self._is_classification(pred_matrix):
            # For classification, require threshold fraction to agree
            consensus_mask = []
            consensus_predictions = []
            
            for i in range(len(pred_matrix)):
                values, counts = np.unique(pred_matrix[i], return_counts=True)
                max_agreement = np.max(counts) / len(pred_matrix[i])
                
                if max_agreement >= self.consensus_threshold:
                    consensus_mask.append(i)
                    consensus_predictions.append(values[np.argmax(counts)])
        else:
            # For continuous, use coefficient of variation
            consensus_mask = []
            consensus_predictions = []
            
            for i in range(len(pred_matrix)):
                cv = np.std(pred_matrix[i]) / (np.abs(np.mean(pred_matrix[i])) + 1e-8)
                
                if cv <= self.disagreement_tolerance:
                    consensus_mask.append(i)
                    consensus_predictions.append(np.mean(pred_matrix[i]))
        
        if len(consensus_predictions) < 10:
            warnings.warn(f"Only {len(consensus_predictions)} high-consensus predictions found. "
                        f"Consider relaxing consensus_threshold or disagreement_tolerance.")
        
        consensus_predictions = np.array(consensus_predictions)
        
        # Fit conformal predictor on consensus predictions
        n_total = len(consensus_predictions)
        n_cal = max(1, int(n_total * self.calibration_ratio))
        
        indices = np.random.permutation(n_total)
        cal_indices = indices[:n_cal]
        
        if self._is_classification(pred_matrix):
            self.conformal_predictors_['consensus'] = SplitCP()
        else:
            self.conformal_predictors_['consensus'] = RegressionSplitCP()
        
        cal_features = torch.FloatTensor(consensus_predictions[cal_indices].reshape(-1, 1)).to(self.device)
        cal_targets = torch.FloatTensor(consensus_predictions[cal_indices]).to(self.device)
        
        identity_model = torch.nn.Identity().to(self.device)
        cal_predictions = identity_model(cal_features).squeeze()
        
        self.conformal_predictors_['consensus'].calibrate(cal_predictions, cal_targets, alpha=self.alpha)
        
        self._consensus_predictions = consensus_predictions
        self._consensus_mask = consensus_mask
        self._identity_model = identity_model
    
    def _fit_hierarchical(self, judge_predictions: Dict[str, np.ndarray], **kwargs):
        """Fit hierarchical conformal predictor with judge clustering."""
        # Simplified hierarchical approach: cluster judges by similarity
        judge_names = list(judge_predictions.keys())
        pred_matrix = np.column_stack([judge_predictions[name] for name in judge_names])
        
        # Compute judge similarity matrix
        similarity_matrix = np.corrcoef(pred_matrix.T)
        
        # Simple clustering: group judges with high correlation
        correlation_threshold = 0.7
        clusters = []
        assigned = set()
        
        for i, judge_name in enumerate(judge_names):
            if judge_name in assigned:
                continue
                
            cluster = [judge_name]
            assigned.add(judge_name)
            
            for j, other_judge in enumerate(judge_names):
                if other_judge != judge_name and other_judge not in assigned:
                    if similarity_matrix[i, j] > correlation_threshold:
                        cluster.append(other_judge)
                        assigned.add(other_judge)
            
            clusters.append(cluster)
        
        # Fit conformal predictor for each cluster
        self.conformal_predictors_ = {}
        self._cluster_predictions = {}
        
        for cluster_id, cluster_judges in enumerate(clusters):
            cluster_preds = np.column_stack([judge_predictions[judge] for judge in cluster_judges])
            cluster_avg = np.mean(cluster_preds, axis=1)
            
            # Calibrate cluster predictor
            n_total = len(cluster_avg)
            n_cal = int(n_total * self.calibration_ratio)
            
            indices = np.random.permutation(n_total)
            cal_indices = indices[:n_cal]
            
            predictor_name = f'cluster_{cluster_id}'
            if self._is_classification(pred_matrix):
                self.conformal_predictors_[predictor_name] = SplitCP()
            else:
                self.conformal_predictors_[predictor_name] = RegressionSplitCP()
            
            cal_features = torch.FloatTensor(cluster_avg[cal_indices].reshape(-1, 1)).to(self.device)
            cal_targets = torch.FloatTensor(cluster_avg[cal_indices]).to(self.device)
            
            identity_model = torch.nn.Identity().to(self.device)
            cal_predictions = identity_model(cal_features).squeeze()
            
            self.conformal_predictors_[predictor_name].calibrate(cal_predictions, cal_targets, alpha=self.alpha)
            
            self._cluster_predictions[predictor_name] = cluster_avg
        
        self._judge_clusters = {judge: i for i, cluster in enumerate(clusters) for judge in cluster}
        self._identity_model = identity_model
    
    def _is_classification(self, pred_matrix: np.ndarray) -> bool:
        """Determine if predictions are for classification or regression."""
        # Simple heuristic: if all values are integers in a small range, treat as classification
        flat_preds = pred_matrix.flatten()
        unique_values = np.unique(flat_preds)
        
        return (len(unique_values) <= 20 and 
                all(isinstance(x, (int, np.integer)) or x.is_integer() for x in unique_values))
    
    def predict(self, new_judge_predictions: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make conformal predictions using the aggregated judges.
        
        Parameters:
        -----------
        new_judge_predictions : Dict[str, np.ndarray], optional
            New predictions from each judge. If None, returns intervals for training data.
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (predictions, uncertainty_measures)
        """
        if not self.fitted_:
            raise ValueError("Must fit predictor before making predictions")
        
        if new_judge_predictions is None:
            # Return predictions for training data
            if hasattr(self, '_weighted_predictions'):
                base_predictions = self._weighted_predictions
            elif hasattr(self, '_majority_predictions'):
                base_predictions = self._majority_predictions
            elif hasattr(self, '_consensus_predictions'):
                base_predictions = self._consensus_predictions
            else:
                # Hierarchical case - use first cluster
                base_predictions = list(self._cluster_predictions.values())[0]
        else:
            # Aggregate new predictions
            base_predictions = self._aggregate_new_predictions(new_judge_predictions)
        
        # Get conformal prediction intervals
        predictor = list(self.conformal_predictors_.values())[0]  # Use first predictor
        test_features = torch.FloatTensor(base_predictions.reshape(-1, 1)).to(self.device)
        test_predictions = self._identity_model(test_features).squeeze()
        
        intervals = predictor.predict(test_predictions)
        
        if isinstance(intervals, tuple):
            # Regression case
            lower, upper = intervals
            predictions = base_predictions
            uncertainties = (upper - lower).cpu().numpy()
        else:
            # Classification case
            predictions = base_predictions
            uncertainties = np.array([len(interval) for interval in intervals])
        
        return predictions, uncertainties
    
    def _aggregate_new_predictions(self, new_judge_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Aggregate new judge predictions using the fitted aggregation method."""
        if self.aggregation_method == "weighted_ensemble":
            judge_names = list(new_judge_predictions.keys())
            pred_matrix = np.column_stack([new_judge_predictions[name] for name in judge_names])
            weights = np.array([self.judge_weights_[name].get_effective_weight() for name in judge_names])
            return np.dot(pred_matrix, weights)
        
        elif self.aggregation_method == "majority_vote":
            pred_matrix = np.column_stack(list(new_judge_predictions.values()))
            if self._is_classification(pred_matrix):
                # Majority vote
                majority_preds = []
                for i in range(len(pred_matrix)):
                    values, counts = np.unique(pred_matrix[i], return_counts=True)
                    majority_preds.append(values[np.argmax(counts)])
                return np.array(majority_preds)
            else:
                # Median
                return np.median(pred_matrix, axis=1)
        
        # Add other aggregation methods as needed
        raise NotImplementedError(f"Aggregation method {self.aggregation_method} not implemented for new predictions")
    
    def get_judge_weights(self) -> Dict[str, float]:
        """Get effective weights for each judge."""
        return {name: weight.get_effective_weight() for name, weight in self.judge_weights_.items()}
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the multi-judge predictor."""
        if not self.fitted_:
            return {"fitted": False}
        
        diagnostics = {
            "fitted": True,
            "aggregation_method": self.aggregation_method,
            "conformal_method": self.conformal_method,
            "alpha": self.alpha,
            "coverage_level": 1 - self.alpha,
            "judge_weight_strategy": self.judge_weight_strategy,
            "num_judges": len(self.judge_names_),
            "judge_names": self.judge_names_.copy(),
            "judge_weights": self.get_judge_weights(),
            "consensus_threshold": self.consensus_threshold,
            "disagreement_tolerance": self.disagreement_tolerance,
            "device": str(self.device)
        }
        
        if hasattr(self, '_judge_clusters'):
            diagnostics["num_clusters"] = len(set(self._judge_clusters.values()))
            diagnostics["judge_clusters"] = self._judge_clusters.copy()
        
        return diagnostics

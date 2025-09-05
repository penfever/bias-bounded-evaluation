"""
Historical analysis based bias sensitivity estimation
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List
from .base import SensitivityEstimator


class HistoricalSensitivity(SensitivityEstimator):
    """
    Estimate bias sensitivity using historical analysis of judge inconsistencies.
    
    This method analyzes past judgment data to identify patterns of inconsistency
    and uses statistical measures to estimate bias sensitivity.
    """
    
    def __init__(self, 
                 percentile: float = 95,
                 window_size: Optional[int] = None,
                 min_samples: int = 10,
                 **kwargs):
        """
        Initialize historical sensitivity estimator.
        
        Parameters:
        -----------
        percentile : float
            Percentile to use for sensitivity estimation (default: 95th percentile)
        window_size : int, optional
            Size of rolling window for temporal analysis
        min_samples : int
            Minimum number of samples required for analysis
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        self.percentile = percentile
        self.window_size = window_size
        self.min_samples = min_samples
        self._historical_differences = []
        self._temporal_patterns = {}
        self._consistency_metrics = {}
        
    def fit(self, judgments: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'HistoricalSensitivity':
        """
        Fit historical sensitivity by analyzing judgment patterns.
        
        Parameters:
        -----------
        judgments : DataFrame or array-like
            Historical judgment data to analyze
        **kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        HistoricalSensitivity : Self for method chaining
        """
        judgments = self._validate_input(judgments)
        
        if isinstance(judgments, pd.DataFrame):
            self._fit_dataframe(judgments, **kwargs)
        else:
            self._fit_array(judgments, **kwargs)
            
        self._fitted = True
        return self
    
    def _fit_dataframe(self, df: pd.DataFrame, **kwargs):
        """Fit using DataFrame with potential temporal and categorical structure."""
        # Look for relevant columns
        score_cols = [col for col in df.columns if 'score' in col.lower()]
        time_cols = [col for col in df.columns if any(time_word in col.lower() 
                    for time_word in ['time', 'date', 'timestamp', 'created'])]
        id_cols = [col for col in df.columns if any(id_word in col.lower()
                  for id_word in ['id', 'question', 'model', 'judge'])]
        
        if not score_cols:
            raise ValueError("No score columns found in DataFrame")
        
        # Analyze each score column
        for score_col in score_cols:
            scores = df[score_col].dropna()
            if len(scores) >= self.min_samples:
                # Basic variability analysis
                self._analyze_basic_variability(scores.values, score_col)
                
                # Temporal analysis if time column available
                if time_cols:
                    self._analyze_temporal_patterns(df, score_col, time_cols[0])
                
                # Categorical analysis if ID columns available
                if id_cols:
                    self._analyze_categorical_patterns(df, score_col, id_cols)
    
    def _fit_array(self, judgments: np.ndarray, **kwargs):
        """Fit using array input (basic variability analysis only)."""
        if len(judgments) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} samples for historical analysis")
        
        self._analyze_basic_variability(judgments, "scores")
    
    def _analyze_basic_variability(self, scores: np.ndarray, column_name: str):
        """Analyze basic variability patterns in scores."""
        # Calculate pairwise differences
        differences = []
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                diff = abs(scores[i] - scores[j])
                differences.append(diff)
        
        if differences:
            self._historical_differences.extend(differences)
            
            # Store metrics for this column
            self._consistency_metrics[column_name] = {
                'mean_difference': float(np.mean(differences)),
                'std_difference': float(np.std(differences)),
                'percentile_difference': float(np.percentile(differences, self.percentile)),
                'max_difference': float(np.max(differences)),
                'n_comparisons': len(differences),
                'score_range': float(scores.max() - scores.min()),
                'cv': float(np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
            }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, score_col: str, time_col: str):
        """Analyze temporal patterns in judgment consistency."""
        try:
            # Try to parse time column
            df_time = df[[score_col, time_col]].dropna()
            df_time[time_col] = pd.to_datetime(df_time[time_col])
            df_time = df_time.sort_values(time_col)
            
            scores = df_time[score_col].values
            
            # Calculate rolling statistics if window size specified
            if self.window_size and len(scores) >= self.window_size:
                rolling_means = []
                rolling_stds = []
                
                for i in range(len(scores) - self.window_size + 1):
                    window = scores[i:i + self.window_size]
                    rolling_means.append(np.mean(window))
                    rolling_stds.append(np.std(window))
                
                # Store temporal patterns
                self._temporal_patterns[score_col] = {
                    'rolling_mean_std': float(np.std(rolling_means)),
                    'rolling_std_mean': float(np.mean(rolling_stds)),
                    'temporal_drift': float(np.mean(rolling_means[-5:]) - np.mean(rolling_means[:5])) if len(rolling_means) >= 10 else 0,
                    'window_size': self.window_size
                }
            
            # Calculate adjacent differences (temporal consistency)
            adjacent_diffs = [abs(scores[i] - scores[i-1]) for i in range(1, len(scores))]
            if adjacent_diffs:
                self._historical_differences.extend(adjacent_diffs)
                
                # Store temporal consistency metrics
                temporal_key = f"{score_col}_temporal"
                self._consistency_metrics[temporal_key] = {
                    'mean_adjacent_diff': float(np.mean(adjacent_diffs)),
                    'percentile_adjacent_diff': float(np.percentile(adjacent_diffs, self.percentile)),
                    'max_adjacent_diff': float(np.max(adjacent_diffs)),
                    'temporal_consistency': float(1 - np.std(adjacent_diffs) / np.mean(scores)) if np.mean(scores) > 0 else 0
                }
                
        except Exception as e:
            # If temporal analysis fails, continue with basic analysis
            pass
    
    def _analyze_categorical_patterns(self, df: pd.DataFrame, score_col: str, id_cols: List[str]):
        """Analyze patterns across categorical groupings."""
        for id_col in id_cols:
            try:
                # Group by categorical variable
                grouped = df.groupby(id_col)[score_col].agg(['count', 'mean', 'std']).dropna()
                
                # Only consider groups with multiple observations
                multi_obs = grouped[grouped['count'] > 1]
                
                if len(multi_obs) > 0:
                    # Calculate between-group and within-group variability
                    between_var = float(np.var(multi_obs['mean']))
                    within_var = float(np.mean(multi_obs['std'] ** 2))
                    
                    # Store categorical patterns
                    categorical_key = f"{score_col}_by_{id_col}"
                    self._consistency_metrics[categorical_key] = {
                        'between_group_variance': between_var,
                        'within_group_variance': within_var,
                        'variance_ratio': between_var / within_var if within_var > 0 else float('inf'),
                        'n_groups': len(multi_obs),
                        'group_consistency': float(1 - within_var / between_var) if between_var > 0 else 0
                    }
                    
                    # Add within-group differences to historical differences
                    for _, group_data in df.groupby(id_col)[score_col]:
                        if len(group_data) > 1:
                            group_scores = group_data.values
                            group_diffs = [abs(group_scores[i] - group_scores[j]) 
                                         for i in range(len(group_scores)) 
                                         for j in range(i + 1, len(group_scores))]
                            self._historical_differences.extend(group_diffs)
            
            except Exception as e:
                # If categorical analysis fails, continue
                continue
    
    def estimate(self, score_range: Optional[float] = None) -> float:
        """
        Estimate bias sensitivity from historical analysis.
        
        Parameters:
        -----------
        score_range : float, optional
            Range of judgment scores. Used for validation.
            
        Returns:
        --------
        float : Bias sensitivity estimate
        """
        self._validate_fitted()
        
        if not self._historical_differences:
            raise ValueError("No historical differences calculated")
        
        # Use specified percentile as conservative estimate
        bias_sensitivity = float(np.percentile(self._historical_differences, self.percentile))
        
        # Validate against score range if provided
        if score_range is not None and bias_sensitivity > score_range:
            # Cap at score range
            bias_sensitivity = score_range
        
        self._bias_sensitivity = bias_sensitivity
        return bias_sensitivity
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about historical analysis.
        
        Returns:
        --------
        Dict[str, Any] : Diagnostic information
        """
        diagnostics = super().get_diagnostics()
        
        if self._fitted:
            diagnostics.update({
                "percentile": self.percentile,
                "window_size": self.window_size,
                "min_samples": self.min_samples,
                "n_differences": len(self._historical_differences),
                "consistency_metrics": self._consistency_metrics,
                "temporal_patterns": self._temporal_patterns
            })
            
            if self._historical_differences:
                # Summary statistics of historical differences
                diffs = np.array(self._historical_differences)
                diagnostics.update({
                    "difference_statistics": {
                        "mean": float(np.mean(diffs)),
                        "std": float(np.std(diffs)),
                        "median": float(np.median(diffs)),
                        "p25": float(np.percentile(diffs, 25)),
                        "p75": float(np.percentile(diffs, 75)),
                        "p95": float(np.percentile(diffs, 95)),
                        "p99": float(np.percentile(diffs, 99)),
                        "max": float(np.max(diffs))
                    }
                })
        
        return diagnostics
    
    def get_consistency_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed consistency metrics for each analyzed dimension.
        
        Returns:
        --------
        Dict[str, Dict[str, Any]] : Consistency metrics by dimension
        """
        return self._consistency_metrics.copy()
    
    def get_temporal_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Get temporal pattern analysis results.
        
        Returns:
        --------
        Dict[str, Dict[str, Any]] : Temporal patterns by score column
        """
        return self._temporal_patterns.copy()
    
    def get_stability_score(self) -> Optional[float]:
        """
        Get overall stability score based on historical patterns.
        
        Returns:
        --------
        float or None : Stability score (0-1, higher is more stable)
        """
        if not self._fitted or not self._historical_differences:
            return None
        
        # Calculate stability based on coefficient of variation of differences
        diffs = np.array(self._historical_differences)
        cv_diffs = np.std(diffs) / np.mean(diffs) if np.mean(diffs) > 0 else float('inf')
        
        # Convert to 0-1 scale (lower CV = higher stability)
        stability = 1 / (1 + cv_diffs)
        
        return float(stability)
    
    def set_percentile(self, percentile: float):
        """
        Set the percentile for sensitivity estimation.
        
        Parameters:
        -----------
        percentile : float
            Percentile value (0-100)
        """
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100")
        
        self.percentile = percentile
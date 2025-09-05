"""
Empirical bias sensitivity estimation
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List, Callable
from .base import SensitivityEstimator


class EmpiricalSensitivity(SensitivityEstimator):
    """
    Estimate bias sensitivity through empirical measurement of judgment variations.
    
    This method directly measures how much judgments change when bias-inducing
    factors are manipulated, providing a data-driven estimate of bias sensitivity.
    """
    
    def __init__(self, 
                 bias_triggers: Optional[List[str]] = None,
                 measurement_function: Optional[Callable] = None,
                 **kwargs):
        """
        Initialize empirical sensitivity estimator.
        
        Parameters:
        -----------
        bias_triggers : List[str], optional
            List of bias trigger names to test
        measurement_function : Callable, optional
            Custom function to measure bias sensitivity
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        self.bias_triggers = bias_triggers or [
            'verbose_vs_concise',
            'formal_vs_casual', 
            'with_vs_without_equations',
            'long_vs_short',
            'detailed_vs_brief'
        ]
        self.measurement_function = measurement_function
        self._measured_differences = []
        self._trigger_results = {}
        
    def fit(self, judgments: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'EmpiricalSensitivity':
        """
        Fit empirical sensitivity by measuring judgment variations.
        
        Parameters:
        -----------
        judgments : DataFrame or array-like
            Judgment data. For DataFrame, expects columns for different
            bias trigger conditions. For array, uses variance-based estimate.
        **kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        EmpiricalSensitivity : Self for method chaining
        """
        judgments = self._validate_input(judgments)
        
        if isinstance(judgments, pd.DataFrame):
            self._fit_dataframe(judgments, **kwargs)
        else:
            self._fit_array(judgments, **kwargs)
            
        self._fitted = True
        return self
    
    def _fit_dataframe(self, df: pd.DataFrame, **kwargs):
        """Fit using DataFrame with bias trigger columns."""
        if self.measurement_function:
            # Use custom measurement function
            sensitivity = self.measurement_function(df)
            self._measured_differences = [sensitivity]
        else:
            # Look for paired columns that indicate bias triggers
            self._measure_pairwise_differences(df)
    
    def _measure_pairwise_differences(self, df: pd.DataFrame):
        """Measure differences between bias trigger pairs."""
        differences = []
        
        # Look for columns that might represent bias trigger pairs
        for trigger in self.bias_triggers:
            trigger_cols = self._find_trigger_columns(df, trigger)
            
            if len(trigger_cols) >= 2:
                # Measure differences between trigger conditions
                trigger_diffs = self._measure_trigger_differences(df, trigger_cols, trigger)
                differences.extend(trigger_diffs)
                
        if not differences:
            # Fallback: use overall variance as proxy
            score_cols = [col for col in df.columns if 'score' in col.lower()]
            if score_cols:
                for col in score_cols:
                    scores = df[col].dropna()
                    if len(scores) > 1:
                        # Use range as proxy for bias sensitivity
                        differences.append(scores.max() - scores.min())
        
        self._measured_differences = differences
    
    def _find_trigger_columns(self, df: pd.DataFrame, trigger: str) -> List[str]:
        """Find columns related to a specific bias trigger."""
        trigger_cols = []
        
        # Common patterns for bias trigger column names
        patterns = {
            'verbose_vs_concise': ['verbose', 'concise', 'detailed', 'brief'],
            'formal_vs_casual': ['formal', 'casual', 'professional', 'informal'],
            'with_vs_without_equations': ['equations', 'math', 'formula', 'symbolic'],
            'long_vs_short': ['long', 'short', 'length'],
            'detailed_vs_brief': ['detailed', 'brief', 'comprehensive', 'summary']
        }
        
        if trigger in patterns:
            for pattern in patterns[trigger]:
                matching_cols = [col for col in df.columns 
                               if pattern.lower() in col.lower()]
                trigger_cols.extend(matching_cols)
        
        return list(set(trigger_cols))  # Remove duplicates
    
    def _measure_trigger_differences(self, df: pd.DataFrame, 
                                   trigger_cols: List[str], 
                                   trigger_name: str) -> List[float]:
        """Measure differences for a specific trigger."""
        differences = []
        
        # For each pair of trigger columns, measure differences
        for i in range(len(trigger_cols)):
            for j in range(i + 1, len(trigger_cols)):
                col1, col2 = trigger_cols[i], trigger_cols[j]
                
                # Get overlapping data
                common_idx = df[[col1, col2]].dropna().index
                if len(common_idx) > 0:
                    scores1 = df.loc[common_idx, col1]
                    scores2 = df.loc[common_idx, col2]
                    
                    # Calculate mean absolute difference
                    diff = float(np.mean(np.abs(scores1 - scores2)))
                    differences.append(diff)
                    
                    # Store result for diagnostics
                    self._trigger_results[f"{trigger_name}_{col1}_vs_{col2}"] = {
                        'mean_diff': diff,
                        'max_diff': float(np.max(np.abs(scores1 - scores2))),
                        'n_samples': len(common_idx)
                    }
        
        return differences
    
    def _fit_array(self, judgments: np.ndarray, **kwargs):
        """Fit using array input (uses variance-based estimate)."""
        # Without paired data, use variance as proxy for sensitivity
        if len(judgments) > 1:
            # Use range as conservative estimate
            sensitivity = float(judgments.max() - judgments.min())
            self._measured_differences = [sensitivity]
        else:
            # Single judgment - cannot estimate
            self._measured_differences = [0.0]
    
    def estimate(self, score_range: Optional[float] = None) -> float:
        """
        Estimate bias sensitivity from empirical measurements.
        
        Parameters:
        -----------
        score_range : float, optional
            Range of judgment scores. Used for validation/scaling.
            
        Returns:
        --------
        float : Bias sensitivity estimate
        """
        self._validate_fitted()
        
        if not self._measured_differences:
            raise ValueError("No bias differences measured. Check input data.")
        
        # Use maximum measured difference as conservative estimate
        bias_sensitivity = float(np.max(self._measured_differences))
        
        # Validate against score range if provided
        if score_range is not None and bias_sensitivity > score_range:
            # Cap at score range (can't be more sensitive than total range)
            bias_sensitivity = score_range
        
        self._bias_sensitivity = bias_sensitivity
        return bias_sensitivity
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about empirical measurements.
        
        Returns:
        --------
        Dict[str, Any] : Diagnostic information
        """
        diagnostics = super().get_diagnostics()
        
        if self._fitted:
            diagnostics.update({
                "measured_differences": self._measured_differences,
                "n_measurements": len(self._measured_differences),
                "mean_difference": float(np.mean(self._measured_differences)) if self._measured_differences else 0,
                "max_difference": float(np.max(self._measured_differences)) if self._measured_differences else 0,
                "bias_triggers_tested": self.bias_triggers,
                "trigger_results": self._trigger_results
            })
        
        return diagnostics
    
    def get_trigger_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed results for each bias trigger tested.
        
        Returns:
        --------
        Dict[str, Dict] : Results for each trigger
        """
        return self._trigger_results.copy()
    
    def add_custom_measurement(self, measurement: float, name: str = "custom"):
        """
        Add a custom bias sensitivity measurement.
        
        Parameters:
        -----------
        measurement : float
            Bias sensitivity measurement value
        name : str
            Name/description of the measurement
        """
        self._measured_differences.append(measurement)
        self._trigger_results[name] = {
            'measurement': measurement,
            'type': 'custom'
        }
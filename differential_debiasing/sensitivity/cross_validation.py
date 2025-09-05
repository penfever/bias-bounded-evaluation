"""
Cross-validation based bias sensitivity estimation
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List, Callable
from .base import SensitivityEstimator


class CrossValidationSensitivity(SensitivityEstimator):
    """
    Estimate bias sensitivity using cross-validation with systematic bias variations.
    
    This method systematically varies non-rubric factors and measures how much
    judgments change, providing an empirical estimate of bias sensitivity.
    """
    
    def __init__(self, 
                 n_folds: int = 5,
                 variation_functions: Optional[List[Callable]] = None,
                 random_seed: Optional[int] = None,
                 **kwargs):
        """
        Initialize cross-validation sensitivity estimator.
        
        Parameters:
        -----------
        n_folds : int
            Number of cross-validation folds
        variation_functions : List[Callable], optional
            Functions to apply systematic variations to judgments
        random_seed : int, optional
            Random seed for reproducible results
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        self.n_folds = n_folds
        self.variation_functions = variation_functions or self._get_default_variations()
        self.random_seed = random_seed
        self._rng = np.random.RandomState(random_seed)
        self._cv_results = []
        self._fold_differences = []
        
    def _get_default_variations(self) -> List[Callable]:
        """Get default variation functions for bias testing."""
        return [
            self._length_variation,
            self._style_variation,
            self._ordering_variation,
            self._formatting_variation,
            self._repetition_variation
        ]
    
    def _length_variation(self, judgments: np.ndarray) -> np.ndarray:
        """Simulate length bias by slightly adjusting scores based on position."""
        # Simulate length bias - later positions get slightly higher scores
        length_bias = np.linspace(0, 0.1, len(judgments))
        return judgments + length_bias
    
    def _style_variation(self, judgments: np.ndarray) -> np.ndarray:
        """Simulate style bias through random systematic shifts."""
        # Simulate style preferences - every other judgment gets small boost
        style_bias = np.array([0.05 if i % 2 == 0 else -0.05 for i in range(len(judgments))])
        return judgments + style_bias
    
    def _ordering_variation(self, judgments: np.ndarray) -> np.ndarray:
        """Simulate ordering bias through position-based adjustments."""
        # Simulate ordering effects - middle positions get slight preference
        n = len(judgments)
        middle = n // 2
        ordering_bias = np.array([0.03 * (1 - abs(i - middle) / middle) for i in range(n)])
        return judgments + ordering_bias
    
    def _formatting_variation(self, judgments: np.ndarray) -> np.ndarray:
        """Simulate formatting bias through periodic adjustments."""
        # Simulate formatting preferences - periodic bias pattern
        formatting_bias = 0.02 * np.sin(np.linspace(0, 4 * np.pi, len(judgments)))
        return judgments + formatting_bias
    
    def _repetition_variation(self, judgments: np.ndarray) -> np.ndarray:
        """Simulate repetition bias through clustering effects."""
        # Simulate repetition bias - similar judgments get clustered
        repetition_bias = np.cumsum(self._rng.normal(0, 0.01, len(judgments)))
        repetition_bias = repetition_bias - np.mean(repetition_bias)  # Center around 0
        return judgments + repetition_bias
    
    def fit(self, judgments: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'CrossValidationSensitivity':
        """
        Fit cross-validation sensitivity by measuring judgment variations.
        
        Parameters:
        -----------
        judgments : DataFrame or array-like
            Judgment data to analyze
        **kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        CrossValidationSensitivity : Self for method chaining
        """
        judgments = self._validate_input(judgments)
        
        if isinstance(judgments, pd.DataFrame):
            self._fit_dataframe(judgments, **kwargs)
        else:
            self._fit_array(judgments, **kwargs)
            
        self._fitted = True
        return self
    
    def _fit_dataframe(self, df: pd.DataFrame, **kwargs):
        """Fit using DataFrame with multiple score columns."""
        # Extract score columns
        score_cols = [col for col in df.columns if 'score' in col.lower()]
        
        if not score_cols:
            raise ValueError("No score columns found in DataFrame")
        
        # Perform cross-validation for each score column
        for col in score_cols:
            scores = df[col].dropna().values
            if len(scores) > 0:
                cv_results = self._perform_cross_validation(scores)
                self._cv_results.extend(cv_results)
    
    def _fit_array(self, judgments: np.ndarray, **kwargs):
        """Fit using array input."""
        cv_results = self._perform_cross_validation(judgments)
        self._cv_results = cv_results
    
    def _perform_cross_validation(self, judgments: np.ndarray) -> List[Dict[str, Any]]:
        """Perform cross-validation with bias variations."""
        if len(judgments) < self.n_folds:
            raise ValueError(f"Need at least {self.n_folds} judgments for {self.n_folds}-fold CV")
        
        # Create fold indices
        fold_size = len(judgments) // self.n_folds
        fold_indices = []
        
        for i in range(self.n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.n_folds - 1 else len(judgments)
            fold_indices.append(list(range(start_idx, end_idx)))
        
        cv_results = []
        
        # For each variation function
        for var_idx, variation_func in enumerate(self.variation_functions):
            variation_name = variation_func.__name__
            fold_differences = []
            
            # Apply variation and measure differences across folds
            for fold_idx in range(self.n_folds):
                # Get fold data
                fold_data = judgments[fold_indices[fold_idx]]
                
                if len(fold_data) > 0:
                    # Apply variation
                    varied_data = variation_func(fold_data)
                    
                    # Measure difference
                    fold_diff = np.mean(np.abs(varied_data - fold_data))
                    fold_differences.append(fold_diff)
            
            if fold_differences:
                # Store results for this variation
                cv_result = {
                    'variation': variation_name,
                    'fold_differences': fold_differences,
                    'mean_difference': np.mean(fold_differences),
                    'std_difference': np.std(fold_differences),
                    'max_difference': np.max(fold_differences),
                    'n_folds': len(fold_differences)
                }
                cv_results.append(cv_result)
        
        return cv_results
    
    def estimate(self, score_range: Optional[float] = None) -> float:
        """
        Estimate bias sensitivity from cross-validation results.
        
        Parameters:
        -----------
        score_range : float, optional
            Range of judgment scores. Used for validation.
            
        Returns:
        --------
        float : Bias sensitivity estimate
        """
        self._validate_fitted()
        
        if not self._cv_results:
            raise ValueError("No cross-validation results available")
        
        # Use maximum mean difference across all variations as conservative estimate
        max_differences = [result['max_difference'] for result in self._cv_results]
        bias_sensitivity = float(np.max(max_differences))
        
        # Validate against score range if provided
        if score_range is not None and bias_sensitivity > score_range:
            # Cap at score range
            bias_sensitivity = score_range
        
        self._bias_sensitivity = bias_sensitivity
        return bias_sensitivity
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about cross-validation results.
        
        Returns:
        --------
        Dict[str, Any] : Diagnostic information
        """
        diagnostics = super().get_diagnostics()
        
        if self._fitted:
            diagnostics.update({
                "n_folds": self.n_folds,
                "n_variations": len(self.variation_functions),
                "cv_results": self._cv_results,
                "variation_names": [func.__name__ for func in self.variation_functions],
                "random_seed": self.random_seed
            })
            
            if self._cv_results:
                # Summary statistics
                all_mean_diffs = [result['mean_difference'] for result in self._cv_results]
                all_max_diffs = [result['max_difference'] for result in self._cv_results]
                
                diagnostics.update({
                    "overall_mean_difference": float(np.mean(all_mean_diffs)),
                    "overall_max_difference": float(np.max(all_max_diffs)),
                    "variation_consistency": float(np.std(all_mean_diffs)),
                })
        
        return diagnostics
    
    def get_cv_results(self) -> List[Dict[str, Any]]:
        """
        Get detailed cross-validation results.
        
        Returns:
        --------
        List[Dict[str, Any]] : Cross-validation results for each variation
        """
        return self._cv_results.copy()
    
    def get_variation_ranking(self) -> List[Dict[str, Any]]:
        """
        Get variations ranked by their impact on judgments.
        
        Returns:
        --------
        List[Dict[str, Any]] : Variations ranked by impact
        """
        if not self._fitted:
            raise ValueError("Must fit estimator before getting variation ranking")
        
        # Rank variations by their maximum difference
        ranked_variations = sorted(
            self._cv_results,
            key=lambda x: x['max_difference'],
            reverse=True
        )
        
        # Format for output
        ranking = []
        for i, result in enumerate(ranked_variations):
            ranking.append({
                'rank': i + 1,
                'variation': result['variation'],
                'impact': result['max_difference'],
                'consistency': result['std_difference'],
                'relative_impact': result['max_difference'] / ranked_variations[0]['max_difference']
            })
        
        return ranking
    
    def add_variation_function(self, func: Callable, name: Optional[str] = None):
        """
        Add a custom variation function.
        
        Parameters:
        -----------
        func : Callable
            Function that takes judgments array and returns varied judgments
        name : str, optional
            Name for the variation function
        """
        if name:
            func.__name__ = name
        
        self.variation_functions.append(func)
        
        # If already fitted, re-fit with new variation
        if self._fitted:
            self._fitted = False
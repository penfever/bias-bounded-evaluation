"""
Factor analysis based bias sensitivity estimation
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List
from .base import SensitivityEstimator
from ..core.utils import estimate_r_squared_from_scores


class FactorAnalysisSensitivity(SensitivityEstimator):
    """
    Estimate bias sensitivity using factor analysis approach.
    
    This method estimates bias sensitivity as sqrt(1 - R²) where R² is the
    proportion of variance in overall judgments explained by explicit factors.
    
    The intuition is that unexplained variance represents potential bias,
    providing a conservative estimate of bias sensitivity.
    """
    
    def __init__(self, 
                 factor_columns: Optional[List[str]] = None,
                 target_column: str = 'score',
                 use_factor_analyzer: bool = False,
                 n_factors: Optional[int] = None,
                 **kwargs):
        """
        Initialize factor analysis sensitivity estimator.
        
        Parameters:
        -----------
        factor_columns : List[str], optional
            Column names for factor scores. If None, will auto-detect.
        target_column : str
            Column name for overall/target score
        use_factor_analyzer : bool
            Whether to use factor_analyzer library for factor analysis
        n_factors : int, optional
            Number of factors to extract (if using factor_analyzer)
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        self.factor_columns = factor_columns
        self.target_column = target_column
        self.use_factor_analyzer = use_factor_analyzer
        self.n_factors = n_factors
        self._r_squared = None
        self._factor_loadings = None
        
    def fit(self, judgments: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'FactorAnalysisSensitivity':
        """
        Fit factor analysis to judgment data.
        
        Parameters:
        -----------
        judgments : DataFrame or array-like
            Judgment data. If DataFrame, should contain factor and target columns.
            If array-like, will treat as single judgment vector.
        **kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        FactorAnalysisSensitivity : Self for method chaining
        """
        judgments = self._validate_input(judgments)
        
        if isinstance(judgments, pd.DataFrame):
            self._fit_dataframe(judgments, **kwargs)
        else:
            self._fit_array(judgments, **kwargs)
            
        self._fitted = True
        return self
    
    def _fit_dataframe(self, df: pd.DataFrame, **kwargs):
        """Fit factor analysis using DataFrame with multiple columns."""
        # Auto-detect factor columns if not provided
        if self.factor_columns is None:
            # Look for columns with 'score' in name, excluding target column
            score_cols = [col for col in df.columns 
                         if 'score' in col.lower() and col != self.target_column]
            
            if len(score_cols) == 0:
                raise ValueError("No factor score columns found. Specify factor_columns explicitly.")
            
            self.factor_columns = score_cols
        
        # Validate columns exist
        missing_cols = [col for col in self.factor_columns + [self.target_column] 
                       if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Calculate R² using linear regression
        self._r_squared = estimate_r_squared_from_scores(
            df, self.factor_columns, self.target_column
        )
        
        # Optionally perform factor analysis for additional diagnostics
        if self.use_factor_analyzer:
            self._perform_factor_analysis(df)
    
    def _fit_array(self, judgments: np.ndarray, **kwargs):
        """
        Fit using array input (assumes no factor structure available).
        
        For arrays, we use a conservative estimate assuming low explained variance.
        """
        # Without factor structure, use conservative estimate
        # Assume only 10% variance explained (high bias sensitivity)
        self._r_squared = 0.1
        
        # Store some basic statistics
        self._mean_judgment = float(np.mean(judgments))
        self._std_judgment = float(np.std(judgments))
    
    def _perform_factor_analysis(self, df: pd.DataFrame):
        """Perform factor analysis using factor_analyzer library."""
        try:
            from factor_analyzer import FactorAnalyzer
            from factor_analyzer.factor_analyzer import calculate_kmo
        except ImportError:
            raise ImportError("factor_analyzer library required for use_factor_analyzer=True")
        
        # Prepare data (factor columns only)
        factor_data = df[self.factor_columns].dropna()
        
        if len(factor_data) < 10:
            raise ValueError("Need at least 10 samples for factor analysis")
        
        # Determine number of factors
        n_factors = self.n_factors or min(len(self.factor_columns), 3)
        
        # Perform factor analysis
        fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
        fa.fit(factor_data)
        
        # Store results
        self._factor_loadings = fa.loadings_
        
        # Calculate KMO (Kaiser-Meyer-Olkin measure)
        try:
            kmo_all, kmo_model = calculate_kmo(factor_data)
            self._kmo = kmo_model
        except:
            self._kmo = None
    
    def estimate(self, score_range: Optional[float] = None) -> float:
        """
        Estimate bias sensitivity from fitted factor analysis.
        
        Parameters:
        -----------
        score_range : float, optional
            Range of judgment scores. If None, assumes normalized [0,1] range.
            
        Returns:
        --------
        float : Bias sensitivity estimate
        """
        self._validate_fitted()
        
        if self._r_squared is None:
            raise ValueError("R² not calculated. Check fitting process.")
        
        # Conservative bound: sqrt(1 - R²)
        normalized_sensitivity = np.sqrt(1 - self._r_squared)
        
        # Scale to judgment units
        if score_range is None:
            score_range = 1.0  # Assume normalized
        
        bias_sensitivity = normalized_sensitivity * score_range
        self._bias_sensitivity = bias_sensitivity
        
        return bias_sensitivity
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about factor analysis.
        
        Returns:
        --------
        Dict[str, Any] : Diagnostic information
        """
        diagnostics = super().get_diagnostics()
        
        if self._fitted:
            diagnostics.update({
                "r_squared": self._r_squared,
                "explained_variance": self._r_squared,
                "unexplained_variance": 1 - self._r_squared,
                "factor_columns": self.factor_columns,
                "target_column": self.target_column,
            })
            
            if hasattr(self, '_kmo') and self._kmo is not None:
                diagnostics["kmo_measure"] = self._kmo
                
            if hasattr(self, '_factor_loadings') and self._factor_loadings is not None:
                diagnostics["factor_loadings_shape"] = self._factor_loadings.shape
        
        return diagnostics
    
    def get_factor_loadings(self) -> Optional[np.ndarray]:
        """
        Get factor loadings if factor analysis was performed.
        
        Returns:
        --------
        np.ndarray or None : Factor loadings matrix
        """
        return getattr(self, '_factor_loadings', None)
    
    def get_r_squared(self) -> Optional[float]:
        """
        Get R² value from factor analysis.
        
        Returns:
        --------
        float or None : R² value
        """
        return self._r_squared
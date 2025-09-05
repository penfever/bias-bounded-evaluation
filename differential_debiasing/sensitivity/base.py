"""
Base class for bias sensitivity estimators
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional


class SensitivityEstimator(ABC):
    """
    Abstract base class for bias sensitivity estimation methods.
    
    Sensitivity estimators measure how much bias can affect judgments,
    providing the Δ_B f parameter for the bias-bounded mechanism.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize sensitivity estimator.
        
        Parameters:
        -----------
        **kwargs : dict
            Method-specific parameters
        """
        self.params = kwargs
        self._fitted = False
        self._bias_sensitivity = None
        
    @abstractmethod
    def fit(self, judgments: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'SensitivityEstimator':
        """
        Fit the sensitivity estimator to judgment data.
        
        Parameters:
        -----------
        judgments : array-like or DataFrame
            Judgment data to analyze
        **kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        SensitivityEstimator : Self for method chaining
        """
        pass
    
    @abstractmethod
    def estimate(self, score_range: Optional[float] = None) -> float:
        """
        Estimate bias sensitivity from fitted data.
        
        Parameters:
        -----------
        score_range : float, optional
            Range of judgment scores (max - min). 
            If None, will be inferred from data.
            
        Returns:
        --------
        float : Bias sensitivity estimate in judgment units
        """
        pass
    
    def fit_estimate(self, 
                    judgments: Union[np.ndarray, pd.DataFrame], 
                    score_range: Optional[float] = None,
                    **kwargs) -> float:
        """
        Convenience method to fit and estimate in one call.
        
        Parameters:
        -----------
        judgments : array-like or DataFrame
            Judgment data to analyze
        score_range : float, optional
            Range of judgment scores
        **kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        float : Bias sensitivity estimate
        """
        return self.fit(judgments, **kwargs).estimate(score_range)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the sensitivity estimation.
        
        Returns:
        --------
        Dict[str, Any] : Diagnostic information
        """
        if not self._fitted:
            raise ValueError("Must fit estimator before getting diagnostics")
        
        return {
            "method": self.__class__.__name__,
            "fitted": self._fitted,
            "bias_sensitivity": self._bias_sensitivity,
            "params": self.params.copy()
        }
    
    def _validate_fitted(self):
        """Check that estimator has been fitted."""
        if not self._fitted:
            raise ValueError(f"{self.__class__.__name__} must be fitted before estimating sensitivity")
    
    def _validate_input(self, judgments: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Validate and convert input judgments.
        
        Parameters:
        -----------
        judgments : array-like or DataFrame
            Input judgment data
            
        Returns:
        --------
        np.ndarray : Validated judgment array
        """
        if isinstance(judgments, pd.DataFrame):
            if judgments.empty:
                raise ValueError("Input DataFrame is empty")
            return judgments
        else:
            judgments = np.asarray(judgments)
            if judgments.size == 0:
                raise ValueError("Input array is empty")
            if not np.isfinite(judgments).all():
                # Remove non-finite values
                judgments = judgments[np.isfinite(judgments)]
                if judgments.size == 0:
                    raise ValueError("No finite values in input")
            return judgments
    
    def __repr__(self) -> str:
        """String representation of estimator."""
        fitted_status = "fitted" if self._fitted else "not fitted"
        return f"{self.__class__.__name__}({fitted_status})"
"""
Domain-specific bias sensitivity estimation
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List
from .base import SensitivityEstimator


class DomainSpecificSensitivity(SensitivityEstimator):
    """
    Domain-specific bias sensitivity estimation using fixed bounds.
    
    This method uses domain knowledge and expert judgment to set 
    bias sensitivity bounds based on the specific evaluation context.
    """
    
    def __init__(self, 
                 sensitivity_value: Optional[float] = None,
                 domain: str = "general",
                 scale_type: str = "1-10",
                 **kwargs):
        """
        Initialize domain-specific sensitivity estimator.
        
        Parameters:
        -----------
        sensitivity_value : float, optional
            Fixed bias sensitivity value. If None, will use domain defaults.
        domain : str
            Domain type for default sensitivity values
        scale_type : str
            Type of judgment scale (e.g., "1-10", "0-100", "0-1")
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        self.sensitivity_value = sensitivity_value
        self.domain = domain
        self.scale_type = scale_type
        self._domain_defaults = self._get_domain_defaults()
        
    def _get_domain_defaults(self) -> Dict[str, Dict[str, float]]:
        """Get default sensitivity values for different domains and scales."""
        return {
            "general": {
                "1-10": 3.0,      # Moderate bias on 10-point scale
                "0-100": 30.0,    # Moderate bias on 100-point scale  
                "0-1": 0.3,       # Moderate bias on normalized scale
                "1-5": 1.5,       # Moderate bias on 5-point scale
            },
            "academic": {
                "1-10": 2.0,      # Lower bias expected in academic context
                "0-100": 20.0,
                "0-1": 0.2,
                "1-5": 1.0,
            },
            "creative": {
                "1-10": 4.0,      # Higher bias expected for subjective content
                "0-100": 40.0,
                "0-1": 0.4,
                "1-5": 2.0,
            },
            "technical": {
                "1-10": 2.5,      # Moderate-low bias for technical content
                "0-100": 25.0,
                "0-1": 0.25,
                "1-5": 1.25,
            },
            "conversational": {
                "1-10": 3.5,      # Higher bias for style-sensitive content
                "0-100": 35.0,
                "0-1": 0.35,
                "1-5": 1.75,
            },
            "medical": {
                "1-10": 1.5,      # Very low bias tolerance
                "0-100": 15.0,
                "0-1": 0.15,
                "1-5": 0.75,
            },
            "legal": {
                "1-10": 1.0,      # Extremely low bias tolerance
                "0-100": 10.0,
                "0-1": 0.1,
                "1-5": 0.5,
            },
            "conservative": {
                "1-10": 8.0,      # Very conservative (high sensitivity)
                "0-100": 80.0,
                "0-1": 0.8,
                "1-5": 4.0,
            }
        }
    
    def fit(self, judgments: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'DomainSpecificSensitivity':
        """
        Fit domain-specific sensitivity (mainly validation).
        
        Parameters:
        -----------
        judgments : DataFrame or array-like
            Judgment data (used for validation/scale detection)
        **kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        DomainSpecificSensitivity : Self for method chaining
        """
        judgments = self._validate_input(judgments)
        
        # Auto-detect scale if not explicitly set
        if isinstance(judgments, np.ndarray):
            self._infer_scale_from_data(judgments)
        elif isinstance(judgments, pd.DataFrame):
            score_cols = [col for col in judgments.columns if 'score' in col.lower()]
            if score_cols:
                scores = judgments[score_cols[0]].dropna()
                if len(scores) > 0:
                    self._infer_scale_from_data(scores.values)
        
        self._fitted = True
        return self
    
    def _infer_scale_from_data(self, scores: np.ndarray):
        """Infer scale type from score data."""
        min_score = float(scores.min())
        max_score = float(scores.max())
        
        # Heuristics for scale detection
        if min_score >= 0 and max_score <= 1.1:  # Likely 0-1 scale
            self.scale_type = "0-1"
        elif min_score >= 0.8 and max_score <= 10.2:  # Likely 1-10 scale
            self.scale_type = "1-10"
        elif min_score >= 0.8 and max_score <= 5.2:  # Likely 1-5 scale
            self.scale_type = "1-5"
        elif min_score >= -5 and max_score <= 105:  # Likely 0-100 scale
            self.scale_type = "0-100"
        else:
            # Keep current scale_type as fallback
            pass
    
    def estimate(self, score_range: Optional[float] = None) -> float:
        """
        Estimate bias sensitivity using domain-specific bounds.
        
        Parameters:
        -----------
        score_range : float, optional
            Range of judgment scores. If provided, used to validate sensitivity.
            
        Returns:
        --------
        float : Bias sensitivity estimate
        """
        # Use explicit value if provided
        if self.sensitivity_value is not None:
            bias_sensitivity = float(self.sensitivity_value)
        else:
            # Use domain defaults
            if self.domain in self._domain_defaults:
                domain_config = self._domain_defaults[self.domain]
                if self.scale_type in domain_config:
                    bias_sensitivity = domain_config[self.scale_type]
                else:
                    # Fallback to general domain
                    bias_sensitivity = self._domain_defaults["general"]["1-10"]
            else:
                # Unknown domain - use conservative estimate
                bias_sensitivity = self._domain_defaults["conservative"][self.scale_type]
        
        # Validate against score range if provided
        if score_range is not None:
            if bias_sensitivity > score_range:
                # Cap at score range
                bias_sensitivity = score_range
            elif bias_sensitivity < 0.01 * score_range:
                # Ensure minimum sensitivity (1% of range)
                bias_sensitivity = 0.01 * score_range
        
        self._bias_sensitivity = bias_sensitivity
        return bias_sensitivity
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about domain-specific estimation.
        
        Returns:
        --------
        Dict[str, Any] : Diagnostic information
        """
        diagnostics = super().get_diagnostics()
        
        diagnostics.update({
            "domain": self.domain,
            "scale_type": self.scale_type,
            "explicit_value": self.sensitivity_value,
            "available_domains": list(self._domain_defaults.keys()),
            "available_scales": list(self._domain_defaults["general"].keys())
        })
        
        if self._fitted and self.domain in self._domain_defaults:
            diagnostics["domain_defaults"] = self._domain_defaults[self.domain].copy()
        
        return diagnostics
    
    def set_sensitivity(self, value: float):
        """
        Set explicit sensitivity value.
        
        Parameters:
        -----------
        value : float
            Bias sensitivity value
        """
        if value <= 0:
            raise ValueError("Sensitivity value must be positive")
        self.sensitivity_value = value
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domain types."""
        return list(self._domain_defaults.keys())
    
    def get_available_scales(self) -> List[str]:
        """Get list of available scale types."""
        return list(self._domain_defaults["general"].keys())
    
    def get_domain_config(self, domain: str) -> Optional[Dict[str, float]]:
        """
        Get configuration for a specific domain.
        
        Parameters:
        -----------
        domain : str
            Domain name
            
        Returns:
        --------
        Dict[str, float] or None : Domain configuration
        """
        return self._domain_defaults.get(domain)
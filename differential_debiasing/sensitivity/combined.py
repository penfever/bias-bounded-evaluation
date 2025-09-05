"""
Combined sensitivity estimation using both psychometric reliability and schematic adherence

This module implements the combined sensitivity approach as formalized in the 
bias-bounded LLM judges paper, combining:
- Psychometric Reliability Sensitivity (Cronbach's α + CLR + HTMT)
- Schematic Adherence Sensitivity (linear/polynomial regression analysis)

The combined approach provides:
Δ_B f ≈ γ * sqrt(α * S_PR² + (1-α) * S_SA²) * Range(J)
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List
from .base import SensitivityEstimator
from .psychometric_reliability import PsychometricReliabilitySensitivity
from .schematic_adherence import SchematicAdherenceSensitivity
import warnings


class CombinedSensitivity(SensitivityEstimator):
    """
    Combined sensitivity estimator using both psychometric reliability and schematic adherence.
    
    This estimator implements the combined approach from the bias-bounded LLM judges paper:
    
    Δ_B f = γ * sqrt(α * (Psychometric Reliability Sensitivity)² + 
                     (1-α) * (Schematic Adherence Sensitivity)²) * Range(J)
    
    where:
    - γ is a calibration constant
    - α weights the relative importance of the two measures
    - Range(J) is the judgment score range
    """
    
    def __init__(self,
                 factor_columns: Optional[List[str]] = None,
                 target_column: str = 'overall_score',
                 alpha: float = 0.5,
                 gamma: float = 1.0,
                 psychometric_kwargs: Optional[Dict] = None,
                 schematic_kwargs: Optional[Dict] = None,
                 **kwargs):
        """
        Initialize combined sensitivity estimator.
        
        Parameters:
        -----------
        factor_columns : List[str], optional
            Column names for factor scores. If None, will auto-detect.
        target_column : str
            Column name for overall/target score
        alpha : float
            Weight parameter balancing psychometric vs schematic measures (0 ≤ α ≤ 1)
            α = 1.0: Pure psychometric reliability
            α = 0.0: Pure schematic adherence
            α = 0.5: Equal weighting (default)
        gamma : float
            Calibration constant for scaling sensitivity estimate
        psychometric_kwargs : Dict, optional
            Additional arguments for PsychometricReliabilitySensitivity
        schematic_kwargs : Dict, optional
            Additional arguments for SchematicAdherenceSensitivity
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        
        self.factor_columns = factor_columns
        self.target_column = target_column
        self.alpha = alpha
        self.gamma = gamma
        
        # Initialize component estimators
        psychometric_kwargs = psychometric_kwargs or {}
        schematic_kwargs = schematic_kwargs or {}
        
        # Pass shared parameters to component estimators
        psychometric_kwargs.update({
            'factor_columns': factor_columns,
        })
        schematic_kwargs.update({
            'factor_columns': factor_columns,
            'target_column': target_column
        })
        
        self.psychometric_estimator = PsychometricReliabilitySensitivity(**psychometric_kwargs)
        self.schematic_estimator = SchematicAdherenceSensitivity(**schematic_kwargs)
        
        # Results storage
        self._psychometric_sensitivity = None
        self._schematic_sensitivity = None
        self._combined_sensitivity = None
        self._component_diagnostics = {}
        
    def fit(self, judgments: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'CombinedSensitivity':
        """
        Fit both psychometric reliability and schematic adherence analyses.
        
        Parameters:
        -----------
        judgments : DataFrame or array-like
            Judgment data. Must be DataFrame with factor and target columns.
        **kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        CombinedSensitivity : Self for method chaining
        """
        judgments = self._validate_input(judgments)
        
        if not isinstance(judgments, pd.DataFrame):
            raise ValueError("CombinedSensitivity requires DataFrame input with factor and target columns")
        
        # Fit component estimators
        try:
            self.psychometric_estimator.fit(judgments, **kwargs)
            self._component_diagnostics['psychometric_fitted'] = True
        except Exception as e:
            warnings.warn(f"Psychometric reliability analysis failed: {e}")
            self._component_diagnostics['psychometric_fitted'] = False
            self._component_diagnostics['psychometric_error'] = str(e)
        
        try:
            self.schematic_estimator.fit(judgments, **kwargs)
            self._component_diagnostics['schematic_fitted'] = True
        except Exception as e:
            warnings.warn(f"Schematic adherence analysis failed: {e}")
            self._component_diagnostics['schematic_fitted'] = False
            self._component_diagnostics['schematic_error'] = str(e)
        
        # Check that at least one component fitted successfully
        if not (self._component_diagnostics.get('psychometric_fitted', False) or 
                self._component_diagnostics.get('schematic_fitted', False)):
            raise ValueError("Both component analyses failed. Check input data and parameters.")
        
        self._fitted = True
        return self
    
    def estimate(self, score_range: Optional[float] = None) -> float:
        """
        Estimate combined bias sensitivity.
        
        Parameters:
        -----------
        score_range : float, optional
            Range of judgment scores. If None, assumes normalized [0,1] range.
            
        Returns:
        --------
        float : Combined bias sensitivity estimate
        """
        self._validate_fitted()
        
        # Get component sensitivities
        psychometric_fitted = self._component_diagnostics.get('psychometric_fitted', False)
        schematic_fitted = self._component_diagnostics.get('schematic_fitted', False)
        
        # Calculate psychometric sensitivity
        if psychometric_fitted:
            try:
                self._psychometric_sensitivity = self.psychometric_estimator.estimate(score_range)
                # Convert to normalized sensitivity (remove score_range scaling for combination)
                psychometric_normalized = self._psychometric_sensitivity / (score_range or 1.0)
            except Exception as e:
                warnings.warn(f"Psychometric sensitivity estimation failed: {e}")
                psychometric_normalized = 0.5  # Fallback moderate sensitivity
                self._psychometric_sensitivity = psychometric_normalized * (score_range or 1.0)
        else:
            psychometric_normalized = 0.5  # Default moderate sensitivity if not fitted
            self._psychometric_sensitivity = psychometric_normalized * (score_range or 1.0)
        
        # Calculate schematic sensitivity
        if schematic_fitted:
            try:
                self._schematic_sensitivity = self.schematic_estimator.estimate(score_range)
                # Convert to normalized sensitivity
                schematic_normalized = self._schematic_sensitivity / (score_range or 1.0)
            except Exception as e:
                warnings.warn(f"Schematic sensitivity estimation failed: {e}")
                schematic_normalized = 0.5  # Fallback moderate sensitivity
                self._schematic_sensitivity = schematic_normalized * (score_range or 1.0)
        else:
            schematic_normalized = 0.5  # Default moderate sensitivity if not fitted
            self._schematic_sensitivity = schematic_normalized * (score_range or 1.0)
        
        # Combine sensitivities using the formula from the paper:
        # Δ_B f = γ * sqrt(α * S_PR² + (1-α) * S_SA²) * Range(J)
        combined_normalized = np.sqrt(
            self.alpha * (psychometric_normalized ** 2) + 
            (1 - self.alpha) * (schematic_normalized ** 2)
        )
        
        # Apply calibration and scale to judgment units
        if score_range is None:
            score_range = 1.0  # Assume normalized
        
        self._combined_sensitivity = self.gamma * combined_normalized * score_range
        self._bias_sensitivity = self._combined_sensitivity
        
        return self._combined_sensitivity
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostic information from both component analyses.
        
        Returns:
        --------
        Dict[str, Any] : Combined diagnostic information
        """
        diagnostics = super().get_diagnostics()
        
        if self._fitted:
            diagnostics.update({
                "method_details": "Combined Psychometric Reliability + Schematic Adherence",
                "combination_parameters": {
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "psychometric_weight": self.alpha,
                    "schematic_weight": 1 - self.alpha
                },
                "component_sensitivities": {
                    "psychometric_sensitivity": self._psychometric_sensitivity,
                    "schematic_sensitivity": self._schematic_sensitivity,
                    "combined_sensitivity": self._combined_sensitivity
                },
                "component_status": self._component_diagnostics.copy(),
                "quality_indicators": self._get_quality_indicators()
            })
            
            # Add component diagnostics
            if self._component_diagnostics.get('psychometric_fitted', False):
                try:
                    psychometric_diag = self.psychometric_estimator.get_diagnostics()
                    diagnostics["psychometric_analysis"] = psychometric_diag
                except:
                    diagnostics["psychometric_analysis"] = {"error": "Failed to get diagnostics"}
            
            if self._component_diagnostics.get('schematic_fitted', False):
                try:
                    schematic_diag = self.schematic_estimator.get_diagnostics()
                    diagnostics["schematic_analysis"] = schematic_diag
                except:
                    diagnostics["schematic_analysis"] = {"error": "Failed to get diagnostics"}
        
        return diagnostics
    
    def _get_quality_indicators(self) -> Dict[str, Any]:
        """Get quality indicators for the combined analysis."""
        if not self._fitted:
            return {}
        
        indicators = {}
        
        # Overall quality assessment
        if self._combined_sensitivity is not None:
            if self._combined_sensitivity < 0.3:
                indicators["overall_quality"] = "excellent"
            elif self._combined_sensitivity < 0.5:
                indicators["overall_quality"] = "good"
            elif self._combined_sensitivity < 0.7:
                indicators["overall_quality"] = "moderate"
            else:
                indicators["overall_quality"] = "poor"
        
        # Component reliability
        psychometric_fitted = self._component_diagnostics.get('psychometric_fitted', False)
        schematic_fitted = self._component_diagnostics.get('schematic_fitted', False)
        
        if psychometric_fitted and schematic_fitted:
            indicators["analysis_completeness"] = "full"
        elif psychometric_fitted or schematic_fitted:
            indicators["analysis_completeness"] = "partial"
        else:
            indicators["analysis_completeness"] = "failed"
        
        # Sensitivity agreement
        if self._psychometric_sensitivity and self._schematic_sensitivity:
            sensitivity_ratio = min(self._psychometric_sensitivity, self._schematic_sensitivity) / \
                               max(self._psychometric_sensitivity, self._schematic_sensitivity)
            
            if sensitivity_ratio > 0.8:
                indicators["component_agreement"] = "high"
            elif sensitivity_ratio > 0.5:
                indicators["component_agreement"] = "moderate"
            else:
                indicators["component_agreement"] = "low"
        
        # Weighting appropriateness
        if self.alpha == 0.5:
            indicators["weighting_strategy"] = "balanced"
        elif self.alpha > 0.7:
            indicators["weighting_strategy"] = "psychometric_focused"
        elif self.alpha < 0.3:
            indicators["weighting_strategy"] = "schematic_focused"
        else:
            indicators["weighting_strategy"] = "custom"
        
        return indicators
    
    def get_psychometric_estimator(self) -> PsychometricReliabilitySensitivity:
        """Get the psychometric reliability estimator."""
        return self.psychometric_estimator
    
    def get_schematic_estimator(self) -> SchematicAdherenceSensitivity:
        """Get the schematic adherence estimator."""
        return self.schematic_estimator
    
    def get_component_sensitivities(self) -> Dict[str, Optional[float]]:
        """Get individual component sensitivity estimates."""
        return {
            "psychometric_sensitivity": self._psychometric_sensitivity,
            "schematic_sensitivity": self._schematic_sensitivity,
            "combined_sensitivity": self._combined_sensitivity
        }
    
    def get_alpha(self) -> float:
        """Get the alpha weighting parameter."""
        return self.alpha
    
    def get_gamma(self) -> float:
        """Get the gamma calibration parameter."""
        return self.gamma
    
    def set_alpha(self, alpha: float) -> 'CombinedSensitivity':
        """
        Update the alpha weighting parameter and re-estimate if fitted.
        
        Parameters:
        -----------
        alpha : float
            New alpha value (0 ≤ α ≤ 1)
            
        Returns:
        --------
        CombinedSensitivity : Self for method chaining
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        
        self.alpha = alpha
        
        # Re-estimate if already fitted
        if self._fitted and (self._psychometric_sensitivity is not None or 
                           self._schematic_sensitivity is not None):
            # Re-calculate combined sensitivity with new alpha
            score_range = 1.0  # Will be overridden if we can infer from existing estimates
            
            if self._psychometric_sensitivity and self._schematic_sensitivity:
                # Infer score range from existing estimates (approximate)
                score_range = max(self._psychometric_sensitivity, self._schematic_sensitivity) / 0.5
            
            self.estimate(score_range)
        
        return self
    
    def set_gamma(self, gamma: float) -> 'CombinedSensitivity':
        """
        Update the gamma calibration parameter and re-estimate if fitted.
        
        Parameters:
        -----------
        gamma : float
            New gamma value (must be positive)
            
        Returns:
        --------
        CombinedSensitivity : Self for method chaining
        """
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        
        self.gamma = gamma
        
        # Re-estimate if already fitted
        if self._fitted and self._combined_sensitivity is not None:
            # Adjust existing estimate
            old_gamma = 1.0  # Assume previous gamma was 1.0 if not stored
            self._combined_sensitivity = (self._combined_sensitivity / old_gamma) * gamma
            self._bias_sensitivity = self._combined_sensitivity
        
        return self
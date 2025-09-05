"""
Profile-enhanced Combined A-BB sensitivity that uses pre-computed sensitivity profiles.

This module provides a wrapper around CombinedABBSensitivity that can use 
pre-computed formatting sensitivity values instead of expensive dynamic measurement.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List, Callable

from .combined_abb_sensitivity import CombinedABBSensitivity
from ..core.sensitivity_profiles import SensitivityProfile, get_formatting_sensitivity


class ProfileEnhancedABBSensitivity(CombinedABBSensitivity):
    """
    Combined A-BB sensitivity estimator that can use pre-computed sensitivity profiles
    to avoid expensive dynamic measurements where possible.
    """
    
    def __init__(self, 
                 static_estimators: Optional[List[Union[str, Any]]] = None,
                 dynamic_generators: Optional[List[Union[str, Any]]] = None,
                 combination_strategy: str = "conservative",
                 static_weights: Optional[List[float]] = None,
                 dynamic_weights: Optional[List[float]] = None,
                 judge_function: Optional[Callable] = None,
                 num_neighbors: int = 10,
                 sensitivity_profile: Optional[Union[SensitivityProfile, str, Dict[str, float]]] = None,
                 profile_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize profile-enhanced A-BB sensitivity estimator.
        
        Parameters:
        -----------
        sensitivity_profile : SensitivityProfile, str, or Dict[str, float], optional
            Pre-computed sensitivity values to use for expensive measurements
        profile_dir : str, optional
            Directory containing sensitivity profiles
        ... (other parameters same as CombinedABBSensitivity)
        """
        super().__init__(
            static_estimators=static_estimators,
            dynamic_generators=dynamic_generators,
            combination_strategy=combination_strategy,
            static_weights=static_weights,
            dynamic_weights=dynamic_weights,
            judge_function=judge_function,
            num_neighbors=num_neighbors,
            **kwargs
        )
        
        # Load sensitivity profile
        self.sensitivity_profile = self._load_profile(sensitivity_profile, profile_dir)
        self._profile_values = {}  # Cache for profile-based sensitivity values
    
    def _load_profile(self, 
                     profile: Optional[Union[SensitivityProfile, str, Dict[str, float]]],
                     profile_dir: Optional[str]) -> Optional[SensitivityProfile]:
        """Load sensitivity profile from various input formats."""
        if profile is None:
            return None
        
        if isinstance(profile, SensitivityProfile):
            return profile
        
        if isinstance(profile, str):
            # Load by judge name
            from ..core.sensitivity_profiles import load_judge_sensitivity_profile
            return load_judge_sensitivity_profile(profile, profile_dir or "sensitivity_profiles")
        
        if isinstance(profile, dict):
            # Create profile from dict
            profile_data = {
                'judge_name': 'custom',
                'profile_version': '1.0',
                'created_date': None,
                'formatting_sensitivity': profile.copy(),
                'measurement_metadata': {'source': 'manual_dict'}
            }
            return SensitivityProfile(profile_data)
        
        return None
    
    def _fit_dynamic_estimators(self, context: Union[Dict, pd.DataFrame], **kwargs):
        """
        Fit dynamic estimators, using profile values where available.
        """
        self._dynamic_sensitivities = {}
        
        for name, generator in self.dynamic_generators.items():
            try:
                # Check if we have a profile value for this type of sensitivity
                profile_value = self._get_profile_value_for_generator(name)
                
                if profile_value is not None:
                    # Use profile value instead of expensive measurement
                    print(f"🚀 Using profile-based {name} sensitivity: {profile_value:.4f}")
                    
                    # Create a mock estimator that returns the profile value
                    mock_estimator = MockProfileEstimator(profile_value)
                    self._dynamic_sensitivities[name] = mock_estimator
                    
                else:
                    # Fall back to normal dynamic measurement
                    print(f"📏 Measuring {name} sensitivity dynamically (no profile available)")
                    super()._fit_dynamic_estimators(context, **kwargs)
                    return  # Let parent handle all measurements
                    
            except Exception as e:
                print(f"Warning: Dynamic generator '{name}' failed: {e}")
                self._dynamic_sensitivities[name] = None
    
    def _get_profile_value_for_generator(self, generator_name: str) -> Optional[float]:
        """Get profile-based sensitivity value for a specific generator type."""
        if self.sensitivity_profile is None:
            return None
        
        # Map generator names to profile methods
        if generator_name in ['formatting', 'FormattingNeighborGenerator']:
            return self.sensitivity_profile.get_formatting_sensitivity()
        
        # Future: Add other mappings (hamming, order, etc.)
        return None


class MockProfileEstimator:
    """
    Mock estimator that returns a pre-computed profile value.
    
    This is used to inject profile-based sensitivity values into the
    Combined A-BB framework without needing expensive dynamic measurement.
    """
    
    def __init__(self, sensitivity_value: float):
        """
        Initialize mock estimator.
        
        Parameters:
        -----------
        sensitivity_value : float
            Pre-computed sensitivity value to return
        """
        self.sensitivity_value = float(sensitivity_value)
        self._fitted = True
    
    def estimate(self, score_range: Optional[float] = None) -> float:
        """Return the pre-computed sensitivity value."""
        return self.sensitivity_value
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics indicating this is a profile-based value."""
        return {
            "estimator_type": "ProfileBased",
            "sensitivity_value": self.sensitivity_value,
            "source": "sensitivity_profile",
            "is_dynamic_measurement": False
        }
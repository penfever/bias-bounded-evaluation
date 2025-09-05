"""
Sensitivity profile management utilities.

This module provides functionality for loading, saving, validating, and managing
pre-computed sensitivity profiles for judges. This enables efficient separation
of sensitivity measurement from bias-bounded mechanism application.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timezone
import warnings


class SensitivityProfile:
    """
    Represents a judge's sensitivity profile with measurements and metadata.
    """
    
    def __init__(self, profile_data: Dict[str, Any]):
        """
        Initialize sensitivity profile from data dictionary.
        
        Parameters:
        -----------
        profile_data : Dict[str, Any]
            Profile data loaded from JSON or created programmatically
        """
        self.data = profile_data.copy()
        self.judge_name = profile_data.get('judge_name', 'unknown')
        self.profile_version = profile_data.get('profile_version', '1.0')
        self.created_date = profile_data.get('created_date')
        
        # Extract sensitivity measurements
        self.formatting_sensitivity = profile_data.get('formatting_sensitivity', {})
        
        # Validation
        self._validate()
    
    def _validate(self) -> None:
        """Validate profile data integrity."""
        if not self.judge_name or self.judge_name == 'unknown':
            warnings.warn("Profile has no judge_name or uses default 'unknown'")
        
        if not self.created_date:
            warnings.warn("Profile has no creation date")
        
        # Validate formatting sensitivity
        if 'error' not in self.formatting_sensitivity:
            if 'value' not in self.formatting_sensitivity:
                raise ValueError("Formatting sensitivity missing required 'value' field")
            
            value = self.formatting_sensitivity['value']
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Invalid formatting sensitivity value: {value}")
    
    def get_formatting_sensitivity(self) -> Optional[float]:
        """
        Get formatting sensitivity value.
        
        Returns:
        --------
        Optional[float] : Formatting sensitivity value, or None if measurement failed
        """
        if 'error' in self.formatting_sensitivity:
            return None
        return float(self.formatting_sensitivity.get('value', 0.0))
    
    def get_formatting_confidence_interval(self) -> Optional[Tuple[float, float]]:
        """
        Get confidence interval for formatting sensitivity.
        
        Returns:
        --------
        Optional[Tuple[float, float]] : (lower, upper) confidence bounds, or None if unavailable
        """
        if 'error' in self.formatting_sensitivity:
            return None
        
        ci = self.formatting_sensitivity.get('confidence_interval')
        if ci and len(ci) == 2:
            return (float(ci[0]), float(ci[1]))
        return None
    
    def is_fresh(self, max_age_days: int = 30) -> bool:
        """
        Check if profile is fresh (recently created).
        
        Parameters:
        -----------
        max_age_days : int
            Maximum age in days to consider fresh
            
        Returns:
        --------
        bool : True if profile is fresh, False otherwise
        """
        if not self.created_date:
            return False
        
        try:
            created = datetime.fromisoformat(self.created_date.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            age_days = (now - created).days
            return age_days <= max_age_days
        except:
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get profile metadata."""
        return self.data.get('measurement_metadata', {})
    
    def get_cost_info(self) -> Dict[str, Any]:
        """Get cost information from measurements."""
        return self.formatting_sensitivity.get('cost_info', {})
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the profile."""
        formatting_value = self.get_formatting_sensitivity()
        formatting_ci = self.get_formatting_confidence_interval()
        
        return {
            'judge_name': self.judge_name,
            'created_date': self.created_date,
            'is_fresh': self.is_fresh(),
            'formatting_sensitivity': {
                'value': formatting_value,
                'confidence_interval': formatting_ci,
                'has_error': 'error' in self.formatting_sensitivity
            },
            'samples_used': self.formatting_sensitivity.get('samples_used'),
            'cost_info': self.get_cost_info()
        }


class SensitivityProfileManager:
    """
    Manages loading, saving, and validation of sensitivity profiles.
    """
    
    def __init__(self, profile_dir: Union[str, Path] = "sensitivity_profiles"):
        """
        Initialize profile manager.
        
        Parameters:
        -----------
        profile_dir : str or Path
            Directory containing sensitivity profile files
        """
        self.profile_dir = Path(profile_dir)
        self._profile_cache = {}  # Simple in-memory cache
    
    def load_profile(self, judge_name: str, use_cache: bool = True) -> Optional[SensitivityProfile]:
        """
        Load sensitivity profile for a judge.
        
        Parameters:
        -----------
        judge_name : str
            Name of the judge
        use_cache : bool
            Whether to use cached profiles
            
        Returns:
        --------
        Optional[SensitivityProfile] : Loaded profile, or None if not found
        """
        # Check cache first
        if use_cache and judge_name in self._profile_cache:
            return self._profile_cache[judge_name]
        
        profile_file = self.profile_dir / f"{judge_name}_sensitivity_profile.json"
        
        if not profile_file.exists():
            return None
        
        try:
            with open(profile_file, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
            
            profile = SensitivityProfile(profile_data)
            
            # Cache the profile
            if use_cache:
                self._profile_cache[judge_name] = profile
            
            return profile
            
        except Exception as e:
            warnings.warn(f"Failed to load sensitivity profile for {judge_name}: {e}")
            return None
    
    def save_profile(self, profile: SensitivityProfile, overwrite: bool = False) -> Path:
        """
        Save sensitivity profile to disk.
        
        Parameters:
        -----------
        profile : SensitivityProfile
            Profile to save
        overwrite : bool
            Whether to overwrite existing profile
            
        Returns:
        --------
        Path : Path to saved profile file
        """
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
        profile_file = self.profile_dir / f"{profile.judge_name}_sensitivity_profile.json"
        
        if profile_file.exists() and not overwrite:
            raise FileExistsError(f"Profile already exists: {profile_file}")
        
        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(profile.data, f, indent=2, ensure_ascii=False)
        
        # Update cache
        self._profile_cache[profile.judge_name] = profile
        
        return profile_file
    
    def list_profiles(self) -> List[str]:
        """
        List available judge profiles.
        
        Returns:
        --------
        List[str] : List of judge names with available profiles
        """
        if not self.profile_dir.exists():
            return []
        
        profiles = []
        for profile_file in self.profile_dir.glob("*_sensitivity_profile.json"):
            judge_name = profile_file.stem.replace('_sensitivity_profile', '')
            profiles.append(judge_name)
        
        return sorted(profiles)
    
    def validate_profile(self, judge_name: str) -> Dict[str, Any]:
        """
        Validate a sensitivity profile.
        
        Parameters:
        -----------
        judge_name : str
            Name of the judge
            
        Returns:
        --------
        Dict[str, Any] : Validation results
        """
        profile = self.load_profile(judge_name, use_cache=False)
        
        if profile is None:
            return {
                'valid': False,
                'error': 'Profile not found',
                'judge_name': judge_name
            }
        
        try:
            # Basic validation
            profile._validate()
            
            # Additional checks
            issues = []
            warnings_list = []
            
            # Check freshness
            if not profile.is_fresh(max_age_days=90):  # 3 months
                warnings_list.append("Profile is older than 90 days")
            
            # Check formatting sensitivity
            formatting_value = profile.get_formatting_sensitivity()
            if formatting_value is None:
                issues.append("Formatting sensitivity measurement failed")
            elif formatting_value <= 0:
                issues.append("Formatting sensitivity is zero or negative")
            elif formatting_value > 50:  # Arbitrary high threshold
                warnings_list.append(f"Formatting sensitivity is very high ({formatting_value:.2f})")
            
            # Check confidence interval
            ci = profile.get_formatting_confidence_interval()
            if ci and (ci[1] - ci[0]) > formatting_value:  # Wide CI
                warnings_list.append("Confidence interval is very wide")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings_list,
                'judge_name': judge_name,
                'summary': profile.summary()
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'judge_name': judge_name
            }
    
    def get_sensitivity_values(self, judge_name: str) -> Dict[str, Optional[float]]:
        """
        Get sensitivity values for a judge, with fallbacks.
        
        Parameters:
        -----------
        judge_name : str
            Name of the judge
            
        Returns:
        --------
        Dict[str, Optional[float]] : Dictionary with sensitivity values
        """
        profile = self.load_profile(judge_name)
        
        result = {
            'formatting_sensitivity': None,
            'profile_available': profile is not None
        }
        
        if profile:
            result['formatting_sensitivity'] = profile.get_formatting_sensitivity()
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the profile cache."""
        self._profile_cache.clear()


# Convenience functions for common operations

def load_judge_sensitivity_profile(judge_name: str, 
                                 profile_dir: Union[str, Path] = "sensitivity_profiles") -> Optional[SensitivityProfile]:
    """
    Convenience function to load a judge's sensitivity profile.
    
    Parameters:
    -----------
    judge_name : str
        Name of the judge
    profile_dir : str or Path
        Directory containing profiles
        
    Returns:
    --------
    Optional[SensitivityProfile] : Loaded profile or None if not found
    """
    manager = SensitivityProfileManager(profile_dir)
    return manager.load_profile(judge_name)


def get_formatting_sensitivity(judge_name: str,
                             profile_dir: Union[str, Path] = "sensitivity_profiles",
                             fallback_value: Optional[float] = None) -> Optional[float]:
    """
    Get formatting sensitivity for a judge with fallback.
    
    Parameters:
    -----------
    judge_name : str
        Name of the judge
    profile_dir : str or Path
        Directory containing profiles
    fallback_value : Optional[float]
        Fallback value if profile not found
        
    Returns:
    --------
    Optional[float] : Formatting sensitivity value
    """
    profile = load_judge_sensitivity_profile(judge_name, profile_dir)
    
    if profile:
        value = profile.get_formatting_sensitivity()
        if value is not None:
            return value
    
    return fallback_value


def validate_all_profiles(profile_dir: Union[str, Path] = "sensitivity_profiles") -> Dict[str, Dict[str, Any]]:
    """
    Validate all profiles in a directory.
    
    Parameters:
    -----------
    profile_dir : str or Path
        Directory containing profiles
        
    Returns:
    --------
    Dict[str, Dict[str, Any]] : Validation results for each judge
    """
    manager = SensitivityProfileManager(profile_dir)
    results = {}
    
    for judge_name in manager.list_profiles():
        results[judge_name] = manager.validate_profile(judge_name)
    
    return results
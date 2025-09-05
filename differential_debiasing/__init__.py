"""
Differential Debiasing for LLM Judges

A Python library for reducing implicit bias in LLM judge evaluations using 
differential privacy-inspired techniques while providing formal guarantees.

Main Components:
- DifferentialDebias: Core debiasing class
- ConfigManager: Configuration management
- Sensitivity estimators: Multiple bias detection methods
- Judge interfaces: Integration with various LLM judge systems
"""

# Core imports
from .core.debias import DifferentialDebias
from .core.config import DebiasConfig, ConfigManager

# Sensitivity estimators
from .sensitivity import (
    SensitivityEstimator,
    FactorAnalysisSensitivity,
    EmpiricalSensitivity, 
    DomainSpecificSensitivity,
    CombinedSensitivity,
    ABBSensitivity,
    CombinedABBSensitivity,
    ProfileEnhancedABBSensitivity,
    ConformalSensitivityEstimator,
    ConformedBiasBoundedPredictor,
    PsychometricReliabilitySensitivity,
    SchematicAdherenceSensitivity
)

# Judge interfaces
from .interfaces import (
    create_oumi_judge_function,
    create_judge_function
)

__version__ = "0.1.0"
__author__ = "Benjamin Feuer"
__email__ = "penfever@gmail.com"

__all__ = [
    # Core classes
    "DifferentialDebias",
    "DebiasConfig",
    "ConfigManager",
    
    # Sensitivity estimators
    "SensitivityEstimator",
    "FactorAnalysisSensitivity",
    "EmpiricalSensitivity", 
    "DomainSpecificSensitivity",
    "CombinedSensitivity",
    "ABBSensitivity", 
    "CombinedABBSensitivity",
    "ProfileEnhancedABBSensitivity",
    "ConformalSensitivityEstimator",
    "ConformedBiasBoundedPredictor",
    "PsychometricReliabilitySensitivity",
    "SchematicAdherenceSensitivity",
    
    # Judge interfaces
    "create_oumi_judge_function",
    "create_judge_function",
]

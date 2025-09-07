"""
Fixed sensitivity estimator that uses an externally provided sensitivity value.

This estimator is agnostic to how the sensitivity was obtained (profiles, offline
analysis, etc.). It simply returns the provided value and participates in the
same interface as other estimators so the debiaser can compute noise and bounds.
"""

from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd

from .base import SensitivityEstimator
from ..core.utils import compute_abb_constraint_validation


class FixedSensitivityEstimator(SensitivityEstimator):
    """
    Sensitivity estimator that returns a fixed, externally provided value.
    """

    def __init__(self, fixed_sensitivity_value: float, **kwargs):
        super().__init__(**kwargs)
        self.fixed_value = float(fixed_sensitivity_value)

    def fit(self, judgments: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'FixedSensitivityEstimator':
        # No-op fit; value is externally provided
        self._bias_sensitivity = self.fixed_value
        self._fitted = True
        return self

    def estimate(self, score_range: Optional[float] = None) -> float:
        self._validate_fitted()
        # Value is already in judgment units; ignore score_range
        return float(self._bias_sensitivity)

    def get_diagnostics(self) -> Dict[str, Any]:
        d = super().get_diagnostics()
        d.update({
            'estimator_type': 'Fixed',
            'source': 'external',
        })
        return d

    # Provide same interface piece used by DifferentialDebias diagnostics for ABB constraint
    def validate_abb_constraint(self, tau: float, delta: float) -> Dict[str, Any]:
        """Validate A-BB constraint using the fixed sensitivity value (raw units)."""
        self._validate_fitted()
        return compute_abb_constraint_validation(
            tau=float(tau), delta=float(delta), sensitivity=float(self._bias_sensitivity)
        )

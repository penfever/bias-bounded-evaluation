"""
Unit tests for the psychometric reliability sensitivity estimator using synthetic data.
"""

import numpy as np

from ..test_data_helpers import load_and_prepare_score_data
from differential_debiasing.sensitivity.psychometric_reliability import (
    PsychometricReliabilitySensitivity,
)


def test_psychometric_reliability_workflow():
    """Psychometric estimator should fit and produce sensible diagnostics."""
    df = load_and_prepare_score_data(n_samples=150, random_seed=2024)

    estimator = PsychometricReliabilitySensitivity()
    estimator.fit(df)

    reliability = estimator.get_reliability_score()
    diagnostics = estimator.get_diagnostics()

    assert 0.0 <= reliability <= 1.0
    assert diagnostics["reliability_score"] == reliability
    assert diagnostics["cronbach_alphas"]
    assert diagnostics["clr_scores"]

    sensitivity = estimator.estimate(score_range=4.0)
    assert isinstance(sensitivity, float)
    assert sensitivity >= 0.0

    # Sanity check: higher reliability should imply lower sensitivity
    normalized = np.sqrt(1.0 - reliability)
    assert np.isclose(sensitivity, normalized * 4.0)

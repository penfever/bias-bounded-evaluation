"""
Tests for utility functions
"""

import pytest
import numpy as np
import pandas as pd
from differential_debiasing.core.utils import (
    normalize_judgments, denormalize_judgments, calculate_noise_parameter,
    validate_input_array, clip_to_range, check_bias_parameters,
    estimate_r_squared_from_scores
)


class TestNormalizationFunctions:
    """Test normalization and denormalization functions."""
    
    def test_normalize_judgments_default_range(self):
        """Test normalization with default range."""
        judgments = np.array([1, 3, 5, 7, 10])
        normalized, min_val, max_val = normalize_judgments(judgments)
        
        assert min_val == 1
        assert max_val == 10
        assert normalized.min() == 0
        assert normalized.max() == 1
        np.testing.assert_array_almost_equal(normalized, [0, 2/9, 4/9, 6/9, 1])
    
    def test_normalize_judgments_custom_range(self):
        """Test normalization with custom range."""
        judgments = np.array([2, 4, 6, 8])
        normalized, min_val, max_val = normalize_judgments(judgments, score_min=0, score_max=10)
        
        assert min_val == 0
        assert max_val == 10
        np.testing.assert_array_almost_equal(normalized, [0.2, 0.4, 0.6, 0.8])
    
    def test_normalize_judgments_invalid_range(self):
        """Test normalization with invalid range."""
        judgments = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="score_max .* must be greater than score_min"):
            normalize_judgments(judgments, score_min=5, score_max=5)
        
        with pytest.raises(ValueError, match="score_max .* must be greater than score_min"):
            normalize_judgments(judgments, score_min=5, score_max=3)
    
    def test_denormalize_judgments(self):
        """Test denormalization."""
        normalized = np.array([0, 0.25, 0.5, 0.75, 1])
        denormalized = denormalize_judgments(normalized, score_min=1, score_max=10)
        
        expected = np.array([1, 3.25, 5.5, 7.75, 10])
        np.testing.assert_array_almost_equal(denormalized, expected)
    
    def test_normalize_denormalize_roundtrip(self):
        """Test that normalize -> denormalize is identity."""
        original = np.array([1.5, 3.7, 6.2, 8.9, 9.1])
        normalized, min_val, max_val = normalize_judgments(original)
        restored = denormalize_judgments(normalized, min_val, max_val)
        
        np.testing.assert_array_almost_equal(original, restored)
    
    def test_normalize_single_value(self):
        """Test normalization with single value."""
        judgments = np.array([5])
        normalized, min_val, max_val = normalize_judgments(judgments)
        
        assert min_val == 5
        assert max_val == 5
        assert normalized[0] == 0  # Single value normalizes to 0
    
    def test_normalize_empty_array(self):
        """Test normalization with empty array."""
        judgments = np.array([])
        
        # Should handle empty array gracefully
        normalized, min_val, max_val = normalize_judgments(judgments)
        assert len(normalized) == 0
        assert np.isnan(min_val) or np.isinf(min_val)
        assert np.isnan(max_val) or np.isinf(max_val)


class TestNoiseCalculation:
    """Test noise parameter calculation."""
    
    def test_calculate_noise_parameter_basic(self):
        """Test basic noise parameter calculation."""
        sigma = calculate_noise_parameter(
            bias_sensitivity=2.0,
            tau=0.5,
            delta=0.05,
            n_samples=100,
            use_average_case=True
        )
        
        assert sigma > 0
        assert isinstance(sigma, float)
    
    def test_calculate_noise_parameter_individual_case(self):
        """Test noise parameter for individual case."""
        sigma_individual = calculate_noise_parameter(
            bias_sensitivity=2.0,
            tau=0.5,
            delta=0.05,
            n_samples=100,
            use_average_case=False
        )
        
        sigma_average = calculate_noise_parameter(
            bias_sensitivity=2.0,
            tau=0.5,
            delta=0.05,
            n_samples=100,
            use_average_case=True
        )
        
        # Average case should have less noise
        assert sigma_individual > sigma_average
        assert sigma_individual == pytest.approx(sigma_average * np.sqrt(100), rel=1e-3)
    
    def test_calculate_noise_parameter_scaling(self):
        """Test noise parameter scaling with different parameters."""
        base_sigma = calculate_noise_parameter(
            bias_sensitivity=1.0,
            tau=0.5,
            delta=0.05,
            n_samples=100,
            use_average_case=True
        )
        
        # Higher sensitivity -> more noise
        higher_sens_sigma = calculate_noise_parameter(
            bias_sensitivity=2.0,
            tau=0.5,
            delta=0.05,
            n_samples=100,
            use_average_case=True
        )
        assert higher_sens_sigma > base_sigma
        
        # Lower tau -> more noise (stronger protection)
        lower_tau_sigma = calculate_noise_parameter(
            bias_sensitivity=1.0,
            tau=0.25,
            delta=0.05,
            n_samples=100,
            use_average_case=True
        )
        assert lower_tau_sigma > base_sigma
        
        # Lower delta -> more noise (higher confidence)
        lower_delta_sigma = calculate_noise_parameter(
            bias_sensitivity=1.0,
            tau=0.5,
            delta=0.01,
            n_samples=100,
            use_average_case=True
        )
        assert lower_delta_sigma > base_sigma
    
    def test_calculate_noise_parameter_invalid_inputs(self):
        """Test noise parameter calculation with invalid inputs."""
        with pytest.raises(ValueError, match="bias_sensitivity must be positive"):
            calculate_noise_parameter(0, 0.5, 0.05, 100)
        
        with pytest.raises(ValueError, match="tau must be positive"):
            calculate_noise_parameter(1.0, 0, 0.05, 100)
        
        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            calculate_noise_parameter(1.0, 0.5, 0, 100)
        
        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            calculate_noise_parameter(1.0, 0.5, 1, 100)
        
        with pytest.raises(ValueError, match="n_samples must be positive"):
            calculate_noise_parameter(1.0, 0.5, 0.05, 0)


class TestValidationFunctions:
    """Test validation functions."""
    
    def test_validate_input_array_numpy(self):
        """Test validation with numpy array."""
        arr = np.array([1, 2, 3, 4, 5])
        validated = validate_input_array(arr)
        
        np.testing.assert_array_equal(validated, arr)
    
    def test_validate_input_array_list(self):
        """Test validation with list."""
        arr = [1, 2, 3, 4, 5]
        validated = validate_input_array(arr)
        
        expected = np.array([1, 2, 3, 4, 5])
        np.testing.assert_array_equal(validated, expected)
    
    def test_validate_input_array_pandas_series(self):
        """Test validation with pandas Series."""
        series = pd.Series([1, 2, 3, 4, 5])
        validated = validate_input_array(series)
        
        expected = np.array([1, 2, 3, 4, 5])
        np.testing.assert_array_equal(validated, expected)
    
    def test_validate_input_array_empty(self):
        """Test validation with empty array."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_input_array([])
        
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_input_array(np.array([]))
    
    def test_validate_input_array_nonfinite(self):
        """Test validation with non-finite values."""
        arr = np.array([1, 2, np.nan, 4, np.inf])
        
        with pytest.warns(UserWarning, match="contains non-finite values"):
            validated = validate_input_array(arr)
        
        # Should remove non-finite values
        expected = np.array([1, 2, 4])
        np.testing.assert_array_equal(validated, expected)
    
    def test_validate_input_array_all_nonfinite(self):
        """Test validation with all non-finite values."""
        arr = np.array([np.nan, np.inf, -np.inf])
        
        with pytest.raises(ValueError, match="contains only non-finite values"):
            validate_input_array(arr)
    
    def test_clip_to_range(self):
        """Test clipping to range."""
        values = np.array([-1, 0, 2, 5, 8, 12])
        clipped = clip_to_range(values, 0, 10)
        
        expected = np.array([0, 0, 2, 5, 8, 10])
        np.testing.assert_array_equal(clipped, expected)
    
    def test_check_bias_parameters_valid(self):
        """Test bias parameter validation with valid inputs."""
        # Should not raise any exceptions
        check_bias_parameters(0.5, 0.05)
        check_bias_parameters(0.1, 0.001)
        check_bias_parameters(1.0, 0.1)
    
    def test_check_bias_parameters_invalid_tau(self):
        """Test bias parameter validation with invalid tau."""
        with pytest.raises(ValueError, match="tau must be a positive number"):
            check_bias_parameters(0, 0.05)
        
        with pytest.raises(ValueError, match="tau must be a positive number"):
            check_bias_parameters(-0.5, 0.05)
        
        with pytest.raises(ValueError, match="tau must be a positive number"):
            check_bias_parameters("invalid", 0.05)
    
    def test_check_bias_parameters_invalid_delta(self):
        """Test bias parameter validation with invalid delta."""
        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            check_bias_parameters(0.5, 0)
        
        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            check_bias_parameters(0.5, 1)
        
        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            check_bias_parameters(0.5, 1.5)
        
        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            check_bias_parameters(0.5, "invalid")


class TestRSquaredEstimation:
    """Test R² estimation function."""
    
    def test_estimate_r_squared_basic(self):
        """Test basic R² estimation."""
        np.random.seed(42)
        n_samples = 100
        
        # Create data with known R²
        factor1 = np.random.normal(0, 1, n_samples)
        factor2 = np.random.normal(0, 1, n_samples)
        
        # Target is perfect linear combination
        target = 0.6 * factor1 + 0.4 * factor2
        
        df = pd.DataFrame({
            'factor1': factor1,
            'factor2': factor2,
            'target': target
        })
        
        r_squared = estimate_r_squared_from_scores(df, ['factor1', 'factor2'], 'target')
        
        # Should be close to 1.0 for perfect linear relationship
        assert r_squared > 0.99
    
    def test_estimate_r_squared_partial_relationship(self):
        """Test R² estimation with partial relationship."""
        np.random.seed(42)
        n_samples = 200
        
        # Create data with partial relationship
        factor1 = np.random.normal(0, 1, n_samples)
        factor2 = np.random.normal(0, 1, n_samples)
        noise = np.random.normal(0, 1, n_samples)
        
        # Target is partially explained by factors
        target = 0.5 * factor1 + 0.3 * factor2 + 0.5 * noise
        
        df = pd.DataFrame({
            'factor1': factor1,
            'factor2': factor2,
            'target': target
        })
        
        r_squared = estimate_r_squared_from_scores(df, ['factor1', 'factor2'], 'target')
        
        # Should be positive but less than 1
        assert 0 < r_squared < 1
        assert r_squared > 0.3  # Should explain significant portion
    
    def test_estimate_r_squared_no_relationship(self):
        """Test R² estimation with no relationship."""
        np.random.seed(42)
        n_samples = 100
        
        # Create independent data
        factor1 = np.random.normal(0, 1, n_samples)
        factor2 = np.random.normal(0, 1, n_samples)
        target = np.random.normal(0, 1, n_samples)  # Independent
        
        df = pd.DataFrame({
            'factor1': factor1,
            'factor2': factor2,
            'target': target
        })
        
        r_squared = estimate_r_squared_from_scores(df, ['factor1', 'factor2'], 'target')
        
        # Should be close to 0 for independent data
        assert r_squared < 0.1
    
    def test_estimate_r_squared_missing_values(self):
        """Test R² estimation with missing values."""
        np.random.seed(42)
        n_samples = 100
        
        factor1 = np.random.normal(0, 1, n_samples)
        factor2 = np.random.normal(0, 1, n_samples)
        target = 0.7 * factor1 + 0.3 * factor2
        
        # Add some missing values
        factor1[10:15] = np.nan
        target[20:25] = np.nan
        
        df = pd.DataFrame({
            'factor1': factor1,
            'factor2': factor2,
            'target': target
        })
        
        r_squared = estimate_r_squared_from_scores(df, ['factor1', 'factor2'], 'target')
        
        # Should still work with missing values
        assert r_squared > 0.8  # Should be high despite missing values
    
    def test_estimate_r_squared_empty_data(self):
        """Test R² estimation with empty data."""
        df = pd.DataFrame({
            'factor1': [],
            'factor2': [],
            'target': []
        })
        
        with pytest.raises(ValueError, match="No valid data rows"):
            estimate_r_squared_from_scores(df, ['factor1', 'factor2'], 'target')
    
    def test_estimate_r_squared_all_missing(self):
        """Test R² estimation with all missing values."""
        df = pd.DataFrame({
            'factor1': [np.nan, np.nan, np.nan],
            'factor2': [np.nan, np.nan, np.nan],
            'target': [np.nan, np.nan, np.nan]
        })
        
        with pytest.raises(ValueError, match="No valid data rows"):
            estimate_r_squared_from_scores(df, ['factor1', 'factor2'], 'target')
    
    def test_estimate_r_squared_single_factor(self):
        """Test R² estimation with single factor."""
        np.random.seed(42)
        n_samples = 100
        
        factor1 = np.random.normal(0, 1, n_samples)
        target = 0.8 * factor1 + 0.2 * np.random.normal(0, 1, n_samples)
        
        df = pd.DataFrame({
            'factor1': factor1,
            'target': target
        })
        
        r_squared = estimate_r_squared_from_scores(df, ['factor1'], 'target')
        
        # Should be reasonably high for strong relationship
        assert r_squared > 0.5
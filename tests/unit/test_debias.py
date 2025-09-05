"""
Tests for core differential debiasing functionality
"""

import pytest
import numpy as np
import pandas as pd
from differential_debiasing.core.debias import DifferentialDebias
from differential_debiasing.sensitivity.factor_analysis import FactorAnalysisSensitivity
from differential_debiasing.sensitivity.domain_specific import DomainSpecificSensitivity


class TestDifferentialDebias:
    """Test suite for DifferentialDebias class."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        debias = DifferentialDebias()
        assert debias.tau == 0.5
        assert debias.delta == 0.05
        assert debias.use_average_case == True
        assert debias.random_seed is None
        assert not debias._fitted
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        debias = DifferentialDebias(
            tau=0.3,
            delta=0.01,
            use_average_case=False,
            random_seed=42
        )
        assert debias.tau == 0.3
        assert debias.delta == 0.01
        assert debias.use_average_case == False
        assert debias.random_seed == 42
    
    def test_init_invalid_tau(self):
        """Test initialization with invalid tau."""
        with pytest.raises(ValueError, match="tau must be a positive number"):
            DifferentialDebias(tau=0)
        
        with pytest.raises(ValueError, match="tau must be a positive number"):
            DifferentialDebias(tau=-0.5)
    
    def test_init_invalid_delta(self):
        """Test initialization with invalid delta."""
        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            DifferentialDebias(delta=0)
        
        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            DifferentialDebias(delta=1)
        
        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            DifferentialDebias(delta=1.5)
    
    def test_fit_array_input(self):
        """Test fitting with array input."""
        np.random.seed(42)
        judgments = np.random.normal(6, 2, 100)
        judgments = np.clip(judgments, 1, 10)
        
        debias = DifferentialDebias()
        result = debias.fit(judgments)
        
        assert result is debias  # Should return self
        assert debias._fitted
        assert debias._bias_sensitivity is not None
        assert debias._bias_sensitivity > 0
        assert debias._score_min is not None
        assert debias._score_max is not None
    
    def test_fit_dataframe_input(self):
        """Test fitting with DataFrame input."""
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'completeness_score': np.random.normal(7, 1.5, n_samples),
            'correctness_score': np.random.normal(6.5, 1.2, n_samples),
            'score': np.random.normal(6.8, 2, n_samples)
        })
        
        debias = DifferentialDebias(sensitivity_estimator="factor_analysis")
        debias.fit(df)
        
        assert debias._fitted
        assert debias._bias_sensitivity is not None
    
    def test_fit_empty_input(self):
        """Test fitting with empty input."""
        debias = DifferentialDebias()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            debias.fit([])
        
        with pytest.raises(ValueError, match="cannot be empty"):
            debias.fit(np.array([]))
    
    def test_transform_before_fit(self):
        """Test transform before fitting."""
        debias = DifferentialDebias()
        
        with pytest.raises(ValueError, match="Must call fit\\(\\) before transform\\(\\)"):
            debias.transform([1, 2, 3])
    
    def test_transform_array_input(self):
        """Test transform with array input."""
        np.random.seed(42)
        judgments = np.random.normal(6, 2, 50)
        judgments = np.clip(judgments, 1, 10)
        
        debias = DifferentialDebias(random_seed=42)
        debias.fit(judgments)
        
        # Transform same data
        debiased = debias.transform(judgments)
        
        assert len(debiased) == len(judgments)
        assert np.all(debiased >= 1)
        assert np.all(debiased <= 10)
        assert not np.array_equal(judgments, debiased)  # Should be different due to noise
    
    def test_transform_different_data(self):
        """Test transform with different data than fit."""
        np.random.seed(42)
        fit_data = np.random.normal(6, 2, 100)
        fit_data = np.clip(fit_data, 1, 10)
        
        transform_data = np.random.normal(7, 1.5, 30)
        transform_data = np.clip(transform_data, 1, 10)
        
        debias = DifferentialDebias(random_seed=42)
        debias.fit(fit_data)
        debiased = debias.transform(transform_data)
        
        assert len(debiased) == len(transform_data)
        assert np.all(debiased >= 1)
        assert np.all(debiased <= 10)
    
    def test_fit_transform(self):
        """Test fit_transform convenience method."""
        np.random.seed(42)
        judgments = np.random.normal(6, 2, 50)
        judgments = np.clip(judgments, 1, 10)
        
        debias = DifferentialDebias(random_seed=42)
        debiased = debias.fit_transform(judgments)
        
        assert debias._fitted
        assert len(debiased) == len(judgments)
        assert np.all(debiased >= 1)
        assert np.all(debiased <= 10)
    
    def test_fit_transform_dataframe(self):
        """Test fit_transform with DataFrame."""
        np.random.seed(42)
        n_samples = 50
        
        df = pd.DataFrame({
            'completeness_score': np.random.normal(7, 1.5, n_samples),
            'correctness_score': np.random.normal(6.5, 1.2, n_samples),
            'score': np.random.normal(6.8, 2, n_samples)
        })
        
        debias = DifferentialDebias(
            sensitivity_estimator="factor_analysis",
            random_seed=42
        )
        debiased = debias.fit_transform(df)
        
        assert debias._fitted
        assert len(debiased) == len(df)
        assert np.all(debiased >= 1)
        assert np.all(debiased <= 10)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        np.random.seed(42)
        judgments = np.random.normal(6, 2, 50)
        judgments = np.clip(judgments, 1, 10)
        
        debias1 = DifferentialDebias(random_seed=42)
        debiased1 = debias1.fit_transform(judgments)
        
        debias2 = DifferentialDebias(random_seed=42)
        debiased2 = debias2.fit_transform(judgments)
        
        np.testing.assert_array_equal(debiased1, debiased2)
    
    def test_different_seeds_different_results(self):
        """Test that different seeds give different results."""
        np.random.seed(42)
        judgments = np.random.normal(6, 2, 50)
        judgments = np.clip(judgments, 1, 10)
        
        debias1 = DifferentialDebias(random_seed=42)
        debiased1 = debias1.fit_transform(judgments)
        
        debias2 = DifferentialDebias(random_seed=123)
        debiased2 = debias2.fit_transform(judgments)
        
        assert not np.array_equal(debiased1, debiased2)
    
    def test_average_case_vs_individual_case(self):
        """Test difference between average case and individual case."""
        np.random.seed(42)
        judgments = np.random.normal(6, 2, 100)
        judgments = np.clip(judgments, 1, 10)
        
        # Average case (default)
        debias_avg = DifferentialDebias(use_average_case=True, random_seed=42)
        debiased_avg = debias_avg.fit_transform(judgments)
        
        # Individual case
        debias_ind = DifferentialDebias(use_average_case=False, random_seed=42)
        debiased_ind = debias_ind.fit_transform(judgments)
        
        # Average case should have less noise (higher correlation)
        corr_avg = np.corrcoef(judgments, debiased_avg)[0, 1]
        corr_ind = np.corrcoef(judgments, debiased_ind)[0, 1]
        
        assert corr_avg > corr_ind
    
    def test_get_bias_bounds(self):
        """Test bias bounds calculation."""
        np.random.seed(42)
        judgments = np.random.normal(6, 2, 50)
        judgments = np.clip(judgments, 1, 10)
        
        debias = DifferentialDebias(tau=0.5, delta=0.05)
        debias.fit(judgments)
        
        bounds = debias.get_bias_bounds(len(judgments))
        
        assert 'tau' in bounds
        assert 'delta' in bounds
        assert 'max_bias_ratio' in bounds
        assert 'bias_sensitivity' in bounds
        assert 'noise_std' in bounds
        assert 'protection_guarantee' in bounds
        
        assert bounds['tau'] == 0.5
        assert bounds['delta'] == 0.05
        assert bounds['max_bias_ratio'] == pytest.approx(np.exp(0.5), rel=1e-3)
        assert bounds['bias_sensitivity'] > 0
        assert bounds['noise_std'] > 0
    
    def test_get_diagnostics(self):
        """Test diagnostics retrieval."""
        np.random.seed(42)
        judgments = np.random.normal(6, 2, 50)
        judgments = np.clip(judgments, 1, 10)
        
        debias = DifferentialDebias(tau=0.3, delta=0.01)
        
        # Before fitting
        diagnostics = debias.get_diagnostics()
        assert diagnostics['fitted'] == False
        assert 'bias_sensitivity' not in diagnostics
        
        # After fitting
        debias.fit(judgments)
        diagnostics = debias.get_diagnostics()
        assert diagnostics['fitted'] == True
        assert 'bias_sensitivity' in diagnostics
        assert 'score_range' in diagnostics
        assert 'sensitivity_estimator' in diagnostics
    
    def test_validate_effectiveness(self):
        """Test validation of debiasing effectiveness."""
        np.random.seed(42)
        judgments = np.random.normal(6, 2, 100)
        judgments = np.clip(judgments, 1, 10)
        
        debias = DifferentialDebias(random_seed=42)
        debiased = debias.fit_transform(judgments)
        
        validation = debias.validate_effectiveness(judgments, debiased)
        
        assert 'correlation' in validation
        assert 'mean_absolute_difference' in validation
        assert 'variance_ratio' in validation
        assert 'signal_preservation' in validation
        assert 'noise_level' in validation
        
        assert 0 <= validation['correlation'] <= 1
        assert validation['mean_absolute_difference'] >= 0
        assert validation['variance_ratio'] > 0
        assert validation['signal_preservation'] == validation['correlation']
    
    def test_validate_effectiveness_mismatched_lengths(self):
        """Test validation with mismatched array lengths."""
        np.random.seed(42)
        judgments = np.random.normal(6, 2, 100)
        judgments = np.clip(judgments, 1, 10)
        
        debias = DifferentialDebias()
        debias.fit(judgments)
        
        with pytest.raises(ValueError, match="must have same length"):
            debias.validate_effectiveness(judgments, judgments[:50])
    
    def test_sensitivity_estimator_instances(self):
        """Test using sensitivity estimator instances."""
        np.random.seed(42)
        judgments = np.random.normal(6, 2, 50)
        judgments = np.clip(judgments, 1, 10)
        
        # Test with FactorAnalysisSensitivity instance
        factor_estimator = FactorAnalysisSensitivity()
        debias1 = DifferentialDebias(sensitivity_estimator=factor_estimator)
        debiased1 = debias1.fit_transform(judgments)
        
        # Test with DomainSpecificSensitivity instance
        domain_estimator = DomainSpecificSensitivity(domain="academic")
        debias2 = DifferentialDebias(sensitivity_estimator=domain_estimator)
        debiased2 = debias2.fit_transform(judgments)
        
        assert len(debiased1) == len(judgments)
        assert len(debiased2) == len(judgments)
        assert not np.array_equal(debiased1, debiased2)  # Different methods should give different results
    
    def test_invalid_sensitivity_estimator(self):
        """Test initialization with invalid sensitivity estimator."""
        with pytest.raises(ValueError, match="Unknown sensitivity estimator"):
            DifferentialDebias(sensitivity_estimator="invalid_method")
    
    def test_extreme_parameters(self):
        """Test with extreme parameter values."""
        np.random.seed(42)
        judgments = np.random.normal(6, 2, 50)
        judgments = np.clip(judgments, 1, 10)
        
        # Very strong protection (low tau, low delta)
        debias_strong = DifferentialDebias(tau=0.1, delta=0.001, random_seed=42)
        debiased_strong = debias_strong.fit_transform(judgments)
        
        # Very weak protection (high tau, high delta)
        debias_weak = DifferentialDebias(tau=1.0, delta=0.1, random_seed=42)
        debiased_weak = debias_weak.fit_transform(judgments)
        
        # Strong protection should have lower correlation (more noise)
        corr_strong = np.corrcoef(judgments, debiased_strong)[0, 1]
        corr_weak = np.corrcoef(judgments, debiased_weak)[0, 1]
        
        assert corr_strong < corr_weak
    
    def test_single_judgment(self):
        """Test with single judgment."""
        debias = DifferentialDebias(random_seed=42)
        
        # Single judgment should work
        single_judgment = np.array([7.5])
        debiased = debias.fit_transform(single_judgment)
        
        assert len(debiased) == 1
        assert 1 <= debiased[0] <= 10
    
    def test_identical_judgments(self):
        """Test with identical judgments."""
        debias = DifferentialDebias(random_seed=42)
        
        # All identical judgments
        identical_judgments = np.array([7.0] * 50)
        debiased = debias.fit_transform(identical_judgments)
        
        assert len(debiased) == len(identical_judgments)
        assert np.all(debiased >= 1)
        assert np.all(debiased <= 10)
        # Should add noise, so not all identical
        assert not np.all(debiased == debiased[0])
    
    def test_custom_score_range(self):
        """Test with custom score range."""
        np.random.seed(42)
        judgments = np.random.normal(50, 10, 100)  # 0-100 scale
        judgments = np.clip(judgments, 0, 100)
        
        debias = DifferentialDebias(random_seed=42)
        debias.fit(judgments)
        
        # Transform with custom range
        debiased = debias.transform(judgments, score_min=0, score_max=100)
        
        assert np.all(debiased >= 0)
        assert np.all(debiased <= 100)
    
    def test_nonfinite_values(self):
        """Test handling of non-finite values."""
        judgments = np.array([1, 2, np.nan, 4, np.inf, 6])
        
        debias = DifferentialDebias()
        
        # Should handle non-finite values gracefully
        debias.fit(judgments)
        assert debias._fitted
        
        # Transform should work with finite values
        finite_judgments = np.array([1, 2, 3, 4, 5, 6])
        debiased = debias.transform(finite_judgments)
        assert len(debiased) == len(finite_judgments)
    
    def test_score_range_validation(self):
        """Test score range validation."""
        np.random.seed(42)
        judgments = np.random.normal(6, 2, 50)
        judgments = np.clip(judgments, 1, 10)
        
        debias = DifferentialDebias()
        debias.fit(judgments)
        
        # Invalid score range
        with pytest.raises(ValueError, match="score_max must be greater than score_min"):
            debias.transform(judgments, score_min=10, score_max=1)
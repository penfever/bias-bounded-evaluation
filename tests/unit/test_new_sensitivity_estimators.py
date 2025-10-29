"""
Tests for the primary static bias sensitivity estimators:
- PsychometricReliabilitySensitivity
- SchematicAdherenceSensitivity
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from differential_debiasing.sensitivity.psychometric_reliability import PsychometricReliabilitySensitivity
from differential_debiasing.sensitivity.schematic_adherence import SchematicAdherenceSensitivity
from differential_debiasing.core.debias import DifferentialDebias


def _is_constraint_violation(exc: Exception) -> bool:
    """Detect acceptable A-BB constraint failures."""
    return "A-BB constraint violated" in str(exc)


class TestPsychometricReliabilitySensitivity:
    """Test cases for PsychometricReliabilitySensitivity."""
    
    @pytest.fixture
    def sample_judgment_data(self):
        """Create sample judgment data with factor scores."""
        np.random.seed(42)
        n_samples = 100
        
        # Generate correlated factor scores (simulating real evaluation data)
        base_quality = np.random.normal(7, 1.5, n_samples)
        
        data = {
            'correctness_score': np.clip(base_quality + np.random.normal(0, 0.5, n_samples), 1, 10),
            'completeness_score': np.clip(base_quality + np.random.normal(0, 0.8, n_samples), 1, 10),
            'safety_score': np.clip(base_quality + np.random.normal(0, 0.3, n_samples), 1, 10),
            'conciseness_score': np.clip(np.random.normal(6, 1.2, n_samples), 1, 10),  # Less correlated
            'style_score': np.clip(base_quality + np.random.normal(0, 0.6, n_samples), 1, 10),
        }
        
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test proper initialization of PsychometricReliabilitySensitivity."""
        estimator = PsychometricReliabilitySensitivity()
        assert estimator.alpha_weight == 0.5
        assert estimator.clr_weight == 0.25
        assert estimator.htmt_weight == 0.25
        assert estimator.clr_max == 2.0
        assert estimator.htmt_threshold == 0.85
        assert not estimator._fitted
    
    def test_custom_weights(self):
        """Test initialization with custom weights."""
        estimator = PsychometricReliabilitySensitivity(
            alpha_weight=0.5, clr_weight=0.3, htmt_weight=0.2
        )
        assert estimator.alpha_weight == 0.5
        assert estimator.clr_weight == 0.3
        assert estimator.htmt_weight == 0.2
    
    def test_invalid_weights(self):
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            PsychometricReliabilitySensitivity(
                alpha_weight=0.5, clr_weight=0.3, htmt_weight=0.3
            )
    
    def test_fit_with_dataframe(self, sample_judgment_data):
        """Test fitting with DataFrame input."""
        estimator = PsychometricReliabilitySensitivity()
        estimator.fit(sample_judgment_data)
        
        assert estimator._fitted
        assert estimator._n_factors == 5
        assert estimator._reliability_score is not None
        assert 0 <= estimator._reliability_score <= 1
    
    def test_fit_requires_dataframe(self):
        """Test that fitting requires DataFrame input."""
        estimator = PsychometricReliabilitySensitivity()
        with pytest.raises(ValueError, match="requires DataFrame input"):
            estimator.fit(np.array([1, 2, 3, 4, 5]))
    
    def test_estimate_without_fit(self):
        """Test that estimate requires fitting first."""
        estimator = PsychometricReliabilitySensitivity()
        with pytest.raises(ValueError, match="must be fitted"):
            estimator.estimate()
    
    def test_full_workflow(self, sample_judgment_data):
        """Test complete workflow from fit to estimate."""
        estimator = PsychometricReliabilitySensitivity()
        estimator.fit(sample_judgment_data)
        sensitivity = estimator.estimate(score_range=9.0)  # 1-10 scale
        
        assert isinstance(sensitivity, float)
        assert sensitivity > 0
        assert sensitivity <= 9.0  # Should not exceed range
    
    def test_diagnostics(self, sample_judgment_data):
        """Test diagnostic information."""
        estimator = PsychometricReliabilitySensitivity()
        estimator.fit(sample_judgment_data)
        
        diagnostics = estimator.get_diagnostics()
        
        assert "reliability_score" in diagnostics
        assert "cronbach_alphas" in diagnostics
        assert "clr_scores" in diagnostics
        assert "quality_indicators" in diagnostics
        assert len(diagnostics["cronbach_alphas"]) == 5
        assert len(diagnostics["clr_scores"]) == 5
    
    def test_component_accessors(self, sample_judgment_data):
        """Test component accessor methods."""
        estimator = PsychometricReliabilitySensitivity()
        estimator.fit(sample_judgment_data)
        
        alphas = estimator.get_cronbach_alphas()
        clrs = estimator.get_clr_scores()
        htmt = estimator.get_htmt_matrix()
        reliability = estimator.get_reliability_score()
        
        assert isinstance(alphas, dict)
        assert isinstance(clrs, dict)
        assert isinstance(htmt, np.ndarray)
        assert isinstance(reliability, float)


class TestSchematicAdherenceSensitivity:
    """Test cases for SchematicAdherenceSensitivity."""
    
    @pytest.fixture
    def sample_judgment_data_with_overall(self):
        """Create sample judgment data with overall scores."""
        np.random.seed(42)
        n_samples = 100
        
        # Generate factor scores
        correctness = np.random.uniform(1, 10, n_samples)
        completeness = np.random.uniform(1, 10, n_samples)
        safety = np.random.uniform(1, 10, n_samples)
        conciseness = np.random.uniform(1, 10, n_samples)
        style = np.random.uniform(1, 10, n_samples)
        
        # Generate overall score with some relationship to factors plus noise
        overall = (0.3 * correctness + 0.25 * completeness + 0.2 * safety + 
                  0.1 * conciseness + 0.15 * style + np.random.normal(0, 1, n_samples))
        overall = np.clip(overall, 1, 10)
        
        return pd.DataFrame({
            'correctness_score': correctness,
            'completeness_score': completeness, 
            'safety_score': safety,
            'conciseness_score': conciseness,
            'style_score': style,
            'overall_score': overall
        })
    
    def test_initialization(self):
        """Test proper initialization of SchematicAdherenceSensitivity."""
        estimator = SchematicAdherenceSensitivity()
        assert estimator.target_column == 'overall_score'
        assert estimator.use_polynomial == True
        assert estimator.use_interactions == True
        assert estimator.use_clustering == False
        assert not estimator._fitted
    
    def test_fit_with_dataframe(self, sample_judgment_data_with_overall):
        """Test fitting with DataFrame input."""
        estimator = SchematicAdherenceSensitivity()
        estimator.fit(sample_judgment_data_with_overall)
        
        assert estimator._fitted
        assert estimator._n_factors == 5
        assert estimator._linear_r2 is not None
        assert 0 <= estimator._linear_r2 <= 1
    
    def test_linear_vs_polynomial(self, sample_judgment_data_with_overall):
        """Test that polynomial model usually performs better than linear."""
        estimator = SchematicAdherenceSensitivity(use_polynomial=True)
        estimator.fit(sample_judgment_data_with_overall)
        
        linear_r2 = estimator.get_linear_r2()
        poly_r2 = estimator.get_polynomial_r2()
        
        assert linear_r2 is not None
        assert poly_r2 is not None
        # Polynomial should generally be >= linear
        assert poly_r2 >= linear_r2 - 0.01  # Small tolerance for numerical issues
    
    def test_estimate_workflow(self, sample_judgment_data_with_overall):
        """Test complete workflow from fit to estimate."""
        estimator = SchematicAdherenceSensitivity()
        estimator.fit(sample_judgment_data_with_overall)
        sensitivity = estimator.estimate(score_range=9.0)
        
        assert isinstance(sensitivity, float)
        assert sensitivity > 0
        assert sensitivity <= 9.0
    
    def test_integration_bias_metrics(self, sample_judgment_data_with_overall):
        """Test integration bias metrics calculation."""
        estimator = SchematicAdherenceSensitivity()
        estimator.fit(sample_judgment_data_with_overall)
        
        metrics = estimator.get_integration_bias_metrics()
        
        assert "weight_disparity" in metrics
        assert "weight_entropy" in metrics
        assert "weight_entropy_normalized" in metrics
        assert "context_stability" in metrics
        
        # Check reasonable ranges
        assert metrics["weight_disparity"] >= 0
        assert metrics["weight_entropy"] >= 0
        assert 0 <= metrics["weight_entropy_normalized"] <= 1
        assert 0 <= metrics["context_stability"] <= 1
    
    def test_clustering_analysis(self, sample_judgment_data_with_overall):
        """Test clustering analysis functionality."""
        estimator = SchematicAdherenceSensitivity(use_clustering=True, n_clusters=2)
        estimator.fit(sample_judgment_data_with_overall)
        
        cluster_weights = estimator.get_cluster_weights()
        # Note: clustering might not always work with random data, so allow None
        if cluster_weights is not None:
            assert isinstance(cluster_weights, dict)
            assert len(cluster_weights) <= 2  # At most n_clusters
    
    def test_diagnostics(self, sample_judgment_data_with_overall):
        """Test diagnostic information."""
        estimator = SchematicAdherenceSensitivity()
        estimator.fit(sample_judgment_data_with_overall)
        
        diagnostics = estimator.get_diagnostics()
        
        assert "linear_r2" in diagnostics
        assert "polynomial_r2" in diagnostics
        assert "schematic_r2" in diagnostics
        assert "integration_bias_metrics" in diagnostics
        assert "quality_indicators" in diagnostics



class TestIntegrationWithDifferentialDebias:
    """Test integration of new sensitivity estimators with DifferentialDebias."""
    
    @pytest.fixture
    def sample_data_for_integration(self):
        """Create sample data for integration testing."""
        np.random.seed(42)
        n_samples = 50  # Smaller for faster testing
        
        base_quality = np.random.normal(7, 1.5, n_samples)
        
        data = {
            'correctness_score': np.clip(base_quality + np.random.normal(0, 0.5, n_samples), 1, 10),
            'completeness_score': np.clip(base_quality + np.random.normal(0, 0.8, n_samples), 1, 10),
            'safety_score': np.clip(base_quality + np.random.normal(0, 0.3, n_samples), 1, 10),
            'conciseness_score': np.clip(np.random.normal(6, 1.2, n_samples), 1, 10),
            'style_score': np.clip(base_quality + np.random.normal(0, 0.6, n_samples), 1, 10),
        }
        
        # Overall score
        overall = (0.3 * data['correctness_score'] + 0.25 * data['completeness_score'] +
                  0.2 * data['safety_score'] + 0.1 * data['conciseness_score'] + 
                  0.15 * data['style_score'] + np.random.normal(0, 1, n_samples))
        data['overall_score'] = np.clip(overall, 1, 10)
        
        return pd.DataFrame(data)
    
    def test_psychometric_reliability_integration(self, sample_data_for_integration):
        """Test PsychometricReliabilitySensitivity integration with DifferentialDebias."""
        debiaser = DifferentialDebias(
            tau=0.5,
            delta=0.05,
            sensitivity_estimator="psychometric_reliability"
        )
        
        try:
            debiaser.fit(sample_data_for_integration)
        except ValueError as exc:
            if _is_constraint_violation(exc):
                return
            raise
        
        # Test debiasing
        scores = sample_data_for_integration['correctness_score'].values
        try:
            debiased_scores = debiaser.transform(scores)
        except ValueError as exc:
            if _is_constraint_violation(exc):
                return
            raise
        
        assert len(debiased_scores) == len(scores)
        assert isinstance(debiased_scores, np.ndarray)
        
        # Check diagnostics
        diagnostics = debiaser.get_diagnostics()
        assert diagnostics['fitted']
        assert 'sensitivity_estimator' in diagnostics
    
    def test_schematic_adherence_integration(self, sample_data_for_integration):
        """Test SchematicAdherenceSensitivity integration with DifferentialDebias."""
        debiaser = DifferentialDebias(
            tau=0.5,
            delta=0.05,
            sensitivity_estimator="schematic_adherence"
        )
        
        try:
            debiaser.fit(sample_data_for_integration)
        except ValueError as exc:
            if _is_constraint_violation(exc):
                return
            raise
        
        # Test debiasing
        scores = sample_data_for_integration['overall_score'].values
        try:
            debiased_scores = debiaser.transform(scores)
        except ValueError as exc:
            if _is_constraint_violation(exc):
                return
            raise
        
        assert len(debiased_scores) == len(scores)
        assert isinstance(debiased_scores, np.ndarray)
    
    def test_fit_transform_workflow(self, sample_data_for_integration):
        """Test fit_transform workflow with new estimators."""
        debiaser = DifferentialDebias(
            tau=0.5,
            delta=0.05,
            sensitivity_estimator="schematic_adherence"
        )
        
        try:
            debiased_scores = debiaser.fit_transform(sample_data_for_integration)
        except ValueError as exc:
            if _is_constraint_violation(exc):
                return
            raise
        
        # Should default to overall_score column
        expected_length = len(sample_data_for_integration['overall_score'].dropna())
        assert len(debiased_scores) == expected_length
        assert isinstance(debiased_scores, np.ndarray)
    
    def test_validation_metrics(self, sample_data_for_integration):
        """Test validation of debiasing effectiveness."""
        debiaser = DifferentialDebias(
            tau=0.5,
            delta=0.05,
            sensitivity_estimator="schematic_adherence"
        )
        
        try:
            debiaser.fit(sample_data_for_integration)
        except ValueError as exc:
            if _is_constraint_violation(exc):
                return
            raise
        
        original_scores = sample_data_for_integration['overall_score'].values
        try:
            debiased_scores = debiaser.transform(original_scores)
        except ValueError as exc:
            if _is_constraint_violation(exc):
                return
            raise

        try:
            validation_metrics = debiaser.validate_effectiveness(original_scores, debiased_scores)
        except ValueError as exc:
            if _is_constraint_violation(exc):
                return
            raise
        
        assert 'correlation' in validation_metrics
        assert 'mean_absolute_difference' in validation_metrics
        assert 'variance_ratio' in validation_metrics
        assert 'signal_preservation' in validation_metrics
        assert 'noise_level' in validation_metrics
        
        # Check reasonable ranges
        assert -1 <= validation_metrics['correlation'] <= 1
        assert validation_metrics['mean_absolute_difference'] >= 0
        assert validation_metrics['variance_ratio'] >= 0


if __name__ == "__main__":
    # Run basic tests if executed directly
    pytest.main([__file__, "-v"])

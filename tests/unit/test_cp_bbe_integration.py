#!/usr/bin/env python3
"""
Test script for Conformal Prediction + Bias-Bounded Evaluation integration.

This script demonstrates the unified CP+BBE framework with multi-judge scenarios,
showing how uncertainty quantification can improve bias protection.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from differential_debiasing.sensitivity.conformal_bbe_unified import (
        ConformedBiasBoundedPredictor, CPBBEConfig
    )
    from differential_debiasing.sensitivity.conformal_sensitivity import ConformalSensitivityEstimator
    from differential_debiasing.sensitivity.multi_judge_conformal import MultiJudgeConformalPredictor
    from differential_debiasing.core.debias import DifferentialDebias
    
    CP_BBE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ CP+BBE modules not available: {e}")
    CP_BBE_AVAILABLE = False


def generate_synthetic_multi_judge_data(n_samples: int = 200, 
                                       n_judges: int = 4,
                                       bias_strength: float = 0.3,
                                       random_seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generate synthetic multi-judge evaluation data with controlled bias patterns.
    
    Parameters:
    -----------
    n_samples : int
        Number of evaluation samples
    n_judges : int
        Number of judges
    bias_strength : float
        Strength of systematic bias (0 = no bias, 1 = strong bias)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict[str, np.ndarray] : Judge predictions and metadata
    """
    rng = np.random.RandomState(random_seed)
    
    # Generate true quality scores (latent ground truth)
    true_quality = rng.uniform(1, 10, n_samples)
    
    # Generate judge-specific biases
    judge_biases = rng.normal(0, bias_strength, n_judges)
    judge_reliabilities = rng.uniform(0.7, 1.0, n_judges)  # Judge reliability varies
    
    judge_predictions = {}
    judge_metadata = {}
    
    for i in range(n_judges):
        judge_name = f"judge_{i+1}"
        
        # Apply judge-specific bias and noise
        base_scores = true_quality + judge_biases[i]
        noise_level = (2 - judge_reliabilities[i]) * 0.5  # Less reliable judges are noisier
        noise = rng.normal(0, noise_level, n_samples)
        
        judge_scores = np.clip(base_scores + noise, 1, 10)
        judge_predictions[judge_name] = judge_scores
        
        judge_metadata[judge_name] = {
            'bias': judge_biases[i],
            'reliability': judge_reliabilities[i],
            'noise_level': noise_level
        }
    
    return {
        'judge_predictions': judge_predictions,
        'judge_metadata': judge_metadata,
        'true_quality': true_quality,
        'judge_biases': judge_biases,
        'judge_reliabilities': judge_reliabilities
    }


def test_conformal_sensitivity():
    """Test basic conformal sensitivity estimation."""
    print("🧪 Testing Conformal Sensitivity Estimation")
    print("-" * 50)
    
    # Generate single-judge data
    rng = np.random.RandomState(42)
    scores = rng.normal(5, 2, 100)
    scores = np.clip(scores, 1, 10)
    
    try:
        # Create conformal sensitivity estimator
        estimator = ConformalSensitivityEstimator(
            conformal_method="split_cp",
            alpha=0.1,  # 90% coverage
            aggregation_strategy="uncertainty_weighted",
            random_seed=42
        )
        
        # Fit estimator
        estimator.fit(scores)
        
        # Get sensitivity estimate
        sensitivity = estimator.estimate()
        
        # Get uncertainty scores and intervals
        uncertainty_scores = estimator.get_uncertainty_scores()
        prediction_intervals = estimator.get_prediction_intervals()
        
        print(f"✅ Conformal sensitivity: {sensitivity:.4f}")
        print(f"   Mean uncertainty: {np.mean(uncertainty_scores):.4f}")
        print(f"   Std uncertainty: {np.std(uncertainty_scores):.4f}")
        
        if isinstance(prediction_intervals, tuple):
            lower, upper = prediction_intervals
            print(f"   Mean interval width: {np.mean(upper - lower):.4f}")
        
        # Get diagnostics
        diagnostics = estimator.get_diagnostics()
        print(f"   Coverage level: {diagnostics['coverage_level']:.1%}")
        print(f"   Conformal method: {diagnostics['conformal_method']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conformal sensitivity test failed: {e}")
        return False


def test_multi_judge_conformal():
    """Test multi-judge conformal prediction."""
    print("\n🧪 Testing Multi-Judge Conformal Prediction")
    print("-" * 50)
    
    # Generate multi-judge data
    data = generate_synthetic_multi_judge_data(n_samples=150, n_judges=4, bias_strength=0.4)
    judge_predictions = data['judge_predictions']
    judge_metadata = data['judge_metadata']
    
    try:
        # Create multi-judge conformal predictor
        predictor = MultiJudgeConformalPredictor(
            aggregation_method="weighted_ensemble",
            conformal_method="split_cp",
            alpha=0.1,
            judge_weight_strategy="reliability_based",
            random_seed=42
        )
        
        # Fit predictor
        predictor.fit(judge_predictions, judge_metadata)
        
        # Get predictions and uncertainties
        aggregated_predictions, uncertainties = predictor.predict()
        
        # Get judge weights
        judge_weights = predictor.get_judge_weights()
        
        print(f"✅ Multi-judge aggregation successful")
        print(f"   Number of judges: {len(judge_predictions)}")
        print(f"   Aggregated predictions shape: {aggregated_predictions.shape}")
        print(f"   Mean uncertainty: {np.mean(uncertainties):.4f}")
        
        print(f"\n   Judge weights:")
        for judge, weight in judge_weights.items():
            original_reliability = judge_metadata[judge]['reliability']
            print(f"     {judge}: {weight:.3f} (true reliability: {original_reliability:.3f})")
        
        # Get diagnostics
        diagnostics = predictor.get_diagnostics()
        print(f"   Aggregation method: {diagnostics['aggregation_method']}")
        print(f"   Judge weight strategy: {diagnostics['judge_weight_strategy']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-judge conformal test failed: {e}")
        return False


def test_unified_cp_bbe_framework():
    """Test the unified CP+BBE framework."""
    print("\n🧪 Testing Unified CP+BBE Framework")
    print("-" * 50)
    
    # Generate multi-judge data with bias
    data = generate_synthetic_multi_judge_data(n_samples=200, n_judges=5, bias_strength=0.5)
    judge_predictions = data['judge_predictions']
    judge_metadata = data['judge_metadata']
    
    try:
        # Create configuration
        config = CPBBEConfig(
            conformal_alpha=0.1,  # 90% coverage
            conformal_method="split_cp",
            bbe_tau=1.0,
            bbe_delta=0.1,
            adaptation_strategy="uncertainty_weighted",
            enable_multi_judge=True,
            judge_aggregation="weighted_ensemble",
            noise_adaptation_factor=0.6  # Reduce noise by up to 60% for uncertain predictions
        )
        
        # Create unified predictor
        predictor = ConformedBiasBoundedPredictor(
            config=config,
            random_seed=42
        )
        
        # Fit predictor
        predictor.fit(judge_predictions, judge_metadata)
        
        # Get sensitivity estimate
        adaptive_sensitivity = predictor.estimate()
        
        # Test some predictions with adaptive noise
        test_scores = np.array([6.2, 7.8, 4.1, 8.9, 5.5])
        debiased_results = predictor.predict_with_adaptive_noise(test_scores)
        
        print(f"✅ Unified CP+BBE framework successful")
        print(f"   Adaptive sensitivity: {adaptive_sensitivity:.4f}")
        print(f"   Original test scores: {test_scores}")
        print(f"   Debiased test scores: {debiased_results['debiased_scores']}")
        print(f"   Adaptive noise applied: {debiased_results['adaptive_noise']}")
        
        # Get comprehensive diagnostics
        diagnostics = predictor.get_diagnostics()
        print(f"\n   Framework: {diagnostics['framework']}")
        print(f"   Conformal coverage: {diagnostics['conformal_coverage']:.1%}")
        print(f"   BBE tau: {diagnostics['bbe_tau']}")
        print(f"   Adaptation strategy: {diagnostics['adaptation_strategy']}")
        print(f"   Noise reduction achieved: {diagnostics['noise_reduction_achieved']:.1%}")
        
        if 'validation' in diagnostics:
            validation = diagnostics['validation']
            print(f"   Validation passed: {validation['passes_validation']}")
            print(f"   Overall validation score: {validation['overall_validation_score']:.3f}")
        
        # Compare with baseline
        baseline_comparison = predictor.compare_with_baseline("fixed_noise")
        print(f"\n   Comparison with fixed noise:")
        print(f"     Noise reduction: {baseline_comparison['noise_reduction']:.1%}")
        print(f"     Efficiency gain: {baseline_comparison['efficiency_gain']:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified CP+BBE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_differential_debias():
    """Test integration with the main DifferentialDebias class."""
    print("\n🧪 Testing Integration with DifferentialDebias")
    print("-" * 50)
    
    # Generate data
    data = generate_synthetic_multi_judge_data(n_samples=100, n_judges=3, bias_strength=0.3)
    judge_predictions = data['judge_predictions']
    
    # Convert to DataFrame format
    df = pd.DataFrame(judge_predictions)
    df.columns = [f"{col}_score" for col in df.columns]  # Add _score suffix
    
    try:
        # Test CP+BBE through DifferentialDebias interface
        debiaser = DifferentialDebias(
            tau=1.2,
            delta=0.15,
            sensitivity_estimator="cp_bbe_unified",
            random_seed=42,
            # CP+BBE specific parameters
            conformal_alpha=0.1,
            adaptation_strategy="uncertainty_weighted",
            enable_multi_judge=True
        )
        
        # Fit and transform
        debiaser.fit(df)
        
        # Get original scores (use first judge as example)
        original_scores = df.iloc[:, 0].values
        debiased_scores = debiaser.transform(original_scores)
        
        # Get diagnostics
        diagnostics = debiaser.get_diagnostics()
        
        print(f"✅ Integration with DifferentialDebias successful")
        print(f"   Estimator type: {diagnostics.get('sensitivity_estimator_type', 'N/A')}")
        print(f"   Bias sensitivity: {diagnostics.get('bias_sensitivity', 'N/A'):.4f}")
        print(f"   Original scores (first 5): {original_scores[:5]}")
        print(f"   Debiased scores (first 5): {debiased_scores[:5]}")
        
        # Validate effectiveness
        validation = debiaser.validate_effectiveness(original_scores, debiased_scores)
        print(f"   Correlation: {validation['correlation']:.3f}")
        print(f"   Signal preservation: {validation['signal_preservation']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_adaptive_noise_benefits():
    """Demonstrate the benefits of adaptive noise injection."""
    print("\n📊 Demonstrating Adaptive Noise Benefits")
    print("-" * 50)
    
    # Generate data with varying uncertainty levels
    rng = np.random.RandomState(42)
    n_samples = 100
    
    # Create scenarios with different uncertainty levels
    # Low uncertainty: judges agree
    low_uncertainty_scores = rng.normal(7, 0.3, (n_samples // 3, 3))
    
    # Medium uncertainty: moderate disagreement
    medium_uncertainty_scores = rng.normal(5, 1.0, (n_samples // 3, 3))
    
    # High uncertainty: high disagreement
    base_high = rng.normal(4, 0.5, n_samples // 3)
    high_uncertainty_scores = np.column_stack([
        base_high + rng.normal(0, 1.5, n_samples // 3),
        base_high + rng.normal(0, 1.2, n_samples // 3),
        base_high + rng.normal(0, 1.8, n_samples // 3)
    ])
    
    # Combine all scenarios
    all_scores = np.vstack([low_uncertainty_scores, medium_uncertainty_scores, high_uncertainty_scores])
    
    # Labels for scenarios
    scenario_labels = (['Low Uncertainty'] * (n_samples // 3) + 
                      ['Medium Uncertainty'] * (n_samples // 3) +
                      ['High Uncertainty'] * (n_samples // 3))
    
    try:
        # Create judge predictions dict
        judge_predictions = {
            f'judge_{i+1}': all_scores[:, i] for i in range(3)
        }
        
        # Create unified predictor
        config = CPBBEConfig(
            adaptation_strategy="uncertainty_weighted",
            noise_adaptation_factor=0.7
        )
        
        predictor = ConformedBiasBoundedPredictor(config=config, random_seed=42)
        predictor.fit(judge_predictions)
        
        # Get adaptive noise parameters
        diagnostics = predictor.get_diagnostics()
        adaptive_params = predictor._adaptive_noise_parameters
        
        # Compare with fixed noise baseline
        baseline_comparison = predictor.compare_with_baseline("fixed_noise")
        
        print(f"✅ Adaptive noise demonstration completed")
        print(f"   Total samples: {n_samples}")
        print(f"   Scenarios: Low/Medium/High uncertainty")
        print(f"   Mean noise reduction: {baseline_comparison['noise_reduction']:.1%}")
        print(f"   Efficiency gain: {baseline_comparison['efficiency_gain']:.2f}x")
        
        # Show noise adaptation by scenario
        uncertainty_scores = adaptive_params['uncertainty_normalized']
        noise_stds = adaptive_params['adaptive_noise_stds']
        
        for scenario in ['Low Uncertainty', 'Medium Uncertainty', 'High Uncertainty']:
            mask = [label == scenario for label in scenario_labels]
            scenario_uncertainty = uncertainty_scores[mask]
            scenario_noise = noise_stds[mask]
            
            print(f"\n   {scenario}:")
            print(f"     Mean uncertainty: {np.mean(scenario_uncertainty):.3f}")
            print(f"     Mean noise std: {np.mean(scenario_noise):.3f}")
            print(f"     Noise reduction: {1 - np.mean(scenario_noise)/np.mean(noise_stds):.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Adaptive noise demonstration failed: {e}")
        return False


def main():
    """Main test execution."""
    print("🎯 CP+BBE Integration Testing")
    print("=" * 60)
    
    if not CP_BBE_AVAILABLE:
        print("❌ CP+BBE modules not available. Please install TorchCP: pip install torchcp")
        return 1
    
    test_results = []
    
    # Run individual tests
    test_results.append(("Conformal Sensitivity", test_conformal_sensitivity()))
    test_results.append(("Multi-Judge Conformal", test_multi_judge_conformal()))
    test_results.append(("Unified CP+BBE Framework", test_unified_cp_bbe_framework()))
    test_results.append(("DifferentialDebias Integration", test_integration_with_differential_debias()))
    test_results.append(("Adaptive Noise Benefits", demonstrate_adaptive_noise_benefits()))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    passed = sum(result for _, result in test_results)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 All tests passed! CP+BBE integration is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())
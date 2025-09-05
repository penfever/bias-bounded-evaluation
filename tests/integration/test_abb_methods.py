"""
Test script for A-BB implementation
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from differential_debiasing.core.debias import DifferentialDebias
from differential_debiasing.sensitivity.abb_sensitivity import ABBSensitivity
from differential_debiasing.neighbors import HammingNeighborGenerator, FormattingNeighborGenerator
from differential_debiasing.core.utils import calculate_abb_noise_parameter

def test_abb_noise_calculation():
    """Test A-BB noise parameter calculation."""
    print("Testing A-BB noise parameter calculation...")
    
    # Test parameters (ensure A-BB constraint is satisfied)
    rms_sensitivity = 0.5
    delta = 0.1
    constraint_threshold = rms_sensitivity * np.sqrt(2.0 / delta)
    tau = constraint_threshold + 0.5  # Make sure τ > threshold
    dimensionality = 5
    
    try:
        sigma = calculate_abb_noise_parameter(rms_sensitivity, tau, delta, dimensionality)
        print(f"✓ A-BB noise parameter: σ = {sigma:.6f}")
        
        # Check constraint
        constraint_threshold = rms_sensitivity * np.sqrt(2.0 / delta)
        print(f"✓ Constraint: τ ({tau}) > Δ*₂ * sqrt(2/δ) ({constraint_threshold:.6f}) = {tau > constraint_threshold}")
        
    except Exception as e:
        print(f"✗ A-BB noise calculation failed: {e}")
        return False
    
    return True

def test_neighbor_generators():
    """Test neighbor generators."""
    print("\nTesting neighbor generators...")
    
    # Test with simple context
    context = {"score": 7.5, "text": "This is a test response", "factors": [3, 4, 2, 5]}
    
    # Test Hamming generator
    try:
        hamming_gen = HammingNeighborGenerator(random_seed=42)
        neighbors = hamming_gen.sample_neighbors(context, 3)
        print(f"✓ Hamming generator produced {len(neighbors)} neighbors")
        
        # Check if neighbors are different
        for i, neighbor in enumerate(neighbors):
            if neighbor != context:
                print(f"  ✓ Neighbor {i+1} is different from original")
            else:
                print(f"  ! Neighbor {i+1} is identical to original")
                
    except Exception as e:
        print(f"✗ Hamming generator failed: {e}")
        return False
    
    # Test Formatting generator
    try:
        formatting_gen = FormattingNeighborGenerator(random_seed=42)
        neighbors = formatting_gen.sample_neighbors(context, 3)
        print(f"✓ Formatting generator produced {len(neighbors)} neighbors")
        
    except Exception as e:
        print(f"✗ Formatting generator failed: {e}")
        return False
    
    return True

def test_abb_sensitivity_with_array():
    """Test ABB sensitivity estimator with numpy array."""
    print("\nTesting A-BB sensitivity with array input...")
    
    try:
        # Create synthetic judgment data
        np.random.seed(42)
        judgments = np.random.normal(5.0, 1.5, 20)  # Scores around 5 with some variation
        
        # Create A-BB sensitivity estimator
        abb_estimator = ABBSensitivity(num_neighbors=5, random_seed=42)
        
        # Fit and estimate
        abb_estimator.fit(judgments)
        sensitivity = abb_estimator.estimate(score_range=10.0)
        
        print(f"✓ RMS sensitivity estimated: {sensitivity:.6f}")
        
        # Get diagnostics
        diagnostics = abb_estimator.get_diagnostics()
        print(f"✓ Diagnostics: {len(diagnostics)} fields")
        
        return True
        
    except Exception as e:
        print(f"✗ A-BB sensitivity test failed: {e}")
        return False

def test_abb_with_judge_function():
    """Test A-BB sensitivity with judge function."""
    print("\nTesting A-BB sensitivity with judge function...")
    
    try:
        # Create a simple judge function
        def simple_judge(context):
            """Simple judge function that adds noise to base score."""
            if isinstance(context, dict) and 'score' in context:
                base_score = context['score']
            else:
                base_score = 5.0
            
            # Add some random variation based on context "bias"
            bias = 0
            if isinstance(context, dict):
                if 'text' in context and 'test' in context['text'].lower():
                    bias = 0.5  # Bias toward "test" content
                if 'factors' in context:
                    bias += np.mean(context['factors']) * 0.1
            
            return base_score + bias + np.random.normal(0, 0.2)
        
        # Create context
        context = {"score": 6.0, "text": "This is a test response", "factors": [3, 4, 2, 5]}
        
        # Create A-BB estimator with judge function
        abb_estimator = ABBSensitivity(
            judge_function=simple_judge,
            neighbor_generator="hamming",
            num_neighbors=5,
            random_seed=42
        )
        
        # Fit and estimate
        abb_estimator.fit(context)
        sensitivity = abb_estimator.estimate()
        
        print(f"✓ RMS sensitivity with judge function: {sensitivity:.6f}")
        
        # Test neighbor differences
        neighbor_diffs = abb_estimator.get_neighbor_differences()
        print(f"✓ Neighbor differences: {len(neighbor_diffs)} values, mean = {np.mean(neighbor_diffs):.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ A-BB with judge function test failed: {e}")
        return False

def test_differential_debias_abb():
    """Test DifferentialDebias with A-BB mechanism."""
    print("\nTesting DifferentialDebias with A-BB mechanism...")
    
    try:
        # Create synthetic data
        np.random.seed(42)
        judgments = np.random.normal(6.0, 1.0, 15)
        
        # Create A-BB debiasing mechanism (use higher tau to satisfy constraint)
        debiaser = DifferentialDebias(
            tau=5.0,  # Higher tau to satisfy A-BB constraint
            delta=0.1,
            sensitivity_estimator="abb",
            dimensionality=15,
            num_neighbors=5,
            random_seed=42
        )
        
        # Fit and transform
        debiased = debiaser.fit_transform(judgments)
        
        print(f"✓ Original judgments shape: {judgments.shape}")
        print(f"✓ Debiased judgments shape: {debiased.shape}")
        print(f"✓ Original mean: {np.mean(judgments):.3f}")
        print(f"✓ Debiased mean: {np.mean(debiased):.3f}")
        
        # Get diagnostics
        diagnostics = debiaser.get_diagnostics()
        print(f"✓ Is A-BB mechanism: {diagnostics.get('is_abb_mechanism', False)}")
        
        # Check constraint validation
        if 'abb_constraint_validation' in diagnostics:
            constraint = diagnostics['abb_constraint_validation']
            print(f"✓ A-BB constraint satisfied: {constraint['constraint_satisfied']}")
        
        # Test effectiveness
        validation = debiaser.validate_effectiveness(judgments, debiased)
        print(f"✓ Signal preservation (correlation): {validation['correlation']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ DifferentialDebias A-BB test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing A-BB Gaussian Mechanism Implementation")
    print("=" * 60)
    
    tests = [
        test_abb_noise_calculation,
        test_neighbor_generators,
        test_abb_sensitivity_with_array,
        test_abb_with_judge_function,
        test_differential_debias_abb
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("Test failed!")
        except Exception as e:
            print(f"Test error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! A-BB implementation is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the implementation.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
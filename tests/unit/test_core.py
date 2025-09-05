#!/usr/bin/env python3
"""
Test script for the new sensitivity estimators implementation
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def create_test_data(n_samples=50, seed=42):
    """Create sample judgment data for testing."""
    np.random.seed(seed)
    
    base_quality = np.random.normal(7, 1.5, n_samples)
    data = {
        'correctness_score': np.clip(base_quality + np.random.normal(0, 0.5, n_samples), 1, 10),
        'completeness_score': np.clip(base_quality + np.random.normal(0, 0.8, n_samples), 1, 10),
        'safety_score': np.clip(base_quality + np.random.normal(0, 0.3, n_samples), 1, 10),
        'conciseness_score': np.clip(np.random.normal(6, 1.2, n_samples), 1, 10),
        'style_score': np.clip(base_quality + np.random.normal(0, 0.6, n_samples), 1, 10),
    }
    
    # Generate overall score with some relationship to factors plus noise
    overall = (0.3 * data['correctness_score'] + 0.25 * data['completeness_score'] +
              0.2 * data['safety_score'] + 0.1 * data['conciseness_score'] + 
              0.15 * data['style_score'] + np.random.normal(0, 1, n_samples))
    data['overall_score'] = np.clip(overall, 1, 10)
    
    return pd.DataFrame(data)

def test_psychometric_reliability():
    """Test PsychometricReliabilitySensitivity estimator."""
    print("🧪 Testing Psychometric Reliability Sensitivity...")
    
    try:
        from sensitivity.psychometric_reliability import PsychometricReliabilitySensitivity
        
        # Create test data
        df = create_test_data()
        
        # Initialize and fit estimator
        estimator = PsychometricReliabilitySensitivity()
        estimator.fit(df)
        
        # Estimate sensitivity
        sensitivity = estimator.estimate(score_range=9.0)
        
        # Get diagnostics
        diagnostics = estimator.get_diagnostics()
        
        print(f"   ✓ Fitted successfully")
        print(f"   ✓ Sensitivity: {sensitivity:.4f}")
        print(f"   ✓ Reliability Score: {diagnostics['reliability_score']:.4f}")
        print(f"   ✓ Cronbach's Alphas: {len(diagnostics['cronbach_alphas'])} factors")
        print(f"   ✓ Quality: {diagnostics['quality_indicators']['overall_quality']}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_schematic_adherence():
    """Test SchematicAdherenceSensitivity estimator."""
    print("🧪 Testing Schematic Adherence Sensitivity...")
    
    try:
        from sensitivity.schematic_adherence import SchematicAdherenceSensitivity
        
        # Create test data
        df = create_test_data()
        
        # Initialize and fit estimator
        estimator = SchematicAdherenceSensitivity()
        estimator.fit(df)
        
        # Estimate sensitivity
        sensitivity = estimator.estimate(score_range=9.0)
        
        # Get diagnostics
        diagnostics = estimator.get_diagnostics()
        
        print(f"   ✓ Fitted successfully")
        print(f"   ✓ Sensitivity: {sensitivity:.4f}")
        print(f"   ✓ Linear R²: {diagnostics['linear_r2']:.4f}")
        print(f"   ✓ Polynomial R²: {diagnostics['polynomial_r2']:.4f}")
        print(f"   ✓ Quality: {diagnostics['quality_indicators']['overall_quality']}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_combined_sensitivity():
    """Test CombinedSensitivity estimator."""
    print("🧪 Testing Combined Sensitivity...")
    
    try:
        from sensitivity.combined import CombinedSensitivity
        
        # Create test data
        df = create_test_data()
        
        # Initialize and fit estimator
        estimator = CombinedSensitivity(alpha=0.6)  # Favor psychometric slightly
        estimator.fit(df)
        
        # Estimate sensitivity
        sensitivity = estimator.estimate(score_range=9.0)
        
        # Get diagnostics
        diagnostics = estimator.get_diagnostics()
        components = estimator.get_component_sensitivities()
        
        print(f"   ✓ Fitted successfully")
        print(f"   ✓ Combined Sensitivity: {sensitivity:.4f}")
        print(f"   ✓ Psychometric Sensitivity: {components['psychometric_sensitivity']:.4f}")
        print(f"   ✓ Schematic Sensitivity: {components['schematic_sensitivity']:.4f}")
        print(f"   ✓ Alpha (weighting): {estimator.get_alpha()}")
        print(f"   ✓ Quality: {diagnostics['quality_indicators']['overall_quality']}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_differential_debias_integration():
    """Test integration with DifferentialDebias class."""
    print("🧪 Testing DifferentialDebias Integration...")
    
    try:
        from debias import DifferentialDebias
        
        # Create test data
        df = create_test_data()
        
        # Test all three new estimators
        estimators = [
            'psychometric_reliability',
            'schematic_adherence', 
            'combined'
        ]
        
        results = {}
        
        for est_name in estimators:
            print(f"   Testing {est_name}...")
            
            # Initialize debiaser
            debiaser = DifferentialDebias(
                tau=0.4,
                delta=0.05,
                sensitivity_estimator=est_name
            )
            
            # Fit and transform 
            # For psychometric/combined/schematic estimators, need to pass the full DataFrame
            if est_name in ['psychometric_reliability', 'combined', 'schematic_adherence']:
                debiaser.fit(df)
            else:
                # For others, can fit with just the target scores
                debiaser.fit(df['overall_score'].values)
            
            # Transform some scores
            original_scores = df['overall_score'].values
            debiased_scores = debiaser.transform(original_scores)
            
            # Get diagnostics
            diagnostics = debiaser.get_diagnostics()
            bias_bounds = debiaser.get_bias_bounds(len(original_scores))
            
            # Validate effectiveness
            validation = debiaser.validate_effectiveness(original_scores, debiased_scores)
            
            results[est_name] = {
                'sensitivity': diagnostics['bias_sensitivity'],
                'correlation': validation['correlation'],
                'noise_level': validation['noise_level']
            }
            
            print(f"     ✓ Processed {len(debiased_scores)} scores")
            print(f"     ✓ Correlation: {validation['correlation']:.3f}")
            print(f"     ✓ Noise level: {validation['noise_level']:.3f}")
        
        print(f"   ✓ All integrations successful!")
        return True, results
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def main():
    """Run all tests."""
    print("🚀 Testing New Bias Sensitivity Estimators Implementation\n")
    
    results = []
    
    # Test individual estimators
    results.append(test_psychometric_reliability())
    print()
    
    results.append(test_schematic_adherence()) 
    print()
    
    results.append(test_combined_sensitivity())
    print()
    
    # Test integration
    integration_success, integration_results = test_differential_debias_integration()
    results.append(integration_success)
    print()
    
    # Summary
    print("📊 Test Summary:")
    test_names = [
        "Psychometric Reliability",
        "Schematic Adherence", 
        "Combined Sensitivity",
        "DifferentialDebias Integration"
    ]
    
    for i, (name, success) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {name}: {status}")
    
    overall_success = all(results)
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if integration_success and integration_results:
        print("\n📈 Integration Results:")
        for est_name, metrics in integration_results.items():
            print(f"   {est_name}:")
            print(f"     Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"     Correlation: {metrics['correlation']:.3f}")
            print(f"     Noise Level: {metrics['noise_level']:.3f}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
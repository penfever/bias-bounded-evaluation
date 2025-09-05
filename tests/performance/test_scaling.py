#!/usr/bin/env python3
"""
Test sensitivity scaling fix
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from differential_debiasing.sensitivity.schematic_adherence import SchematicAdherenceSensitivity
from differential_debiasing.sensitivity.psychometric_reliability import PsychometricReliabilitySensitivity

def create_test_data():
    """Create test data with known variance patterns."""
    np.random.seed(42)
    n_samples = 100
    
    # Create factor scores on 1-5 scale
    correctness = np.random.uniform(1, 5, n_samples)
    completeness = np.random.uniform(1, 5, n_samples)
    style = np.random.uniform(1, 5, n_samples)
    
    # Create overall score with strong relationship to factors (high R²)
    overall_high_r2 = 0.8 * correctness + 0.2 * completeness + np.random.normal(0, 0.5, n_samples)
    overall_high_r2 = np.clip(overall_high_r2, 1, 5)
    
    # Create overall score with weak relationship (low R²)  
    overall_low_r2 = 0.2 * correctness + 0.1 * completeness + np.random.normal(0, 1.5, n_samples)
    overall_low_r2 = np.clip(overall_low_r2, 1, 5)
    
    return {
        'high_r2': pd.DataFrame({
            'correctness_score': correctness,
            'completeness_score': completeness, 
            'style_score': style,
            'overall_score': overall_high_r2
        }),
        'low_r2': pd.DataFrame({
            'correctness_score': correctness,
            'completeness_score': completeness,
            'style_score': style, 
            'overall_score': overall_low_r2
        })
    }

def test_scaling():
    """Test sensitivity scaling with different R² values."""
    test_data = create_test_data()
    
    print("🔍 Testing sensitivity scaling fixes...")
    print("=" * 60)
    
    for scenario, df in test_data.items():
        print(f"\n📊 {scenario.upper()} SCENARIO:")
        print(f"Data range: {df['overall_score'].min():.2f} to {df['overall_score'].max():.2f}")
        
        # Test schematic adherence
        try:
            sa_estimator = SchematicAdherenceSensitivity(
                factor_columns=['correctness_score', 'completeness_score', 'style_score'],
                target_column='overall_score'
            )
            sa_estimator.fit(df)
            
            # Test with different score ranges
            for score_range in [None, 4.0, 1.0]:
                sensitivity = sa_estimator.estimate(score_range)
                r2 = sa_estimator.get_schematic_r2()
                unexplained_var = 1.0 - r2
                
                print(f"  Schematic Adherence (range={score_range}):")
                print(f"    R² = {r2:.3f}, Unexplained = {unexplained_var:.1%}")
                print(f"    Sensitivity = {sensitivity:.4f}")
                print(f"    Expected range: {np.sqrt(unexplained_var) * (score_range or 4.0):.4f}")
                
        except Exception as e:
            print(f"  Schematic Adherence failed: {e}")
        
        # Test psychometric reliability
        try:
            pr_estimator = PsychometricReliabilitySensitivity()
            pr_estimator.fit(df)
            
            for score_range in [None, 4.0, 1.0]:
                sensitivity = pr_estimator.estimate(score_range)
                reliability = pr_estimator.get_reliability_score()
                
                print(f"  Psychometric Reliability (range={score_range}):")
                print(f"    Reliability = {reliability:.3f}")
                print(f"    Sensitivity = {sensitivity:.4f}")
                
        except Exception as e:
            print(f"  Psychometric Reliability failed: {e}")
            
        print()

if __name__ == "__main__":
    test_scaling()
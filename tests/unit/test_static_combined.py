#!/usr/bin/env python3
"""
Test Combined A-BB with both static and dynamic measurements
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import sys
sys.path.append(str(Path(__file__).parent.parent))
from test_data_helpers import load_and_prepare_score_data, create_synthetic_judge_function
from differential_debiasing.sensitivity.combined_abb_sensitivity import CombinedABBSensitivity

def test_combined_static_dynamic():
    """Test Combined A-BB with both static and dynamic measurements."""
    print("🔍 Testing Combined A-BB with static + dynamic measurements...")
    
    # Load some real data
    base_path = Path("/Users/benjaminfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Papers/bias-bounded-evaluation/sos-addl-data/InDepthAnalysis")
    df = load_and_prepare_score_data("QwQ-32B-setting1", base_path)
    
    print(f"Loaded {len(df)} real judge evaluations")
    
    # Initialize Combined A-BB with both static and dynamic
    combined_estimator = CombinedABBSensitivity(
        static_estimators=['psychometric_reliability', 'schematic_adherence'], 
        dynamic_generators=['hamming', 'formatting'],
        combination_strategy='conservative',
        num_neighbors=5,  # Fewer neighbors for faster testing
        judge_function=create_synthetic_judge_function(),
        random_seed=42
    )
    
    # Fit with factor columns
    factor_columns = [col for col in df.columns if col.endswith('_score') and col != 'overall_score']
    combined_estimator.fit(df, factor_columns=factor_columns)
    
    # Get sensitivity estimate (should show both static and dynamic)
    print("\nGetting sensitivity estimates...")
    sensitivity = combined_estimator.estimate(score_range=4.0)
    
    print(f"\nFinal combined sensitivity: {sensitivity:.4f}")
    
    # Get measurement breakdown
    breakdown = combined_estimator.get_measurement_breakdown()
    print("\nMeasurement breakdown:")
    print(f"  Static measurements: {breakdown['static_measurements']}")  
    print(f"  Dynamic measurements: {breakdown['dynamic_measurements']}")
    print(f"  Strategy: {breakdown['combination_strategy']}")
    
    return sensitivity

if __name__ == "__main__":
    test_combined_static_dynamic()
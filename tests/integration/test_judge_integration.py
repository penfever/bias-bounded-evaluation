#!/usr/bin/env python3
"""
Test real judge A-BB sensitivity measurement with a small dataset
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

def test_real_judge_abb():
    """Test Combined A-BB with real judge on small dataset."""
    print("🔍 Testing Real Judge A-BB Sensitivity Measurement...")
    
    # Load a small subset of real data
    base_path = Path("/Users/benjaminfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Papers/bias-bounded-evaluation/sos-addl-data/InDepthAnalysis")
    df = load_and_prepare_score_data("QwQ-32B-setting1", base_path)
    
    print(f"Loaded {len(df)} real judge evaluations")
    
    # Take a very small sample for cost management
    small_df = df.sample(n=50, random_state=42).copy()
    print(f"Testing with {len(small_df)} samples for cost management")
    
    # Create real judge function with very low budget
    print("Creating real judge function...")
    judge_function = create_judge_function(
        'gpt-3.5-turbo-0125',
        cost_budget_usd=1.0,  # Very conservative budget
        cache_responses=True
    )
    
    # Initialize Combined A-BB with minimal neighbors for testing
    print("Initializing Combined A-BB estimator...")
    combined_estimator = CombinedABBSensitivity(
        static_estimators=['psychometric_reliability', 'schematic_adherence'], 
        dynamic_generators=['hamming'],  # Only hamming for testing
        combination_strategy='conservative',
        num_neighbors=2,  # Very small for cost management
        judge_function=judge_function,
        random_seed=42
    )
    
    # Fit with factor columns
    print("Fitting estimator...")
    factor_columns = [col for col in small_df.columns if col.endswith('_score') and col != 'overall_score']
    try:
        combined_estimator.fit(small_df, factor_columns=factor_columns)
        
        # Get sensitivity estimate
        print("Getting sensitivity estimates...")
        sensitivity = combined_estimator.estimate(score_range=4.0)
        
        print(f"\n🎯 Results:")
        print(f"Final combined sensitivity: {sensitivity:.4f}")
        
        # Get measurement breakdown
        breakdown = combined_estimator.get_measurement_breakdown()
        print(f"\nMeasurement breakdown:")
        for measurement_type in ['static_measurements', 'dynamic_measurements']:
            measurements = breakdown.get(measurement_type, {})
            if measurements:
                print(f"  {measurement_type}:")
                for name, value in measurements.items():
                    if value is not None:
                        print(f"    {name}: {value:.4f}")
        
        # Get judge cost information
        if hasattr(judge_function, 'interface'):
            cost_summary = judge_function.interface.get_cost_summary()
            print(f"\n💰 Judge Costs:")
            print(f"  Spent: ${cost_summary['spent_usd']:.4f}")
            print(f"  Queries cached: {cost_summary['queries_cached']}")
            print(f"  Budget remaining: ${cost_summary['budget_remaining']:.4f}")
            
        return sensitivity, breakdown, cost_summary
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        # Still show cost summary
        if hasattr(judge_function, 'interface'):
            cost_summary = judge_function.interface.get_cost_summary()
            print(f"💰 Judge Costs (partial): ${cost_summary['spent_usd']:.4f}")
        
        return None, None, None

if __name__ == "__main__":
    print("This will make real API calls and incur costs!")
    confirmation = input("Continue? (y/N): ").strip().lower()
    if confirmation in ['y', 'yes']:
        test_real_judge_abb()
    else:
        print("Test cancelled.")
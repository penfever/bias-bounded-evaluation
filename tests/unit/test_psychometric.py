#!/usr/bin/env python3
"""
Test psychometric reliability calculation details
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
from test_data_helpers import load_and_prepare_score_data
from differential_debiasing.sensitivity.psychometric_reliability import PsychometricReliabilitySensitivity

def test_psychometric_details():
    """Test psychometric reliability calculation with detailed diagnostics."""
    print("🔍 Testing psychometric reliability calculation details...")
    
    # Load real data
    base_path = Path("/Users/anonymous/Library/CloudStorage/GoogleDrive-anon@gmail.com/My Drive/Current Papers/bias-bounded-evaluation/sos-addl-data/InDepthAnalysis")
    df = load_and_prepare_score_data("QwQ-32B-setting1", base_path)
    
    print(f"Loaded {len(df)} real judge evaluations")
    
    # Initialize psychometric estimator
    pr_estimator = PsychometricReliabilitySensitivity()
    
    # Fit 
    pr_estimator.fit(df)
    
    # Get detailed diagnostics
    diagnostics = pr_estimator.get_diagnostics()
    reliability_score = pr_estimator.get_reliability_score()
    
    print(f"\n📊 Psychometric Analysis Results:")
    print(f"Reliability score: {reliability_score:.4f}")
    
    # Show calculation steps
    normalized_sensitivity = np.sqrt(1.0 - reliability_score)
    scaled_sensitivity = normalized_sensitivity * 4.0
    
    print(f"\n🧮 Calculation steps:")
    print(f"1. Reliability score (R): {reliability_score:.4f}")
    print(f"2. Unreliability (1-R): {1.0 - reliability_score:.4f}")
    print(f"3. Normalized sensitivity sqrt(1-R): {normalized_sensitivity:.4f}")
    print(f"4. 4-point scale sensitivity: {scaled_sensitivity:.4f}")
    
    # Get component breakdowns
    cronbach_alphas = pr_estimator.get_cronbach_alphas()
    clr_scores = pr_estimator.get_clr_scores()
    
    print(f"\n📋 Component Analysis:")
    if cronbach_alphas:
        print(f"Cronbach's alpha scores: {cronbach_alphas}")
        avg_alpha = np.mean(list(cronbach_alphas.values()))
        print(f"Average Cronbach's alpha: {avg_alpha:.4f}")
    
    if clr_scores:
        print(f"Cross-loading ratios: {clr_scores}")
        avg_clr = np.mean(list(clr_scores.values()))
        print(f"Average CLR: {avg_clr:.4f}")
    
    # Show HTMT matrix
    htmt_matrix = pr_estimator.get_htmt_matrix()
    if htmt_matrix is not None:
        print(f"HTMT matrix shape: {htmt_matrix.shape}")
        print(f"HTMT max: {np.max(htmt_matrix):.4f}")
        print(f"HTMT mean (off-diagonal): {np.mean(htmt_matrix[htmt_matrix != 1.0]):.4f}")
    
    # Show quality indicators
    quality = diagnostics.get('quality_indicators', {})
    print(f"\n⭐ Quality indicators: {quality}")
    
    return reliability_score, scaled_sensitivity

if __name__ == "__main__":
    test_psychometric_details()
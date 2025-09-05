#!/usr/bin/env python3
"""
Test sigmoid CLR normalization implementation
"""

import numpy as np

def test_sigmoid_clr():
    """Test the sigmoid CLR normalization against expected values."""
    
    # Our observed CLR values
    clr_values = [2.0, 0.05394, 0.10043, 0.0, 0.0]
    clr_names = ['correctness', 'completeness', 'safety', 'conciseness', 'style']
    
    print("🔍 Sigmoid CLR Normalization Test")
    print("=" * 50)
    
    # Apply sigmoid normalization: 1.0 / (1.0 + exp(-2*(clr - 1.5)))
    sigmoid_values = [1.0 / (1.0 + np.exp(-2*(clr - 1.5))) for clr in clr_values]
    
    print("CLR → Sigmoid Transformation:")
    for name, clr, sigmoid in zip(clr_names, clr_values, sigmoid_values):
        print(f"  {name:12s}: {clr:6.3f} → {sigmoid:.4f}")
    
    clr_component = np.mean(sigmoid_values)
    print(f"\nMean CLR component: {clr_component:.4f}")
    
    # Other components (from previous test)
    alpha_component = 0.7100  # Mean Cronbach's alpha
    htmt_component = 0.6387   # 1 - mean_HTMT
    
    # Final reliability calculation (equal weights)
    reliability_score = (alpha_component + clr_component + htmt_component) / 3
    sensitivity = np.sqrt(1 - reliability_score) * 4.0
    
    print(f"\nFinal Calculation:")
    print(f"  Alpha component:  {alpha_component:.4f}")
    print(f"  CLR component:    {clr_component:.4f}") 
    print(f"  HTMT component:   {htmt_component:.4f}")
    print(f"  Reliability (R):  {reliability_score:.4f}")
    print(f"  Sensitivity:      {sensitivity:.4f}")
    
    # Show sigmoid curve behavior at key thresholds
    print(f"\nSigmoid Curve at Key Thresholds:")
    test_clrs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    for test_clr in test_clrs:
        sigmoid_val = 1.0 / (1.0 + np.exp(-2*(test_clr - 1.5)))
        print(f"  CLR = {test_clr:.1f} → {sigmoid_val:.4f}")

if __name__ == "__main__":
    test_sigmoid_clr()
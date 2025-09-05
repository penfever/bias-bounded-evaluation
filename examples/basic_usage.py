#!/usr/bin/env python3
"""
Example usage of differential debiasing
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from debias import DifferentialDebias
from config import ConfigManager
from sensitivity import FactorAnalysisSensitivity, DomainSpecificSensitivity


def example_basic_usage():
    """Basic usage example with synthetic data."""
    print("=" * 50)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 50)
    
    # Create synthetic judgment data
    np.random.seed(42)
    n_samples = 100
    
    # Simulate biased judgments (systematic bias toward higher scores)
    true_scores = np.random.normal(6, 2, n_samples)
    bias_pattern = np.random.exponential(1, n_samples)  # Positive bias
    biased_scores = true_scores + bias_pattern
    biased_scores = np.clip(biased_scores, 1, 10)
    
    print(f"Original scores: mean={np.mean(biased_scores):.3f}, std={np.std(biased_scores):.3f}")
    
    # Apply differential debiasing
    debias = DifferentialDebias(tau=0.5, delta=0.05)
    debiased_scores = debias.fit_transform(biased_scores)
    
    print(f"Debiased scores: mean={np.mean(debiased_scores):.3f}, std={np.std(debiased_scores):.3f}")
    print(f"Correlation: {np.corrcoef(biased_scores, debiased_scores)[0,1]:.3f}")
    
    # Get bias protection info
    bias_bounds = debias.get_bias_bounds(n_samples)
    print(f"Bias protection: {bias_bounds['protection_guarantee']}")
    print(f"Noise level: σ={bias_bounds['noise_std']:.3f}")
    

def example_factor_analysis():
    """Example using factor analysis with structured data."""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Factor Analysis Method")
    print("=" * 50)
    
    # Create synthetic factor-based data
    np.random.seed(42)
    n_samples = 200
    
    # Generate factor scores
    completeness = np.random.normal(7, 1.5, n_samples)
    correctness = np.random.normal(6.5, 1.2, n_samples)
    conciseness = np.random.normal(6, 1.8, n_samples)
    safety = np.random.normal(8, 0.8, n_samples)
    style = np.random.normal(6.2, 1.6, n_samples)
    
    # Create overall score with bias (not fully explained by factors)
    explained_part = 0.3 * completeness + 0.25 * correctness + 0.2 * conciseness + 0.15 * safety + 0.1 * style
    bias_part = np.random.normal(0, 2, n_samples)  # Unexplained bias
    overall_score = explained_part + bias_part
    overall_score = np.clip(overall_score, 1, 10)
    
    # Create DataFrame
    df = pd.DataFrame({
        'completeness_score': completeness,
        'correctness_score': correctness,
        'conciseness_score': conciseness,
        'safety_score': safety,
        'style_score': style,
        'score': overall_score
    })
    
    print(f"Created dataset with {len(df)} samples")
    
    # Apply factor analysis debiasing
    debias = DifferentialDebias(
        tau=0.5,
        delta=0.05,
        sensitivity_estimator="factor_analysis",
        factor_columns=['completeness_score', 'correctness_score', 'conciseness_score', 'safety_score', 'style_score'],
        target_column='score'
    )
    
    debiased_scores = debias.fit_transform(df)
    
    print(f"Original scores: mean={np.mean(overall_score):.3f}, std={np.std(overall_score):.3f}")
    print(f"Debiased scores: mean={np.mean(debiased_scores):.3f}, std={np.std(debiased_scores):.3f}")
    
    # Get diagnostics
    diagnostics = debias.get_diagnostics()
    r_squared = diagnostics['sensitivity_estimator'].get('r_squared', 'N/A')
    bias_sensitivity = diagnostics['bias_sensitivity']
    
    print(f"R² (explained variance): {r_squared:.3f}")
    print(f"Bias sensitivity: {bias_sensitivity:.3f}")
    
    # Validation
    validation = debias.validate_effectiveness(overall_score, debiased_scores)
    print(f"Signal preservation: {validation['signal_preservation']:.3f}")
    print(f"Noise level: {validation['noise_level']:.3f}")


def example_domain_specific():
    """Example using domain-specific sensitivity."""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Domain-Specific Method")
    print("=" * 50)
    
    # Test different domains
    domains = ['medical', 'academic', 'creative', 'general']
    
    np.random.seed(42)
    scores = np.random.normal(7, 2, 50)
    scores = np.clip(scores, 1, 10)
    
    print(f"Original scores: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
    
    for domain in domains:
        debias = DifferentialDebias(
            tau=0.5,
            delta=0.05,
            sensitivity_estimator="domain_specific",
            domain=domain,
            scale_type="1-10"
        )
        
        debiased_scores = debias.fit_transform(scores)
        noise_level = debias.get_bias_bounds(len(scores))['noise_std']
        
        print(f"{domain:>10}: noise_std={noise_level:.3f}, "
              f"debiased_mean={np.mean(debiased_scores):.3f}")


def example_config_system():
    """Example using configuration system."""
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Configuration System")
    print("=" * 50)
    
    config_manager = ConfigManager()
    
    # List available presets
    print("Available presets:")
    for preset in config_manager.list_presets():
        print(f"  - {preset}")
    
    # Use a preset
    config = config_manager.get_preset('conservative')
    print(f"\nUsing 'conservative' preset:")
    print(f"  tau={config.tau}, delta={config.delta}")
    print(f"  sensitivity_method={config.sensitivity_method}")
    
    # Create debiasing instance from config
    np.random.seed(42)
    scores = np.random.normal(6, 2, 100)
    scores = np.clip(scores, 1, 10)
    
    debias = DifferentialDebias(
        tau=config.tau,
        delta=config.delta,
        sensitivity_estimator=config.sensitivity_method,
        use_average_case=config.use_average_case
    )
    
    debiased_scores = debias.fit_transform(scores)
    
    print(f"Result: correlation={np.corrcoef(scores, debiased_scores)[0,1]:.3f}")
    print(f"Bias protection: {debias.get_bias_bounds(len(scores))['protection_guarantee']}")


def example_sensitivity_comparison():
    """Compare different sensitivity estimation methods."""
    print("\n" + "=" * 50)
    print("EXAMPLE 5: Sensitivity Method Comparison")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    n_samples = 150
    
    # Create DataFrame with factor structure
    df = pd.DataFrame({
        'completeness_score': np.random.normal(7, 1.5, n_samples),
        'correctness_score': np.random.normal(6.5, 1.2, n_samples),
        'conciseness_score': np.random.normal(6, 1.8, n_samples),
        'score': np.random.normal(6.5, 2, n_samples)
    })
    df['score'] = np.clip(df['score'], 1, 10)
    
    methods = ['factor_analysis', 'empirical', 'domain_specific', 'historical']
    
    print("Sensitivity estimates by method:")
    
    for method in methods:
        try:
            if method == 'factor_analysis':
                debias = DifferentialDebias(
                    sensitivity_estimator=method,
                    factor_columns=['completeness_score', 'correctness_score', 'conciseness_score']
                )
            else:
                debias = DifferentialDebias(sensitivity_estimator=method)
            
            debias.fit(df)
            sensitivity = debias.get_diagnostics()['bias_sensitivity']
            
            print(f"  {method:>18}: {sensitivity:.3f}")
            
        except Exception as e:
            print(f"  {method:>18}: Failed ({e})")


if __name__ == "__main__":
    print("Differential Debiasing Examples")
    print("=" * 50)
    
    example_basic_usage()
    example_factor_analysis()
    example_domain_specific()
    example_config_system()
    example_sensitivity_comparison()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)
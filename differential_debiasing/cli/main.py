#!/usr/bin/env python3
"""
Command-line interface for differential debiasing
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from ..core.debias import DifferentialDebias
from ..core.utils import validate_input_array


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """Save list of dictionaries to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def extract_scores_from_jsonl(data: List[Dict[str, Any]], 
                             score_field: str = 'score') -> np.ndarray:
    """Extract scores from JSONL data."""
    scores = []
    for item in data:
        if score_field in item:
            score = item[score_field]
            if isinstance(score, (int, float)):
                scores.append(float(score))
            elif isinstance(score, str):
                # Try to parse string scores
                try:
                    scores.append(float(score))
                except ValueError:
                    # Skip invalid scores
                    continue
    
    if not scores:
        raise ValueError(f"No valid scores found in field '{score_field}'")
    
    return np.array(scores)


def apply_debiasing_to_jsonl(data: List[Dict[str, Any]], 
                           debiased_scores: np.ndarray,
                           score_field: str = 'score',
                           debiased_field: str = 'debiased_score') -> List[Dict[str, Any]]:
    """Apply debiased scores back to JSONL data."""
    result = []
    score_idx = 0
    
    for item in data:
        new_item = item.copy()
        
        if score_field in item:
            original_score = item[score_field]
            if isinstance(original_score, (int, float, str)):
                try:
                    # Validate original score
                    float(original_score)
                    # Add debiased score
                    new_item[debiased_field] = float(debiased_scores[score_idx])
                    score_idx += 1
                except (ValueError, IndexError):
                    # Skip invalid scores
                    pass
        
        result.append(new_item)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Apply differential debiasing to LLM judge evaluations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument('input', help='Input JSONL file with judgments')
    parser.add_argument('output', help='Output JSONL file with debiased scores')
    
    # Debiasing parameters
    parser.add_argument('--tau', type=float, default=0.5,
                       help='Bias protection parameter (lower = stronger protection)')
    parser.add_argument('--delta', type=float, default=0.05,
                       help='Failure probability (0 < delta < 1)')
    
    # Sensitivity estimation
    parser.add_argument('--sensitivity-method', choices=['schematic_adherence', 'psychometric_reliability', 'fixed'],
                       default='schematic_adherence', help='Bias sensitivity estimation method')
    parser.add_argument('--sensitivity-value', type=float,
                       help='Explicit sensitivity value (used when method is fixed)')
    
    # Data processing
    parser.add_argument('--score-field', default='score',
                       help='Field name containing scores to debias')
    parser.add_argument('--debiased-field', default='debiased_score',
                       help='Field name for debiased scores')
    parser.add_argument('--factor-columns', nargs='+',
                       help='Factor score column names (for schematic_adherence / psychometric methods)')
    parser.add_argument('--target-column', default='overall_score',
                       help='Target score column name (for schematic_adherence method)')
    
    # Options
    parser.add_argument('--no-average-case', action='store_true',
                       help='Disable average-case optimization')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--diagnostics', action='store_true',
                       help='Save diagnostics to JSON file')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load data
        if args.verbose:
            print(f"Loading data from {args.input}")
        data = load_jsonl(args.input)
        print(f"Loaded {len(data)} records")
        
        # Extract scores
        if args.verbose:
            print(f"Extracting scores from field '{args.score_field}'")
        scores = extract_scores_from_jsonl(data, args.score_field)
        print(f"Found {len(scores)} valid scores")
        
        if len(scores) == 0:
            print("Error: No valid scores found", file=sys.stderr)
            sys.exit(1)
        
        # Set up sensitivity estimator kwargs
        estimator_kwargs = {}
        if args.sensitivity_method in ('schematic_adherence', 'psychometric_reliability'):
            if args.factor_columns:
                estimator_kwargs['factor_columns'] = args.factor_columns
            estimator_kwargs['target_column'] = args.target_column
        if args.sensitivity_method == 'fixed' and args.sensitivity_value is not None:
            estimator_kwargs['fixed_sensitivity_value'] = args.sensitivity_value
        
        # Create debiasing mechanism
        if args.verbose:
            print(f"Creating debiasing mechanism with τ={args.tau}, δ={args.delta}")
        
        debias = DifferentialDebias(
            tau=args.tau,
            delta=args.delta,
            sensitivity_estimator=args.sensitivity_method,
            use_average_case=not args.no_average_case,
            random_seed=args.seed,
            **estimator_kwargs
        )
        
        # Fit and transform
        if args.verbose:
            print("Fitting debiasing mechanism")
        
        # For static estimators that rely on factor columns, we need the full DataFrame
        if args.sensitivity_method in ('schematic_adherence', 'psychometric_reliability') and len(data) > 0:
            # Convert to DataFrame so the estimator can access factor columns
            df = pd.DataFrame(data)
            debias.fit(df)
            debiased_scores = debias.transform(scores)
        else:
            # For other methods, scores are sufficient
            debiased_scores = debias.fit_transform(scores)
        
        print(f"Applied debiasing to {len(debiased_scores)} scores")
        
        # Apply debiased scores back to data
        if args.verbose:
            print("Applying debiased scores to data")
        
        result_data = apply_debiasing_to_jsonl(
            data, debiased_scores, args.score_field, args.debiased_field
        )
        
        # Save results
        if args.verbose:
            print(f"Saving results to {args.output}")
        
        save_jsonl(result_data, args.output)
        print(f"Saved debiased data to {args.output}")
        
        # Save diagnostics if requested
        if args.diagnostics:
            diagnostics_path = Path(args.output).with_suffix('.diagnostics.json')
            diagnostics = debias.get_diagnostics()
            
            # Add validation metrics
            validation = debias.validate_effectiveness(scores, debiased_scores)
            diagnostics['validation'] = validation
            
            # Add bias bounds
            bias_bounds = debias.get_bias_bounds(len(scores))
            diagnostics['bias_bounds'] = bias_bounds
            
            with open(diagnostics_path, 'w') as f:
                json.dump(diagnostics, f, indent=2, default=str)
            
            print(f"Saved diagnostics to {diagnostics_path}")
        
        # Print summary
        print("\nSummary:")
        print(f"  Original scores: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
        print(f"  Debiased scores: mean={np.mean(debiased_scores):.3f}, std={np.std(debiased_scores):.3f}")
        print(f"  Correlation: {np.corrcoef(scores, debiased_scores)[0,1]:.3f}")
        
        # Show bias protection info
        bias_bounds = debias.get_bias_bounds(len(scores))
        print(f"  Bias protection: {bias_bounds['protection_guarantee']}")
        print(f"  Noise level: σ={bias_bounds['noise_std']:.3f}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

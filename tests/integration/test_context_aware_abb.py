#!/usr/bin/env python3
"""
Test Context-Aware Dynamic A-BB with Real Judge Querying

This script demonstrates the new context-aware A-BB sensitivity measurement
that operates on semantic content rather than just numerical scores.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import sys
sys.path.append(str(Path(__file__).parent.parent))
from test_data_helpers import load_and_prepare_score_data, create_synthetic_judge_function
from differential_debiasing.sensitivity.combined_abb_sensitivity import CombinedABBSensitivity


def load_arena_hard_context_data(judge_name: str, base_path: Path, max_samples: int = 20) -> pd.DataFrame:
    """Load Arena-Hard evaluation data with full semantic context."""
    print(f"Loading Arena-Hard context data for {judge_name}...")
    
    # Find the judge directory
    judge_dir = base_path / judge_name
    if not judge_dir.exists():
        raise ValueError(f"Judge directory not found: {judge_dir}")
    
    # Load evaluations from base_processed directory
    base_processed_dir = judge_dir / "base_processed"
    if not base_processed_dir.exists():
        raise ValueError(f"Base processed directory not found: {base_processed_dir}")
    
    # Load judge evaluations with full context
    evaluations = load_judge_evaluations(base_processed_dir)
    if not evaluations:
        raise ValueError(f"No evaluations found in {base_processed_dir}")
    
    print(f"Loaded {len(evaluations)} model evaluation files")
    
    # Convert to DataFrame with semantic context
    context_data = []
    sample_count = 0
    
    for model_name, model_evaluations in evaluations.items():
        if sample_count >= max_samples:
            break
            
        for eval_data in model_evaluations:
            if sample_count >= max_samples:
                break
                
            question_id = eval_data.get('question_id', '')
            games = eval_data.get('games', [])
            
            for game in games[:1]:  # Take first game for simplicity
                if sample_count >= max_samples:
                    break
                    
                # Extract user prompt and model responses
                user_prompt = game.get('user_prompt', '')
                
                # Parse user prompt to extract question and answers
                if '<|User Prompt|>' in user_prompt:
                    # Extract the actual question
                    question_start = user_prompt.find('<|User Prompt|>') + len('<|User Prompt|>\n')
                    question_end = user_prompt.find('<|The Start of Assistant A')
                    if question_end == -1:
                        question = user_prompt[question_start:].strip()
                        answer = "No answer provided"
                    else:
                        question = user_prompt[question_start:question_end].strip()
                        
                        # Extract Assistant A's answer
                        answer_start = user_prompt.find('<|The Start of Assistant A\'s Answer|>') + len('<|The Start of Assistant A\'s Answer|>\n')
                        answer_end = user_prompt.find('<|The End of Assistant A\'s Answer|>')
                        if answer_start > 0 and answer_end > answer_start:
                            answer = user_prompt[answer_start:answer_end].strip()
                        else:
                            answer = "Could not extract answer"
                else:
                    question = user_prompt
                    answer = "Could not parse structure"
                
                # Get scores from the game
                overall_score = float(game.get('score', 3.0)) if 'score' in game else 3.0
                
                # Create context record
                context_record = {
                    'question_id': question_id,
                    'model': model_name,
                    'question': question,
                    'answer': answer,
                    'overall_score': overall_score,
                    'correctness_score': float(game.get('correctness_score', overall_score)),
                    'completeness_score': float(game.get('completeness_score', overall_score)),
                    'safety_score': float(game.get('safety_score', overall_score)),
                    'conciseness_score': float(game.get('conciseness_score', overall_score)),
                    'style_score': float(game.get('style_score', overall_score))
                }
                
                context_data.append(context_record)
                sample_count += 1
    
    df = pd.DataFrame(context_data)
    
    print(f"Extracted {len(df)} evaluation contexts with semantic content")
    print(f"Sample question: {df.iloc[0]['question'][:100]}...")
    print(f"Sample answer: {df.iloc[0]['answer'][:100]}...")
    
    return df


def test_context_aware_dynamic_abb():
    """Test Combined A-BB with context-aware dynamic measurements."""
    print("🚀 Testing Context-Aware Dynamic A-BB")
    print("=" * 60)
    
    # Configuration
    USE_REAL_JUDGE = True  # Test with real judge
    JUDGE_NAME = "gpt-3.5-turbo-0125"  # Cheaper model for testing
    BUDGET_USD = 2.0  # Small budget for testing
    
    if USE_REAL_JUDGE:
        print(f"⚠️  REAL JUDGE MODE: Will query {JUDGE_NAME} API")
        print(f"⚠️  This will incur API costs (budget: ${BUDGET_USD})!")
        
        # Ask for confirmation
        confirmation = input("Continue with real judge querying? (y/N): ").strip().lower()
        if confirmation not in ['y', 'yes']:
            print("Aborting test.")
            return
    
    # Base path for score data
    base_path = Path("/Users/anonymous/Library/CloudStorage/GoogleDrive-anon@gmail.com/My Drive/Current Papers/bias-bounded-evaluation/sos-addl-data/InDepthAnalysis")
    
    # Load Arena-Hard context data
    try:
        context_df = load_arena_hard_context_data("QwQ-32B-setting1", base_path, max_samples=15)
        print(f"✅ Loaded {len(context_df)} contexts with semantic content")
    except Exception as e:
        print(f"❌ Failed to load context data: {e}")
        return
    
    # Create real judge function
    if USE_REAL_JUDGE:
        print(f"Creating real judge function for {JUDGE_NAME}...")
        try:
            judge_function = create_judge_function(
                JUDGE_NAME,
                cost_budget_usd=BUDGET_USD,
                cache_responses=True
            )
            print("✅ Judge function created successfully")
        except Exception as e:
            print(f"❌ Failed to create judge function: {e}")
            return
    else:
        # Create mock judge function
        def mock_judge(context):
            base_score = 3.0
            if isinstance(context, dict):
                if 'overall_score' in context:
                    base_score = float(context['overall_score'])
            return np.array([base_score + np.random.normal(0, 0.2)])
        judge_function = mock_judge
        print("✅ Using mock judge function")
    
    # Initialize Combined A-BB with context-aware generators
    print("Initializing Combined A-BB with context-aware generators...")
    try:
        combined_estimator = CombinedABBSensitivity(
            static_estimators=['psychometric_reliability', 'schematic_adherence'],
            dynamic_generators=['question_paraphrasing', 'answer_perturbation', 'model_obfuscation'],
            combination_strategy='conservative',
            num_neighbors=2,  # Small for testing
            judge_function=judge_function,
            random_seed=42
        )
        print("✅ Combined A-BB estimator created with context-aware generators")
        
        # Show available generators
        print("📋 Available generators:")
        for name, generator in combined_estimator.dynamic_generators.items():
            print(f"  • {name}: {generator.get_description()}")
            
    except Exception as e:
        print(f"❌ Failed to create Combined A-BB estimator: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Fit with context data
    print("Fitting Combined A-BB on context data...")
    try:
        # Use factor columns for static measurements
        factor_columns = [col for col in context_df.columns if col.endswith('_score') and col != 'overall_score']
        print(f"Factor columns: {factor_columns}")
        
        combined_estimator.fit(context_df, factor_columns=factor_columns)
        print("✅ Combined A-BB fitted successfully")
        
    except Exception as e:
        print(f"❌ Failed to fit Combined A-BB: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get sensitivity estimate
    print("Computing sensitivity estimates...")
    try:
        sensitivity = combined_estimator.estimate(score_range=4.0)
        print(f"✅ Combined sensitivity: {sensitivity:.4f}")
        
        # Get detailed breakdown
        breakdown = combined_estimator.get_measurement_breakdown()
        
        print("\n📊 Measurement Breakdown:")
        for measurement_type in ['static_measurements', 'dynamic_measurements']:
            measurements = breakdown.get(measurement_type, {})
            if measurements:
                print(f"\n  {measurement_type.replace('_', ' ').title()}:")
                for name, value in measurements.items():
                    if value is not None:
                        print(f"    • {name}: {value:.4f}")
        
        print(f"\n🔧 Combination strategy: {breakdown.get('combination_strategy')}")
        
        # Show judge cost information if using real judge
        if USE_REAL_JUDGE and hasattr(judge_function, 'interface'):
            cost_summary = judge_function.interface.get_cost_summary()
            print(f"\n💰 Judge Costs:")
            print(f"  • Spent: ${cost_summary['spent_usd']:.4f}")
            print(f"  • Queries cached: {cost_summary['queries_cached']}")
            print(f"  • Budget remaining: ${cost_summary['budget_remaining']:.4f}")
        
        print(f"\n🎯 Final Results:")
        print(f"  • Combined A-BB sensitivity: {sensitivity:.4f}")
        print(f"  • This represents the system-level bias sensitivity")
        print(f"  • Incorporates both psychometric and dynamic semantic measurements")
        
        return sensitivity, breakdown
        
    except Exception as e:
        print(f"❌ Failed to compute sensitivity: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    result = test_context_aware_dynamic_abb()
    if result[0] is not None:
        print("\n✅ Context-aware dynamic A-BB test completed successfully!")
    else:
        print("\n❌ Test failed!")
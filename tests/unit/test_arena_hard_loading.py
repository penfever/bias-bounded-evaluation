#!/usr/bin/env python3
"""
Test script to verify Arena-Hard data loading works correctly.
"""

import sys
from pathlib import Path
import pandas as pd

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from differential_debiasing.interfaces.oumi_interface import _load_judgment_data, _parse_arena_hard_prompt

def test_data_loading():
    """Test loading Arena-Hard judgment data."""
    print("🔍 Testing Arena-Hard data loading...")
    
    base_dir = "/Users/benjaminfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Papers/bias-bounded-evaluation/sos-addl-data/InDepthAnalysis"
    
    # Test loading a specific judgment
    question_id = "328c149ed45a41c0b9d6f14659e63599"
    model_name = "gpt-3.5-turbo-0125"
    
    try:
        print(f"Loading: question_id={question_id}, model={model_name}")
        judgment_data = _load_judgment_data(base_dir, question_id, model_name)
        print(f"✅ Successfully loaded judgment data")
        print(f"Keys: {list(judgment_data.keys())}")
        
        # Test parsing the prompt
        user_prompt = judgment_data['games'][0]['user_prompt']
        print(f"User prompt length: {len(user_prompt)} characters")
        
        question, answer_a, answer_b = _parse_arena_hard_prompt(user_prompt)
        print(f"✅ Successfully parsed prompt")
        print(f"Question: {question[:100]}...")
        print(f"Answer A: {answer_a[:100]}...")
        print(f"Answer B: {answer_b[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_with_sample_dataframe():
    """Test with a small sample DataFrame like the real analysis."""
    print("\n🔍 Testing with sample DataFrame...")
    
    # Create a small test DataFrame similar to what the analysis uses
    test_data = [
        {"question_id": "328c149ed45a41c0b9d6f14659e63599", "model": "gpt-3.5-turbo-0125"},
        {"question_id": "b43c07656ead4150b360294ee932b410", "model": "gpt-3.5-turbo-0125"},
    ]
    
    df = pd.DataFrame(test_data)
    print(f"Test DataFrame:\n{df}")
    
    base_dir = "/Users/benjaminfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Papers/bias-bounded-evaluation/sos-addl-data/InDepthAnalysis"
    
    for i, row in df.iterrows():
        question_id = row['question_id']
        model_name = row['model']
        
        try:
            judgment_data = _load_judgment_data(base_dir, question_id, model_name)
            user_prompt = judgment_data['games'][0]['user_prompt']
            question, answer_a, answer_b = _parse_arena_hard_prompt(user_prompt)
            
            print(f"✅ Row {i}: Successfully processed {question_id}")
            print(f"   Question: {question[:50]}...")
            print(f"   Answer A: {answer_a[:50]}...")
            print(f"   Answer B: {answer_b[:50]}...")
            
        except Exception as e:
            print(f"❌ Row {i}: Failed to process {question_id}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("🧪 Arena-Hard Data Loading Test")
    print("=" * 50)
    
    success1 = test_data_loading()
    success2 = test_with_sample_dataframe()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Arena-Hard integration is working.")
    else:
        print("\n💥 Some tests failed. Check the errors above.")
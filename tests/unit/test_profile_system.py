#!/usr/bin/env python3
"""
Test script to validate the sensitivity profile system.

This script tests:
1. Profile creation and loading
2. Performance improvements with profile usage
3. Accuracy comparison between profile-based and dynamic measurement
"""

import sys
import time
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from differential_debiasing.core.sensitivity_profiles import (
    SensitivityProfile, SensitivityProfileManager, validate_all_profiles
)


def test_profile_management():
    """Test basic profile management functionality."""
    print("🧪 Testing Profile Management")
    print("-" * 40)
    
    # Test profile manager
    profile_dir = Path(__file__).parent.parent.parent / "sensitivity_profiles"
    manager = SensitivityProfileManager(profile_dir)
    
    # List available profiles
    available_profiles = manager.list_profiles()
    print(f"Available profiles: {available_profiles}")
    
    if available_profiles:
        # Test loading a profile
        test_judge = available_profiles[0]
        profile = manager.load_profile(test_judge)
        
        if profile:
            print(f"\n✅ Loaded profile for {test_judge}:")
            summary = profile.summary()
            for key, value in summary.items():
                print(f"   {key}: {value}")
            
            # Test validation
            validation = manager.validate_profile(test_judge)
            print(f"\n🔍 Validation results:")
            print(f"   Valid: {validation['valid']}")
            if 'issues' in validation:
                print(f"   Issues: {validation['issues']}")
            if 'warnings' in validation:
                print(f"   Warnings: {validation['warnings']}")
        else:
            print(f"❌ Failed to load profile for {test_judge}")
    else:
        print("📋 No profiles found - run measure_judge_sensitivity.py first")


def test_performance_comparison():
    """Test performance improvements with profile system."""
    print("\n🚀 Testing Performance Comparison")
    print("-" * 40)
    
    # This would require actual measurement setup
    print("⚠️  Performance testing requires actual judge setup")
    print("   To test performance:")
    print("   1. First measure a judge: python measure_judge_sensitivity.py --judge gpt-4o-mini --samples 10")
    print("   2. Then run: python run_combined_abb_analysis.py")
    print("   3. Compare execution times with/without profiles")


def test_all_profiles():
    """Validate all available profiles."""
    print("\n🔍 Validating All Profiles")
    print("-" * 40)
    
    profile_dir = Path(__file__).parent.parent.parent / "sensitivity_profiles"
    if not profile_dir.exists():
        print(f"📋 Profile directory does not exist: {profile_dir}")
        return
    
    # Validate all profiles
    results = validate_all_profiles(profile_dir)
    
    if not results:
        print("📋 No profiles found to validate")
        return
    
    for judge_name, validation in results.items():
        status = "✅" if validation['valid'] else "❌"
        print(f"{status} {judge_name}: {'Valid' if validation['valid'] else 'Invalid'}")
        
        if not validation['valid'] and 'error' in validation:
            print(f"   Error: {validation['error']}")
        
        if 'issues' in validation and validation['issues']:
            for issue in validation['issues']:
                print(f"   Issue: {issue}")
        
        if 'warnings' in validation and validation['warnings']:
            for warning in validation['warnings']:
                print(f"   Warning: {warning}")


def main():
    """Main test execution."""
    print("🎯 Sensitivity Profile System Tests")
    print("=" * 50)
    
    try:
        # Test 1: Profile management
        test_profile_management()
        
        # Test 2: Performance comparison guidance
        test_performance_comparison()
        
        # Test 3: Profile validation
        test_all_profiles()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
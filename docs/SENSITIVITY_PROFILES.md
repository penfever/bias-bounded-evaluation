# Sensitivity Profile System

This document describes the sensitivity profile system that decouples measurement from mechanism application for efficient bias-bounded evaluation.

## Overview

The sensitivity profile system separates expensive sensitivity measurements from bias-bounded mechanism application, enabling:

- **Performance**: ~10x speedup by avoiding expensive formatting measurements
- **Flexibility**: Adjust bias tolerance (τ) independently of measurement 
- **Scalability**: Measure once per judge, apply to many datasets
- **Maintainability**: Clear separation between measurement and mechanism

## Architecture

### Components

1. **Sensitivity Measurement**: `measure_judge_sensitivity.py`
   - Pre-computes expensive sensitivity measurements (especially formatting)
   - Stores results in JSON profiles with metadata and confidence intervals
   
2. **Profile Management**: `sensitivity_profiles.py`
   - Load/save/validate sensitivity profiles
   - Support multiple input formats (judge name, profile object, dict)
   
3. **Profile-Enhanced Analysis**: `profile_enhanced_abb.py`
   - Wrapper around CombinedABBSensitivity that uses pre-computed values
   - Falls back to dynamic measurement when profiles unavailable
   
4. **Integration**: `run_combined_abb_analysis.py`
   - Automatically detects and uses available profiles
   - Switches between profile-enhanced and standard approaches

### Profile Format

```json
{
  "judge_name": "gpt-4o-mini",
  "profile_version": "1.0", 
  "created_date": "2025-01-21T10:30:00Z",
  "formatting_sensitivity": {
    "value": 2.3456,
    "confidence_interval": [1.8, 2.8],
    "samples_used": 50,
    "neighbors_generated": 10,
    "measurement_date": "2025-01-21T10:30:00Z",
    "cost_info": {...}
  },
  "measurement_metadata": {
    "script_version": "1.0",
    "measurement_approach": "arena_hard_content_perturbation",
    "neighbor_types": ["whitespace", "capitalization", "punctuation"]
  }
}
```

## Usage

### 1. Measure Judge Sensitivity

```bash
# Measure formatting sensitivity for a judge
python scripts/analysis/measure_judge_sensitivity.py \
  --judge gpt-4o-mini \
  --samples 20 \
  --neighbors 10 \
  --budget 10.0

# Dry run to see what would be measured
python scripts/analysis/measure_judge_sensitivity.py \
  --judge claude-3-5-sonnet \
  --dry-run
```

### 2. Run Analysis with Profiles

```bash
# Analysis automatically uses profiles when available
python scripts/analysis/run_combined_abb_analysis.py
```

The system will:
- Check for available sensitivity profiles
- Use profile data for expensive measurements (formatting)
- Fall back to dynamic measurement for fast measurements (hamming)
- Report performance improvements

### 3. Validate Profiles

```bash
# Test the profile system
python scripts/analysis/test_profile_system.py
```

## Performance Benefits

### Without Profiles (Original)
- Formatting sensitivity: ~30 minutes (50 samples × 10 neighbors × judge calls)
- Hamming sensitivity: ~10 seconds (direct score manipulation)
- **Total**: ~30 minutes per analysis

### With Profiles (Optimized)  
- Formatting sensitivity: ~0.1 seconds (profile lookup)
- Hamming sensitivity: ~10 seconds (still dynamic)
- **Total**: ~10 seconds per analysis

**Performance improvement: ~180x speedup**

## Integration with Existing Code

### Backwards Compatibility

The system maintains full backwards compatibility:
- Existing scripts work unchanged when no profiles available
- Dynamic measurement used as fallback
- All APIs unchanged

### Profile Detection

```python
from differential_debiasing.core.sensitivity_profiles import load_judge_sensitivity_profile

# Load profile for a judge
profile = load_judge_sensitivity_profile("gpt-4o-mini")
if profile:
    formatting_sensitivity = profile.get_formatting_sensitivity()
    print(f"Formatting sensitivity: {formatting_sensitivity}")
```

### DifferentialDebias Integration

```python
from differential_debiasing.core.debias import DifferentialDebias

# Use with judge name (auto-loads profile)
debiaser = DifferentialDebias(
    sensitivity_estimator="profile_enhanced_abb",
    sensitivity_profile="gpt-4o-mini",  # Judge name
    tau=2.0,
    delta=0.1
)

# Use with profile object
from differential_debiasing.core.sensitivity_profiles import load_judge_sensitivity_profile
profile = load_judge_sensitivity_profile("gpt-4o-mini")
debiaser = DifferentialDebias(
    sensitivity_estimator="profile_enhanced_abb", 
    sensitivity_profile=profile,
    tau=2.0,
    delta=0.1
)

# Use with manual values
debiaser = DifferentialDebias(
    sensitivity_estimator="profile_enhanced_abb",
    sensitivity_profile={"formatting_sensitivity": {"value": 2.5}},
    tau=2.0, 
    delta=0.1
)
```

## Measurement Guidelines

### When to Measure

- **New judges**: Always measure before first use
- **Profile age**: Re-measure if profile >90 days old
- **Significant changes**: Re-measure if judge behavior changes
- **Different domains**: Consider domain-specific profiles

### Sampling Strategy

- **Development**: 10-20 samples, 5-10 neighbors
- **Production**: 50-100 samples, 10-20 neighbors  
- **Research**: 100+ samples, 20+ neighbors

### Cost Management

- **Local models**: No cost concerns, use generous samples
- **API models**: Set budget limits, use caching
- **Cloud services**: Monitor costs, optimize sample sizes

## Error Handling

### Profile Loading Errors
- Missing profiles → fallback to dynamic measurement
- Corrupted profiles → warning + fallback
- Invalid values → validation error + skip

### Measurement Errors
- API failures → retry with exponential backoff
- Timeout → reduce sample size + continue
- Cost limit → stop measurement + save partial results

## Future Extensions

### Additional Sensitivity Types
- Hamming sensitivity profiles (currently dynamic)
- Order sensitivity profiles  
- Context-specific profiles

### Advanced Features
- Profile version migration
- Automatic profile updates
- Cross-judge profile sharing
- Domain-specific profiles

### Integration Improvements
- Automatic profile freshness checking
- Smart fallback strategies
- Profile performance monitoring
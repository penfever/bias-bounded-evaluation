# Differential Debiasing for LLM Judges

A Python implementation of differential privacy-inspired techniques for reducing implicit bias in LLM judge evaluations while providing formal guarantees.

## Overview

This library provides tools to:
- **Detect implicit bias** in LLM judge evaluations using multiple sensitivity estimation methods
- **Apply calibrated noise** to mask bias patterns while preserving legitimate signal
- **Provide formal guarantees** that systematic bias patterns are indistinguishable from random noise
- **Maintain evaluation utility** through average-case optimizations for datasets

## Key Features

### 🔒 Formal Bias Guarantees
- **Mathematically proven** bias bounds using differential privacy techniques
- **Configurable protection levels** via τ (bias tolerance) and δ (failure probability)
- **Composable guarantees** for multi-judge scenarios

### 📊 Multiple Sensitivity Estimation Methods
1. **Factor Analysis**: Uses R² to estimate unexplained variance as bias (recommended)
2. **Empirical**: Direct measurement from controlled bias trigger variations
3. **Cross-Validation**: Systematic testing with bias-inducing modifications
4. **Historical**: Analysis of past judgment inconsistencies
5. **Domain-Specific**: Expert-defined bounds for different evaluation contexts

### ⚡ Average-Case Optimization
- **Dataset-level guarantees** with 1/√n noise scaling
- **Better utility** than individual judgment protection
- **Suitable for typical evaluation scenarios**

### 🛠️ Easy Integration
- **Command-line interface** for batch processing JSONL files
- **Python API** for programmatic use
- **Configuration system** with presets for common scenarios

## Quick Start

### Installation

```bash
# From source (recommended for development)
git clone https://github.com/penfever/differential-debiasing.git
cd differential-debiasing
pip install -e .

# Or install from PyPI (when available)
pip install differential-debiasing
```

### Basic Usage

```python
import numpy as np
from differential_debiasing import DifferentialDebias

# Your biased judgment scores
biased_scores = np.array([7.2, 8.1, 6.5, 9.0, 5.8, 7.8, 6.2, 8.5])

# Apply differential debiasing
debias = DifferentialDebias(tau=0.5, delta=0.05)
debiased_scores = debias.fit_transform(biased_scores)

# Get bias protection info
bias_bounds = debias.get_bias_bounds(len(biased_scores))
print(f"Protection: {bias_bounds['protection_guarantee']}")
print(f"Noise level: {bias_bounds['noise_std']:.3f}")
```

### Command-Line Usage

```bash
# Apply debiasing to JSONL file
differential-debias input.jsonl output.jsonl \
    --tau 0.5 --delta 0.05 \
    --sensitivity-method factor_analysis \
    --verbose --diagnostics

# Use domain-specific sensitivity
differential-debias input.jsonl output.jsonl \
    --sensitivity-method domain_specific \
    --domain medical --scale-type 1-10
```

## Sensitivity Estimation Methods

### Factor Analysis (Recommended)
Uses R² from linear regression to estimate bias sensitivity as √(1 - R²):

```python
from differential_debiasing import DifferentialDebias

debias = DifferentialDebias(
    sensitivity_estimator="factor_analysis",
    factor_columns=['completeness_score', 'correctness_score', 'conciseness_score'],
    target_column='score'
)
```

### Empirical Measurement
Directly measures bias from controlled variations:

```python
debias = DifferentialDebias(
    sensitivity_estimator="empirical",
    bias_triggers=['verbose_vs_concise', 'formal_vs_casual']
)
```

### Domain-Specific Bounds
Uses expert knowledge for different evaluation contexts:

```python
debias = DifferentialDebias(
    sensitivity_estimator="domain_specific",
    domain="medical",  # Options: medical, academic, creative, etc.
    scale_type="1-10"
)
```

## Configuration System

### Using Presets

```python
from differential_debiasing.config import ConfigManager

config_manager = ConfigManager()

# List available presets
print(config_manager.list_presets())
# ['conservative', 'moderate', 'permissive', 'academic', 'medical', ...]

# Use a preset
config = config_manager.get_preset('conservative')  # tau=0.3, delta=0.01
debias = DifferentialDebias(tau=config.tau, delta=config.delta)
```

### Custom Configuration

```yaml
# config.yaml
tau: 0.5
delta: 0.05
sensitivity_method: factor_analysis
factor_columns: ['completeness_score', 'correctness_score']
target_column: score
use_average_case: true
random_seed: 42
```

```python
config = config_manager.load_config('config.yaml')
```

## Understanding the Parameters

### Bias Protection (τ - tau)
Controls how much bias is tolerated:
- **τ = 0.3**: Strong protection (bias patterns ≤ 1.35× random chance)
- **τ = 0.5**: Moderate protection (bias patterns ≤ 1.65× random chance)  
- **τ = 0.7**: Weak protection (bias patterns ≤ 2.0× random chance)

### Failure Probability (δ - delta)
Probability that bias protection fails:
- **δ = 0.01**: Very high confidence (99% guarantee)
- **δ = 0.05**: High confidence (95% guarantee)
- **δ = 0.1**: Moderate confidence (90% guarantee)

### Sensitivity Estimation Impact
Higher sensitivity estimates → more noise → stronger protection but lower utility.

## Real-World Example

Using data from the paper showing 93% unexplained variance in LLM judge evaluations:

```python
# Your LLM judge data with massive implicit bias
# R² = 0.068 (only 6.8% explained by explicit factors!)

debias = DifferentialDebias(
    tau=0.5,           # Moderate protection
    delta=0.05,        # 95% confidence
    sensitivity_estimator="factor_analysis"
)

# For 500 judgments, transforms:
# - Bias sensitivity: 8.7/10 points → Average-case noise: 0.39 points
# - Formal guarantee: Systematic bias > 50% strength occurs with p < 0.05
# - Utility preservation: ~0.85 correlation with original rankings
```

## Advanced Usage

### Multi-Judge Composition

```python
# Multiple judges with composition guarantees
judge1_debias = DifferentialDebias(tau=0.5, delta=0.05)
judge2_debias = DifferentialDebias(tau=0.5, delta=0.05)

# Combined protection: (τ₁ + τ₂, δ₁ + δ₂) = (1.0, 0.1)
```

### Custom Sensitivity Estimator

```python
from differential_debiasing.sensitivity import SensitivityEstimator

class CustomSensitivity(SensitivityEstimator):
    def fit(self, judgments, **kwargs):
        # Your custom bias detection logic
        return self
    
    def estimate(self, score_range=None):
        # Return your bias sensitivity estimate
        return 3.0

debias = DifferentialDebias(sensitivity_estimator=CustomSensitivity())
```

### Validation and Diagnostics

```python
# Validate effectiveness
validation = debias.validate_effectiveness(original_scores, debiased_scores)
print(f"Signal preservation: {validation['signal_preservation']:.3f}")
print(f"Noise level: {validation['noise_level']:.3f}")

# Get detailed diagnostics
diagnostics = debias.get_diagnostics()
print(f"R² from factor analysis: {diagnostics['sensitivity_estimator']['r_squared']:.3f}")
print(f"Bias sensitivity: {diagnostics['bias_sensitivity']:.3f}")
```

## File Structure

```
differential-debiasing/
├── __init__.py              # Main exports
├── debias.py               # Core DifferentialDebias class
├── utils.py                # Utility functions
├── config.py               # Configuration system
├── cli.py                  # Command-line interface
├── sensitivity/            # Sensitivity estimation methods
│   ├── __init__.py
│   ├── base.py            # Abstract base class
│   ├── factor_analysis.py # Factor analysis method
│   ├── empirical.py       # Empirical measurement
│   ├── cross_validation.py # Cross-validation approach
│   ├── historical.py      # Historical analysis
│   └── domain_specific.py # Domain-specific bounds
├── tests/                  # Test suite
├── examples/              # Usage examples
│   ├── example_usage.py   # Python examples
│   └── configs/          # Example configurations
└── README.md              # This file
```

## Mathematical Background

The method is based on the formal guarantee:

**For neighboring judgment contexts D and D', the mechanism M satisfies:**
```
Pr[M(D) ∈ S] ≤ e^τ × Pr[M(D') ∈ S] + δ
```

This ensures that systematic bias patterns stronger than τ are indistinguishable from calibrated Gaussian noise with probability 1-δ.

**Key insight**: Any "signal" that survives the noise threshold represents legitimate, non-biased evaluation differences.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `python -m pytest tests/`
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this library in your research, please cite:

```bibtex
@article{bias_bounded_llm_judges,
  title={Bias-Bounded LLM Judge Mechanisms: A Differential Privacy Approach},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## Oumi Batching & Remote Config

For API engines (OpenAI, Anthropic) the library uses Oumi’s `infer_online` with batching.

- Judge YAML keys (example `configs/judges/openai/gpt-4o-mini.yaml`):
  - `batch_size`: number of prompts per request batch (default 8)
  - `max_retries`: retry attempts on transient errors (default 3)
  - `retry_backoff_sec`: initial backoff seconds (exponential) (default 1.0)
  - `remote`: API params for Oumi RemoteParams
    - `api_key`: null to use environment; or set explicitly
    - `num_workers`: parallel workers (default 8)
    - `politeness_policy`: rate-limit pacing in seconds (default 60.0)

- Environment fallbacks:
  - `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
  - `OUMI_BATCH_SIZE`, `OUMI_MAX_RETRIES`, `OUMI_RETRY_BACKOFF_SEC`
  - `OUMI_NUM_WORKERS`, `OUMI_POLITENESS_SEC`

The `OumiJudgeInterface` builds `RemoteParams` from the YAML (or env vars) and issues batched `infer_online` calls with `InferenceConfig`. This reduces latency and handles rate limits more gracefully during sensitivity measurement.

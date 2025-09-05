# Factor Reliability Analysis for Arena-Hard-Auto

This README describes the factor reliability analysis tools for Arena-Hard-Auto, which provide a more sophisticated approach to assessing the quality of individual judgment factors and appropriately calculating confidence intervals.

## Overview

The factor reliability analysis addresses a key insight: the independence of judgment factors (like safety, correctness, etc.) is often desirable, not problematic. Instead of penalizing independence (as factor analysis might), we now:

1. Directly measure how consistently each factor is assessed across questions
2. Assess whether factors are appropriately distinct from each other
3. Use these measurements to adjust confidence intervals based on a factor's intrinsic reliability

## Key Metrics Calculated

The `factor_reliability.py` script calculates three important metrics:

1. **Cronbach's Alpha**: Measures internal consistency within a factor
   - Higher values (>0.7) indicate consistent assessment across questions
   - Lower values suggest inconsistent assessment or multi-dimensional factors

2. **Cross-loading Ratio**: Compares primary to secondary factor loadings
   - Higher values indicate factors are measuring distinct constructs
   - Lower values suggest factors may be redundant or confounded

3. **HTMT Ratio** (Heterotrait-Monotrait): More sophisticated measure of discriminant validity
   - Lower values (<0.85) indicate good discriminant validity
   - Higher values suggest factors are too closely related

These are combined into a single **Reliability Score** (0-1 scale) that represents the overall quality of each factor.

## Usage

### Step 1: Calculate Factor Reliability Metrics

```bash
# Basic usage
python factor_reliability.py /path/to/judgment/directory

# Quick mode (faster calculation with fewer bootstrap samples)
python factor_reliability.py /path/to/judgment/directory --quick

# Skip bootstrap confidence intervals for very fast calculation
python factor_reliability.py /path/to/judgment/directory --skip-bootstrap

# Example:
python factor_reliability.py llm-judge-oumi/DeepSeek-R1-32B-jr-new-bench/DeepSeek-R1_processed
```

This will generate two key files in the same directory:
- `factor_reliability_metrics.csv`: Contains all reliability metrics for each factor
- `factor_reliability_confidence_intervals.csv`: Bootstrap confidence intervals for the metrics

### Step 2: Use Reliability-Based Bootstrap for More Accurate Rankings

```bash
python show_result.py --bootstrap-method reliability --reliability-metrics-path /path/to/factor_reliability_metrics.csv

# Example:
python show_result.py --bootstrap-method reliability --reliability-metrics-path llm-judge-oumi/DeepSeek-R1-32B-jr-new-bench/DeepSeek-R1_processed/factor_reliability_metrics.csv
```

This approach adjusts confidence intervals based on the intrinsic reliability of each factor:
- Highly reliable factors get tighter confidence intervals
- Less reliable factors get wider confidence intervals

## How It Works

The reliability-based bootstrap differs from other methods by:

1. Measuring **intrinsic reliability** of each factor through psychometric techniques
2. Using a more intuitive relationship where reliability directly impacts confidence
3. Preserving formal statistical guarantees with an approach grounded in measurement theory

The uncertainty scaling is calculated as:
```
uncertainty_scale = 2.0 - reliability_score
```

This means:
- Perfect reliability (1.0) → No additional uncertainty (scale = 1.0)
- Zero reliability (0.0) → Doubled uncertainty (scale = 2.0)

## Interpreting Reliability Metrics

When examining the factor reliability metrics:

- **Reliability Score > 0.8**: Excellent - factor is consistent and distinct
- **Reliability Score 0.6-0.8**: Good - factor is generally reliable
- **Reliability Score 0.4-0.6**: Moderate - factor may need refinement
- **Reliability Score < 0.4**: Poor - factor may not be reliably measured

Low reliability doesn't necessarily mean the factor is "bad" - it may indicate limitations in how that particular aspect is currently being assessed.

## Example Workflow

1. Generate model answers and judgments as usual
2. Run factor analysis to understand the broad relationships between metrics
3. Run reliability analysis to assess the quality of each specific metric
4. Use the reliability-based bootstrap method to generate rankings with properly calibrated confidence intervals

This approach provides a more nuanced and theoretically sound method for estimating uncertainty in model rankings across different evaluation dimensions.
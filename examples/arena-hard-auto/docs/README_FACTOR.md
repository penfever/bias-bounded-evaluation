# Score Analysis Tools

This directory contains tools for analyzing score correlations and performing factor analysis on evaluation scores from language model outputs.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Correlation Analysis

The `get_corrs.py` script computes correlations between different scoring metrics and generates a correlation matrix and heatmap visualization.

### Usage

```bash
python get_corrs.py path/to/scores/directory
```

You can also run factor analysis together with correlation analysis:

```bash
# Run correlation and factor analysis
python get_corrs.py path/to/scores/directory --factor-analysis

# Run correlation, factor analysis, and question reliability analysis 
python get_corrs.py path/to/scores/directory --factor-analysis --analyze-questions

# Adjust the threshold for flagging low reliability questions (default: 2.0)
python get_corrs.py path/to/scores/directory --factor-analysis --analyze-questions --threshold 1.5
```

### Input

The script expects CSV files in the specified directory, each containing at least:
- A `model` column with model names
- One or more score columns (e.g., `score`, `correctness_score`, `style_score`, etc.)

### Output

The script produces:
- `score_correlations.csv`: A correlation matrix between all score columns
- `score_correlations_heatmap.png`: A heatmap visualization of the correlation matrix

## Factor Analysis

The `factor_analysis.py` script performs factor analysis on the score columns to identify underlying factor structure and can also identify questions where factor analysis fails to explain model performance differences.

### Usage

```bash
# Process JSONL files in the specified directory
python factor_analysis.py path/to/jsonl/directory

# Specify an output directory different from the input directory
python factor_analysis.py path/to/jsonl/directory --output-dir path/to/output/directory

# Specify minimum eigenvalue for factor retention
python factor_analysis.py path/to/jsonl/directory --min-eigenvalue 0.8

# Analyze question reliability - identify questions with high unexplained variance
python factor_analysis.py path/to/jsonl/directory --analyze-questions

# Adjust the threshold for flagging low reliability questions (default: 2.0)
# Lower values (e.g., 1.5) will flag more questions, higher values (e.g., 2.5) will flag fewer
python factor_analysis.py path/to/jsonl/directory --analyze-questions --threshold 1.5
```

### Input

The script can analyze two types of input data:

1. **JSONL files (primary)**: The script first tries to load JSONL files from the specified directory. These files should contain raw model judgments with score data in a structure similar to:
   ```json
   {
     "games": [
       {
         "question_id": "123",
         "score": "A>B",
         "correctness_score": "A>>B",
         "safety_score": "A=B",
         "completeness_score": "B>A",
         "conciseness_score": "A>B",
         "style_score": "A=B"
       },
       ...
     ]
   }
   ```

2. **CSV files (fallback)**: If no valid JSONL files are found, the script tries to load CSV files as with the `get_corrs.py` script.

### Output

The script produces:
- `factor_scree_plot.png`: A scree plot showing eigenvalues for each potential factor
- `factor_loadings.csv`: Factor loadings for each variable
- `factor_loadings_heatmap.png`: Heatmap visualization of factor loadings
- `factor_interpretation.png`: Horizontal bar chart showing the contribution of each variable to each factor
- `factor_variance.csv`: Variance explained by each factor
- `factor_communalities.csv`: Communalities for each variable
- `factor_biplot.png`: Biplot showing the relationship between factors and variables (if at least 2 factors)

The factor loadings have been implemented with automatic sign flipping. If the majority of loadings for a factor are negative, all signs for that factor will be flipped to make the interpretation more intuitive. This doesn't change the mathematical properties of the solution, only the presentation.

## Interpretation

### Correlation Analysis
- The correlation matrix shows how different scoring metrics relate to each other
- High positive correlations indicate metrics that tend to move together
- High negative correlations indicate metrics that tend to move in opposite directions

### Factor Analysis
- Factor analysis identifies underlying latent variables that explain patterns in the data
- Factor loadings show how strongly each score metric is associated with each factor
- Factors with eigenvalues > 0.75 are considered significant 
- Communalities show how much of each variable's variance is explained by the factors
- The biplot helps visualize the relationship between factors and variables

### Question Reliability Analysis
- When using `--analyze-questions`, the script identifies questions where the factor model fails to explain score patterns
- High residuals indicate questions where models perform differently than predicted by the factors
- Questions with high residuals might represent:
  - Tasks requiring novel capabilities not captured by existing factors
  - Questions with contradictory judgments across metrics
  - Potential evaluation gaps worth investigating
- The clustering approach groups similar high-residual questions to identify patterns

## Using Factor Analysis with Rankings

You can use the factor analysis results to improve confidence intervals in rankings in several ways:

### 1. Heuristic Adjustment (Simple)

```bash
# Adjust confidence intervals heuristically based on factor analysis results
python show_result.py --adjust-confidence
```

This widens confidence intervals for metrics with high unexplained variance using a simple multiplier. Quick but less rigorous.

### 2. Residual Resampling Bootstrap (Recommended)

```bash
# Use residual resampling bootstrap to formally incorporate unexplained variance
python show_result.py --bootstrap-method residual --factor-analysis-dir path/to/tables
```

Adds noise to bootstrap samples in proportion to unexplained variance. More statistically sound approach that maintains the formal guarantees of confidence intervals.

### 3. Bayesian Hierarchical Bootstrap

```bash
# Use Bayesian hierarchical model bootstrap
python show_result.py --bootstrap-method bayesian --factor-analysis-dir path/to/tables
```

Creates a multilevel model with inflated variance based on factor analysis. More sophisticated approach that models the uncertainty hierarchically.

## Example

```bash
# Run just correlation analysis
python get_corrs.py llm-judge-oumi/QwQ-32B-jr-new-bench/tables

# Run correlation analysis and factor analysis together
python get_corrs.py llm-judge-oumi/QwQ-32B-jr-new-bench/tables --factor-analysis

# Run just factor analysis
python factor_analysis.py llm-judge-oumi/QwQ-32B-jr-new-bench/tables
```
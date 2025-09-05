# Arena-Hard-Auto (Local Copy)

This directory contains a local copy of essential Arena-Hard-Auto files to make the differential-debiasing repository self-contained and reproducible.

## Purpose

This copy ensures that:
- The repository can be used without external dependencies on the original Arena-Hard-Auto installation
- Scripts and interfaces can reliably import necessary utilities 
- Reproducibility is maintained across different environments

## Contents

### Core Files
- `utils.py` - Core utility functions for judge evaluation
- `gen_answer.py` - Answer generation pipeline
- `gen_judgment.py` - Judge evaluation pipeline  
- `show_result.py` - Results display and analysis
- `factor_analysis.py` - Factor analysis for judge scores

### Configuration
- `config/` - Judge and API configuration files
  - `api_config.yaml` - API settings
  - `judge_config.yaml` - Judge prompt templates
  - Additional specialized configs

### Data
- `data/question.jsonl` - Arena-Hard evaluation questions

### Documentation  
- `docs/` - Additional documentation
- `LICENSE` - License information
- `requirements.txt` - Python dependencies

## Usage

The differential-debiasing package automatically uses this local copy when:
- Standard judge interfaces are created
- Evaluation scripts are run
- Arena-Hard utilities are needed

No manual configuration is required - the package paths are automatically updated to use this local copy.

## Original Source

This is a copy of the Arena-Hard-Auto framework from:
https://github.com/lm-sys/arena-hard-auto

See the original README.md for full documentation and usage instructions.

## Maintenance

This copy includes only the essential files needed for differential-debiasing functionality. For the complete Arena-Hard-Auto experience, please refer to the original repository.
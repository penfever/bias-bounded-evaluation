# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reference implementation of bias-bounded evaluation for LLM judges, applying differential privacy-inspired noise calibration to reduce systematic bias. Core parameters: `tau` (bias protection level) and `delta` (failure probability).

## Commands

```bash
pip install -e ".[dev]"          # Dev install
make test                        # All tests
make test-unit                   # Unit tests only
pytest tests/unit/test_debias.py # Single test file
pytest tests/unit/test_debias.py::TestClassName::test_name  # Single test
make lint                        # flake8
make format                      # black + isort
make type-check                  # mypy
make check-all                   # lint + type-check + test
```

## Code Style

- Black formatter, 100 char line length, isort with black profile
- Pre-commit hooks: black, isort, flake8, mypy (mypy runs against `differential_debiasing/` only)
- Python >=3.8

## Architecture

**Core pipeline** (`differential_debiasing/core/`):
- `debias.py` — `DifferentialDebias` orchestrator: normalizes scores, estimates sensitivity, calibrates noise (σ = Δ_B / (τ · Φ⁻¹(1-δ))), adds Gaussian noise, computes bias bounds
- `config.py` — `DebiasConfig` dataclass + `ConfigManager` with presets (schematic_default, psychometric_default, abb_default, etc.)
- `neighbors.py` — A-BB neighbor generators (Hamming, Formatting, Order) used for sensitivity measurement
- `sensitivity_profiles.py` — `SensitivityProfile`/`SensitivityProfileManager` for caching pre-computed judge sensitivities as JSON

**Sensitivity estimators** (`differential_debiasing/sensitivity/`): All extend `SensitivityEstimator` base class with `fit()`/`estimate()`/`fit_estimate()` interface:
- `psychometric_reliability.py` — Cronbach's α + CLR + HTMT → unified reliability score
- `schematic_adherence.py` — Linear/polynomial regression R² analysis
- `abb_sensitivity.py` — Dynamic A-BB mechanism using neighbor sampling and RMS sensitivity
- `combined_abb_sensitivity.py` — Hybrid static + dynamic estimation
- `fixed.py` — Pass-through for externally provided values

**Interfaces** (`differential_debiasing/interfaces/`): Arena-Hard dataset parsing, battle processing, ELO bootstrap, Oumi inference backend integration.

**CLI** (`differential_debiasing/cli/main.py`): JSONL processor — reads `score` field, writes `debiased_score`. Entry point: `differential-debias` command.

**Scripts** (`scripts/`): Analysis pipelines (`measure_judge_sensitivity.py`, `run_combined_abb_analysis.py`) and visualization tools (`create_comparisons.py`, ranking/ELO generation, scaling plots).

## Key Patterns

- Strategy pattern for sensitivity estimators — `DifferentialDebias._create_sensitivity_estimator()` creates from string name
- Profile-based caching separates sensitivity measurement from application (stored in `sensitivity_profiles/`)
- Scores are normalized to [0,1] before processing, then denormalized back

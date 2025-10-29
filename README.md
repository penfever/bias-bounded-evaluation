# Differential Debiasing for LLM Judges

This repository contains the reference implementation of the bias‑bounded evaluation pipeline used to audit and stabilize large language model (LLM) judges. It combines classical differential-privacy style guarantees with practical tooling for measuring judge sensitivity, applying calibrated noise, and visualising downstream impact.

## Getting Started

### Installation

```bash
git clone https://github.com/penfever/bias-bounded-evaluation.git
cd bias-bounded-evaluation
pip install -e ".[dev]"
```

### Quick Smoke Test

```bash
source ~/.zshrc && conda activate abb   # or any env with the deps above
python -m pytest -q
```

### Minimal Python Example

```python
import numpy as np
import pandas as pd
from differential_debiasing.core.debias import DifferentialDebias

judgments = pd.DataFrame({
    "score": np.array([7.2, 8.1, 6.5, 9.0, 5.8, 7.8, 6.2, 8.5]),
    "overall_score": np.array([7.2, 8.1, 6.5, 9.0, 5.8, 7.8, 6.2, 8.5]),
    "correctness_score": np.array([7.0, 8.0, 6.2, 8.7, 5.5, 7.5, 6.0, 8.0]),
    "clarity_score": np.array([7.5, 8.4, 6.8, 9.1, 6.0, 8.0, 6.4, 8.7]),
    "completeness_score": np.array([7.1, 8.2, 6.4, 8.9, 5.7, 7.7, 6.1, 8.4]),
})

debias = DifferentialDebias(
    tau=0.5,
    delta=0.05,
    sensitivity_estimator="schematic_adherence",
    factor_columns=["correctness_score", "clarity_score", "completeness_score"],
    target_column="overall_score",
    random_seed=42,
)

debiased_scores = debias.fit_transform(judgments)
print(debiased_scores)
print(debias.get_bias_bounds(len(judgments)))
```

### Command-Line Interface

The lightweight CLI handles JSONL inputs with numeric `score` fields:

```bash
python -m differential_debiasing.cli.main \
  input.jsonl output.jsonl \
  --tau 0.5 --delta 0.05 \
  --sensitivity-method schematic_adherence \
  --factor-columns completeness_score correctness_score conciseness_score \
  --target-column overall_score
```

Supported CLI estimators: `schematic_adherence`, `psychometric_reliability`, and `fixed`.

## Typical Workflow

1. **Measure judge sensitivity** – run `scripts/analysis/measure_judge_sensitivity.py` to cache formatting / hamming profiles for each judge you plan to debias.
2. **Run debiasing analysis** – use `scripts/analysis/run_combined_abb_analysis.py` (or the Python API) to apply Combined A‑BB or static estimators to your score tables.
3. **Inspect outputs** – visualise score deltas and ranking changes with the tools under `scripts/visualization/`.

See `scripts/README_scripts.md` for a narrative description of every helper script.

## Core Components

- `differential_debiasing/core` – the `DifferentialDebias` orchestration logic plus utilities, noise calibration, and profile management.
- `differential_debiasing/sensitivity` – individual sensitivity estimators (psychometric reliability, schematic adherence, and A‑BB variants).
- `differential_debiasing/interfaces` – integrations with Arena-Hard datasets, Oumi inference, and helper judge abstractions.
- `differential_debiasing/cli` – minimal JSONL CLI wrapper.
- `scripts/analysis` – end-to-end pipelines for measuring sensitivity and executing Combined A‑BB analyses.
- `scripts/dev` – ad hoc tooling such as the consistency probe for repeat-judgement variance checks.
- `scripts/visualization` – ranking/plotting utilities and orchestration scripts for generating publication figures.

## Repository Layout

```
bias-bounded-evaluation/
├── differential_debiasing/
│   ├── core/                 # Debiasing mechanism, profiles, utilities
│   ├── sensitivity/          # All estimator implementations
│   ├── interfaces/           # Arena-Hard + Oumi helpers
│   ├── neighbors.py          # A-BB neighbor generators
│   └── cli/                  # JSONL command-line interface
├── scripts/                  # Analysis, viz, and dev utilities (see README_scripts.md)
└── tests/                    # Pytest suite covering core estimators and utilities
```

## Testing & Development

- Run the full suite: `python -m pytest -q`
- Lint, format, and type-check: `make lint`, `make format`, `make type-check`
- Integration and sensitivity measurement scripts expect Arena-Hard style directories beneath `sos-addl-data/` or `sos-artifacts/`.

## License

MIT License – see the accompanying `LICENSE` file.

# Scripts Overview

## Analysis

### factor_reliability_robust.py
Implements robust psychometric diagnostics used by the Combined A‑BB pipeline. The script can be run standalone to compute Cronbach’s alpha, cross-loading ratios, HTMT matrices, and related reliability scores from Arena-Hard style factor tables, applying extensive safeguards for missing data, outliers, and over-correlation so that downstream estimators receive numerically stable reliability inputs.

### measure_judge_sensitivity.py
Measures formatting and hamming sensitivities for a judge by replaying Arena-Hard prompts through the Oumi interface, caching the resulting profiles, and emitting JSON artifacts (`sensitivity_profiles/`, sample diffs) that the debiasing pipeline can load without recomputing dynamic neighbors. Supports sampling or targeting specific questions/models and integrates with `DifferentialDebias` to keep measurement and application decoupled.

### run_combined_abb_analysis.py
Loads judge score JSONL trees, merges in cached sensitivity profiles, and executes the Combined A‑BB debiaser across multiple aggregation strategies (conservative, RMS, weighted, adaptive). It prints diagnostics, writes transformed scores, and validates the ABB constraints so researchers can compare how different noise/calibration settings affect downstream metrics without touching notebook code.

## Dev

### consistency_probe.py
CLI tool that fires repeated, identical Arena-style pairwise prompts at an Oumi-backed judge to quantify verdict variance. It can synthesize prompts or pull an Arena-Hard sample, exposes toggles for token limits and timeout-friendly “test mode,” and summarizes disagreement rates alongside RMS score shifts so researchers can spot stochasticity before running expensive analyses.

## Visualization

### data_loader.py
Shared loader utilities for all visualization scripts. It discovers project paths, ingests original (`base_processed`) and debiased (`base_debiased_*`) JSONL files, converts pairwise tokens to win rates, harmonizes model naming, and produces the aggregated DataFrames that plotting and ELO scripts expect.

### create_comparisons.py
Wrapper that orchestrates visualization runs for each judge and Combined A‑BB strategy by delegating to `visualize_bias_transformation.py`. It iterates through processed JSONL directories, copies/organizes ranking tables, and emits comparison plots so analysts can regenerate the full suite of side-by-side figures with one command.

### generate_rankings.py
Builds the ranking CSVs consumed by the comparison visualizer. It aggregates model scores from JSONL data, computes win-rate confidence intervals (reusing bootstrap helpers), stitches in precomputed ELO outputs when available, and writes the canonical `tables_debiased_<strategy>` artifacts expected by the plotting scripts.

### generate_debiased_rankings_elo.py
Constructs pseudo-battles from debiased scores (`base_debiased_<approach>`), fits ELO ratings via MLE plus bootstrap, converts them to win-rate deltas versus the baseline, and saves CSV summaries. Designed for “Track A” evaluations where debiased outputs need the same treatment as Arena-Hard battles, with debug options and battle filtering.

### generate_original_rankings_elo.py
Runs the same ELO bootstrap workflow on the original judge outputs (`base_processed`), optionally cross-checking Arena-Hard CSV exports. Produces leaderboards that establish the pre-debias baseline so changes introduced by Combined A‑BB can be compared apples-to-apples.

### visualize_bias_transformation.py
Generates the flagship “original vs debiased” plots: line charts with confidence intervals, critical difference diagrams, and summary panels. It relies on the shared loader to normalize model names, lines up corresponding ranking CSVs, and provides configuration flags for judge/strategy subsets, output locations, and styling.

### visualize_results.py
Broad diagnostic viewer for Combined A‑BB runs. It reads the JSON results emitted by `run_combined_abb_analysis.py`, merges original/debiased score tables, and produces heatmaps, scatter plots, and summary tables that highlight how sensitivity settings impact model ordering, score deltas, and confidence.

### run_all_battles.sh
Bash entry point that only runs the battle/ELO portion of the visualization pipeline. After validating environment variables and locating `sos-addl-data` settings, it iterates through each judge and combined debiasing approach, calling `generate_debiased_rankings_elo.py` with debug logging so teams can regenerate bootstrap leaderboards in bulk.

### run_all_visualizations.sh
Full visualization orchestrator. Starting from DATA_BASE discovery, it executes the ranking generators, ELO bootstrap, comparison plots, and result dashboards for every judge/strategy combination, making it the one-stop script to refresh tables and figures when new debiased outputs land.

### run_all_visualizations_skipelo.sh
Variant of the orchestration script that skips the ELO bootstrap stage. Useful when pseudo-battles are unavailable or analysts want quicker refreshes limited to direct win-rate aggregations and visualization overlays.

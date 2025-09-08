#!/usr/bin/env python3
"""
Generate original (non-debiased) rankings via Arena-Hard-style battles + ELO bootstrap
from base_processed JSONL, and optionally compare against arena-hard-auto CSV outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import sys
from pathlib import Path as _Path
# Ensure project root and this directory are importable when run directly
_SCRIPT_DIR = _Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
sys.path.append(str(_REPO_ROOT))
sys.path.append(str(_SCRIPT_DIR))

from differential_debiasing.interfaces.elo_bootstrap import EloParams, bootstrap_elo, get_win_rate_column
from differential_debiasing.interfaces.arena_battles import build_original_battles_from_jsonl
from data_loader import detect_baseline_model


def write_ranking_csv(output_dir: Path, metric_name: str, baseline: str,
                      med_wr: pd.Series, q025_wr: pd.Series, q975_wr: pd.Series) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        'model': med_wr.index,
        f'{metric_name}': med_wr.values,
        'rating_q025': q025_wr.reindex(med_wr.index).values,
        'rating_q975': q975_wr.reindex(med_wr.index).values,
    })
    df['CI'] = df.apply(lambda r: f"({(r['rating_q025']-r[metric_name]):.2f}, +{(r['rating_q975']-r[metric_name]):.2f})", axis=1)
    df['avg_tokens'] = 0.0
    from datetime import datetime
    df['date'] = datetime.now().strftime('%Y-%m-%d')
    df = df.sort_values(metric_name, ascending=False)
    fname = f"arena_hard_leaderboard_original_{baseline}_{metric_name}_elo.csv"
    df.to_csv(output_dir / fname, index=False)
    return output_dir / fname


def compare_to_arena(original_csv: Path, reference_dir: Path):
    try:
        ref_files = list(reference_dir.glob('*.csv'))
        if not ref_files:
            print('No reference CSVs found for comparison')
            return
        # Compare widths for same metric if possible
        ours = pd.read_csv(original_csv)
        # Pick a reference with same metric name
        metric_col = next((c for c in ours.columns if c.endswith('_score') or c=='score'), 'score')
        for ref in ref_files:
            refdf = pd.read_csv(ref)
            if metric_col in refdf.columns:
                merged = refdf.merge(ours, on='model', suffixes=('_ref','_ours'))
                if 'rating_q025_ref' in merged.columns and 'rating_q025_ours' in merged.columns:
                    merged['w_ref'] = merged['rating_q975_ref'] - merged['rating_q025_ref']
                    merged['w_ours'] = merged['rating_q975_ours'] - merged['rating_q025_ours']
                    ratio = (merged['w_ours'] / merged['w_ref'].replace({0: np.nan})).median()
                    print(f"Median CI width ratio (ours/ref): {ratio:.3f}")
                break
    except Exception as e:
        print(f"Comparison failed: {e}")


def main():
    ap = argparse.ArgumentParser(description='Original rankings via ELO bootstrap from base_processed JSONL')
    ap.add_argument('--setting-dir', required=True, help='Judge setting directory (e.g., .../DeepSeek-R1-32B-setting1)')
    ap.add_argument('--metric', default='score', help='Metric field to use for battles (e.g., score, correctness_score)')
    ap.add_argument('--baseline', default=None, help='Override baseline model (else auto-detect from original CIS)')
    ap.add_argument('--rounds', type=int, default=100, help='Bootstrap rounds')
    ap.add_argument('--compare-ref', action='store_true', help='Compare against arena-hard-auto original CIS CSVs if present')
    args = ap.parse_args()

    setting = Path(args.setting_dir)
    baseline = args.baseline or detect_baseline_model(setting) or 'gpt-4-0314'
    print(f"Using baseline: {baseline}")

    base_processed = setting / 'base_processed'
    battles = build_original_battles_from_jsonl(base_processed, baseline_model=baseline, target_metric=args.metric)
    if battles.empty:
        print('No battles built from base_processed')
        return 1
    print(f"Constructed {len(battles)} battles across {battles['model_b'].nunique()+1} models")

    params = EloParams(baseline_model=baseline)
    med, q025, q975 = bootstrap_elo(battles, params, num_rounds=args.rounds)
    med_wr = get_win_rate_column(med, params)
    q025_wr = get_win_rate_column(q025, params)
    q975_wr = get_win_rate_column(q975, params)

    out_dir = setting / 'tables' / 'factor_scores_original_cis_elo'
    csv_path = write_ranking_csv(out_dir, metric_name=('score' if args.metric=='score' else args.metric),
                                 baseline=baseline, med_wr=med_wr, q025_wr=q025_wr, q975_wr=q975_wr)
    print(f"Saved: {csv_path}")

    if args.compare_ref:
        ref_dir = setting / 'tables' / 'factor_scores_original_cis'
        compare_to_arena(csv_path, ref_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

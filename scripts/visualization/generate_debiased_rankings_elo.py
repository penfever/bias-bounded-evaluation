#!/usr/bin/env python3
"""
Generate debiased rankings using pseudo-battles + ELO bootstrap (Track A, Stage 2/3).

Constructs pseudo-battles from base_debiased_<approach> JSONLs by comparing the debiased
score of the baseline model vs each other model for matching (question_id, game).

Then runs MLE ELO + bootstrap (as in show_result.py), converts to win-rate vs baseline,
and writes CSVs under tables_debiased_<approach>/tables/factor_scores_updated_cis_elo/.

TODOs:
 - Add original battles generator to compare against ground truth for original outputs
 - Add Bayesian bootstrap option
 - Wire into main generate_rankings/visualization flows via a flag
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Set
import json
import pandas as pd
import numpy as np

import sys
# Ensure repository root and this visualization folder are on sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from differential_debiasing.interfaces.elo_bootstrap import (
    EloParams, compute_mle_elo, bootstrap_elo, get_win_rate_column
)
from data_loader import detect_baseline_model


def _iter_debiased_entries(jsonl_path: Path):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            yield d

def _discover_metric_fields(base_dir: Path, sample_files: int = 3, sample_lines: int = 200) -> Set[str]:
    """Scan a few JSONL files to discover available score_debiased* fields.

    Returns a set of keys like: {'score_debiased', 'score_debiased_correctness', ...}
    """
    keys: Set[str] = set()
    count_files = 0
    for jsonl in sorted(base_dir.glob('*.jsonl')):
        count_files += 1
        line_count = 0
        try:
            with open(jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    for k in d.keys():
                        if isinstance(k, str) and k.startswith('score_debiased'):
                            keys.add(k)
                    line_count += 1
                    if line_count >= sample_lines:
                        break
        except Exception:
            pass
        if count_files >= sample_files:
            break
    return keys


def _resolve_debiased_dir(judge_dir: Path, approach: str) -> Path:
    """Resolve the debiased directory for a given approach with flexible naming.

    Tries, in order:
      - base_debiased_{approach}
      - base_debiased_abb_{approach} (if not already startswith 'abb_')
      - any base_debiased_* directory that contains the approach token
    """
    # Build prioritized candidate names
    name_variants = [
        f'base_debiased_{approach}',
        f'base_debiased_abb_{approach}',
        f'base_debiased_combined_abb_{approach}',
    ]
    candidates: List[Path] = [judge_dir / n for n in name_variants]

    # Collect all existing base_debiased_* dirs with JSONLs
    existing = [p for p in judge_dir.glob('base_debiased_*') if any(p.glob('*.jsonl'))]
    existing_names = [p.name for p in existing]

    # Try exact matches first
    for c in candidates:
        if c.exists() and any(c.glob('*.jsonl')):
            return c

    # Try equality on suffix after prefix
    suffix_equal: List[Path] = []
    for p in existing:
        suffix = p.name.replace('base_debiased_', '')
        if suffix in (approach, f'abb_{approach}', f'combined_abb_{approach}'):
            suffix_equal.append(p)
    if suffix_equal:
        # Prefer the longest (most specific) match
        return sorted(suffix_equal, key=lambda x: len(x.name), reverse=True)[0]

    # No match: raise a helpful error
    available = ', '.join(sorted(existing_names)) or 'none'
    raise FileNotFoundError(f"Could not resolve debiased dir for approach='{approach}'. Available: {available}")


def build_pseudo_battles(judge_dir: Path, approach: str, baseline: str,
                         metric_field: str = 'score_debiased', debug: bool = False) -> pd.DataFrame:
    """
    Construct pseudo-battles by comparing baseline vs model per (question_id, game).

    Winner is the side with the higher debiased metric. Ties when equal within 1e-9.
    """
    base_dir = _resolve_debiased_dir(judge_dir, approach)
    if debug:
        print(f"Resolved debiased dir for approach='{approach}': {base_dir}")
    if not base_dir.exists():
        available = ', '.join(sorted([p.name for p in judge_dir.glob('base_debiased_*')])) or 'none'
        raise FileNotFoundError(f"Debiased directory not found for approach='{approach}'. Tried common variants. Available: {available}")

    rows = []
    seen_models = set()
    for jsonl in base_dir.glob('*.jsonl'):
        model = jsonl.stem
        if model == baseline:
            continue
        seen_models.add(model)
        for d in _iter_debiased_entries(jsonl):
            if metric_field not in d:
                continue
            try:
                score = float(d.get(metric_field))
            except Exception:
                continue
            q = str(d.get('question_id'))
            g = int(d.get('game', d.get('game_index', 0)))
            # Interpret numeric debiased score using Arena-Hard mapping semantics
            # Score scale: 1..5 with 3 as tie; >3 means 'B preferred', <3 means 'A preferred'
            if abs(score - 3.0) < 1e-9:
                winner = 'tie'
            else:
                b_preferred = score > 3.0
                if g == 0:
                    # game 1: model_a=baseline, model_b=model
                    winner = 'model_b' if b_preferred else 'model_a'
                else:
                    # game 2: roles flipped in source; keeping model_a=baseline => invert mapping
                    winner = 'model_a' if b_preferred else 'model_b'
            rows.append({'question_id': q, 'model_a': baseline, 'model_b': model, 'winner': winner})

    if debug:
        print(f"Models in debiased dir: {len(seen_models)} -> {sorted(list(seen_models))[:5]}{(' ...' if len(seen_models)>5 else '')}")
        print(f"Constructed rows: {len(rows)} for metric={metric_field}")
    if not rows:
        return pd.DataFrame(columns=['question_id','model_a','model_b','winner'])
    return pd.DataFrame(rows)


def _normalize_strategy(approach: str) -> str:
    m = {
        'combined_abb_conservative': 'conservative',
        'combined_abb_rms': 'rms',
        'combined_abb_weighted': 'weighted',
        'combined_abb_montecarlo': 'montecarlo',
        'abb_formatting_only': 'formatting_only',
    }
    return m.get(approach, approach)


def _metric_field_to_colname(metric_field: str) -> str:
    # score_debiased -> score; score_debiased_correctness -> correctness_score
    if metric_field == 'score_debiased':
        return 'score'
    if metric_field.startswith('score_debiased_'):
        base = metric_field.replace('score_debiased_', '')
        return f'{base}_score'
    return 'score'


def write_ranking_csv(output_dir: Path, metric_colname: str, baseline: str,
                      med_wr: pd.Series, q025_wr: pd.Series, q975_wr: pd.Series):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        'model': med_wr.index,
        f'{metric_colname}': med_wr.values,
        'rating_q025': q025_wr.reindex(med_wr.index).values,
        'rating_q975': q975_wr.reindex(med_wr.index).values,
    })
    # CI string as delta
    def _fmt(row):
        return f"({(row['rating_q025']-row[metric_colname]):.2f}, +{(row['rating_q975']-row[metric_colname]):.2f})"
    df['CI'] = df.apply(_fmt, axis=1)
    df['avg_tokens'] = 0.0
    from datetime import datetime
    df['date'] = datetime.now().strftime('%Y-%m-%d')
    # Sort by metric desc
    df = df.sort_values(metric_colname, ascending=False)
    # Save
    fname = f"arena_hard_leaderboard_debiased_{baseline}_{metric_colname}_elo.csv"
    df.to_csv(output_dir / fname, index=False)
    return output_dir / fname


def main():
    ap = argparse.ArgumentParser(description='Debiased rankings via pseudo-battles + ELO bootstrap')
    ap.add_argument('--setting-dir', required=True, help='Judge setting directory (e.g., sos-addl-data/.../DeepSeek-R1-32B-setting1)')
    ap.add_argument('--approach', required=False, default=None, help='Approach name suffix (e.g., formatting_only). If omitted, inferred from --debiased-dir')
    ap.add_argument('--debiased-dir', required=False, default=None, help='Explicit base_debiased* directory to read JSONLs from')
    ap.add_argument('--metric', default='all', help='Metric field: score_debiased, score_debiased_<factor>, or "all"')
    ap.add_argument('--rounds', type=int, default=100, help='Bootstrap rounds')
    ap.add_argument('--baseline', default=None, help='Override baseline model (else auto-detect)')
    ap.add_argument('--debug', action='store_true', help='Print debug info about directory resolution and battle construction')
    args = ap.parse_args()

    judge_dir = Path(args.setting_dir)
    baseline = args.baseline or detect_baseline_model(judge_dir) or 'gpt-4-0314'
    print(f"Using baseline: {baseline}")

    # Determine metrics to generate
    if args.metric == 'all':
        # Prefer explicit discovery from the target debiased directory (explicit or resolved)
        target_dir = Path(args.debiased_dir) if args.debiased_dir else None
        if target_dir is None:
            # If no explicit dir, try to resolve once using the provided/resolved approach
            try:
                target_dir = _resolve_debiased_dir(judge_dir, args.approach or '')
            except Exception:
                target_dir = None
        discovered = _discover_metric_fields(target_dir) if target_dir and target_dir.exists() else set()
        # Standard ordering
        preferred_order = [
            'score_debiased',
            'score_debiased_correctness',
            'score_debiased_completeness',
            'score_debiased_safety',
            'score_debiased_conciseness',
            'score_debiased_style',
        ]
        metric_fields = [m for m in preferred_order if m in discovered] if discovered else ['score_debiased']
        if args.debug:
            print(f"Discovered metric fields in {target_dir}: {sorted(discovered)}")
            print(f"Planned metrics: {metric_fields}")
    else:
        metric_fields = [args.metric]

    # Resolve debiased base directory and approach name
    explicit_dir: Optional[Path] = Path(args.debiased_dir) if args.debiased_dir else None
    if explicit_dir is not None:
        if not explicit_dir.exists():
            raise FileNotFoundError(f"--debiased-dir not found: {explicit_dir}")
        base_name = explicit_dir.name
        # Extract approach from directory name
        suffix = base_name.replace('base_debiased_', '') if base_name != 'base_debiased' else 'default'
        resolved_approach = args.approach or suffix
    else:
        if not args.approach:
            raise ValueError("Either --approach or --debiased-dir must be provided")
        resolved_approach = args.approach

    # Normalized strategy name for output
    norm_strategy = _normalize_strategy(resolved_approach)

    params = EloParams(baseline_model=baseline)

    generated = 0
    for metric_field in metric_fields:
        # Build battles per metric
        if explicit_dir is not None:
            base_dir = explicit_dir
            if args.debug:
                print(f"Using explicit debiased dir: {base_dir}")
            # Monkey-patch resolution by passing the precise folder via a local wrapper
            def _build_from_dir(dir_path: Path) -> pd.DataFrame:
                rows = []
                seen_models = set()
                for jsonl in dir_path.glob('*.jsonl'):
                    model = jsonl.stem
                    if model == baseline:
                        continue
                    seen_models.add(model)
                    for d in _iter_debiased_entries(jsonl):
                        if metric_field not in d:
                            continue
                        try:
                            score = float(d.get(metric_field))
                        except Exception:
                            continue
                        q = str(d.get('question_id'))
                        g = int(d.get('game', d.get('game_index', 0)))
                        if abs(score - 3.0) < 1e-9:
                            winner = 'tie'
                        else:
                            b_preferred = score > 3.0
                            if g == 0:
                                winner = 'model_b' if b_preferred else 'model_a'
                            else:
                                winner = 'model_a' if b_preferred else 'model_b'
                        rows.append({'question_id': q, 'model_a': baseline, 'model_b': model, 'winner': winner})
                if args.debug:
                    print(f"Models in debiased dir: {len(seen_models)} -> {sorted(list(seen_models))[:5]}{(' ...' if len(seen_models)>5 else '')}")
                    print(f"Constructed rows: {len(rows)} for metric={metric_field}")
                return pd.DataFrame(rows, columns=['question_id','model_a','model_b','winner']) if rows else pd.DataFrame(columns=['question_id','model_a','model_b','winner'])

            battles = _build_from_dir(base_dir)
        else:
            battles = build_pseudo_battles(judge_dir, resolved_approach, baseline, metric_field=metric_field, debug=args.debug)
        if battles.empty:
            print(f"No battles constructed for metric '{metric_field}'. Skipping.")
            continue
        print(f"Constructed {len(battles)} pseudo-battles for {metric_field} across {battles['model_b'].nunique()+1} models")
        if args.debug:
            # Summarize wins/losses/ties vs baseline per model
            def _sum_counts(df: pd.DataFrame) -> pd.DataFrame:
                wins = (df['winner'] == 'model_b').groupby(df['model_b']).sum()
                losses = (df['winner'] == 'model_a').groupby(df['model_b']).sum()
                ties = (df['winner'] == 'tie').groupby(df['model_b']).sum()
                summary = pd.DataFrame({'wins': wins, 'losses': losses, 'ties': ties}).fillna(0).astype(int)
                summary['total'] = summary['wins'] + summary['losses'] + summary['ties']
                return summary.sort_values('wins', ascending=False)
            summary = _sum_counts(battles)
            print("Top 5 win counts vs baseline:")
            print(summary.head(5))

        med, q025, q975 = bootstrap_elo(battles, params, num_rounds=args.rounds)
        med_wr = get_win_rate_column(med, params)
        q025_wr = get_win_rate_column(q025, params)
        q975_wr = get_win_rate_column(q975, params)

        # Output directory with normalized strategy name
        out_dir = judge_dir / f'tables_debiased_{norm_strategy}' / 'tables' / 'factor_scores_updated_cis_elo'
        metric_colname = _metric_field_to_colname(metric_field)
        csv_path = write_ranking_csv(out_dir, metric_colname=metric_colname,
                                     baseline=baseline, med_wr=med_wr, q025_wr=q025_wr, q975_wr=q975_wr)
        print(f"Saved: {csv_path}")
        if args.debug:
            # Also write a debug summary next to outputs
            debug_out = (judge_dir / f'tables_debiased_{norm_strategy}' / 'tables' / 'debug')
            debug_out.mkdir(parents=True, exist_ok=True)
            summary.to_csv(debug_out / f'battle_summary_{metric_colname}.csv')
            print(f"Saved debug battle summary: {debug_out / f'battle_summary_{metric_colname}.csv'}")
        generated += 1

    if generated == 0:
        print("No CSVs generated (no battles for any metrics).")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

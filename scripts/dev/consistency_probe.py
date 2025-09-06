#!/usr/bin/env python3
"""
Consistency Probe: issue multiple identical Arena-Hard pairwise judgments and report variance.

Usage:
  python bias-bounded-evaluation/scripts/dev/consistency_probe.py \
      --judge gpt-4o-mini --attempts 5

Optional custom prompt:
  --question "..." --answer-a "..." --answer-b "..."

Notes:
  - Uses Oumi-based judge function with caching disabled.
  - Prints pairwise tokens and numeric scores per attempt, plus summary stats.
"""

import argparse
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from differential_debiasing.interfaces.oumi_interface import create_oumi_judge_function
from differential_debiasing.core.utils import rms_from_differences
from differential_debiasing.interfaces.arena_hard_utils import (
    find_sample_data as find_arena_sample_data,
    load_judgment_data as load_arena_judgment,
    parse_arena_hard_prompt,
    setting_dir_for_judge,
)


def get_judge_config_path(judge_name: str) -> Path:
    base_config_dir = Path(__file__).parent.parent.parent / "configs" / "judges"
    judge_mapping = {
        # OpenAI models
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo.yaml",
        "gpt-4o-mini": "openai/gpt-4o-mini.yaml",
        # Anthropic
        "claude-3-5-sonnet": "anthropic/claude-3-5-sonnet.yaml",
        # Local GGUF
        "qwq-32b-gguf": "local/qwq-32b-gguf.yaml",
        "deepseek-r1-32b-gguf": "local/deepseek-r1-32b-gguf.yaml",
    }
    if judge_name not in judge_mapping:
        raise FileNotFoundError(f"Unknown judge '{judge_name}'. Available: {sorted(judge_mapping.keys())}")
    path = base_config_dir / judge_mapping[judge_name]
    if not path.exists():
        raise FileNotFoundError(f"Judge config not found: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Probe judgment variance over repeated identical prompts")
    parser.add_argument("--judge", "-j", required=True, help="Judge name (e.g., gpt-4o-mini)")
    parser.add_argument("--attempts", "-k", type=int, default=5, help="Number of repeated judgments")
    parser.add_argument("--question", type=str, default="Summarize the benefits of unit testing in large codebases.")
    parser.add_argument("--answer-a", type=str, default=(
        "Unit testing improves code quality by catching regressions early, "
        "enabling safe refactoring, clarifying design intent, and serving as living documentation. "
        "In large codebases, fast feedback loops reduce integration risk and facilitate modular, maintainable architectures."
    ))
    parser.add_argument("--answer-b", type=str, default=(
        "Unit tests ensure functions behave as expected, which helps prevent bugs from shipping. "
        "They also speed up development by allowing quick checks after changes, "
        "and encourage simpler, decoupled designs that scale better."
    ))
    parser.add_argument("--arena", action="store_true", help="Use a real Arena-Hard sample instead of synthetic prompt")
    parser.add_argument("--data-path", "-d", type=str, default=None, help="Base path containing Arena-Hard InDepthAnalysis data")
    parser.add_argument("--question-id", type=str, default=None, help="Specific Arena-Hard question_id to probe")
    parser.add_argument("--model", type=str, default=None, help="Specific Arena-Hard model file (e.g., gpt-4o-mini-2024-07-18)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Override max_new_tokens for the judge (caps rationale length)")
    parser.add_argument("--test-mode", action="store_true", help="Enable per-call timeout in interface (15s)")
    args = parser.parse_args()

    cfg_path = get_judge_config_path(args.judge)
    judge_fn = create_oumi_judge_function(
        str(cfg_path),
        cost_budget_usd=5.0,
        cache_responses=False,           # ensure each attempt hits the model
        prefer_existing_scores=False     # always recompute
    )

    # Use the underlying interface directly to get pairwise token in addition to numeric
    interface = judge_fn.interface
    # Apply overrides before first query
    try:
        interface.config.setdefault('generation', {})['max_new_tokens'] = int(args.max_new_tokens)
    except Exception:
        pass
    if args.test_mode:
        import os as _os
        _os.environ['TEST_MODE'] = 'true'

    # Resolve prompt context: synthetic default or Arena-Hard sample
    if args.arena:
        if not args.data_path:
            raise SystemExit("--arena requires --data-path to locate Arena-Hard data")
        try:
            import pandas as pd
            data_path = Path(args.data_path)
            question = None
            answer_a = None
            answer_b = None
            chosen_qid = args.question_id
            chosen_model = args.model
            chosen_source_dir = None

            if chosen_qid and chosen_model:
                # Direct lookup in expected setting dir (or scan)
                # Try expected setting for judge; fall back to scan
                expected = setting_dir_for_judge(args.judge)
                candidate_dirs = []
                if expected and (data_path / expected / 'base_processed').exists():
                    candidate_dirs = [data_path / expected / 'base_processed']
                else:
                    # Scan for any base_processed
                    candidate_dirs = [p / 'base_processed' for p in data_path.glob('*-setting*') if (p / 'base_processed').exists()]
                record = None
                for base_dir in candidate_dirs:
                    try:
                        record = load_arena_judgment(base_dir, chosen_qid, chosen_model)
                        chosen_source_dir = str(base_dir)
                        break
                    except Exception:
                        continue
                if record is None:
                    raise RuntimeError(f"Could not find record for question_id={chosen_qid}, model={chosen_model}")
            else:
                # Sample a random row from available data for this judge
                df = find_arena_sample_data(data_path, judge_name=args.judge, max_samples=50)
                row = df.sample(n=1, random_state=args.seed).iloc[0]
                chosen_qid = row['question_id']
                chosen_model = row['model']
                chosen_source_dir = row['source_dir']
                record = load_arena_judgment(chosen_source_dir, chosen_qid, chosen_model)

            user_prompt = record['games'][0]['user_prompt']
            question, answer_a, answer_b = parse_arena_hard_prompt(user_prompt)
            print("Using Arena-Hard sample:")
            print(f"  question_id: {chosen_qid}")
            print(f"  model:       {chosen_model}")
            print(f"  source_dir:  {chosen_source_dir}")
        except Exception as e:
            raise SystemExit(f"Failed to load Arena-Hard sample: {e}")
        context = {
            "question": question,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "model": chosen_model,
            "question_id": chosen_qid
        }
    else:
        context = {
            "question": args.question,
            "answer_a": args.answer_a,
            "answer_b": args.answer_b,
            "model": "consistency-probe"
        }

    print(f"Judge: {args.judge}")
    print(f"Attempts: {args.attempts}")
    print("\nPrompt summary:")
    q_preview = context['question'][:80] if isinstance(context['question'], str) else str(context['question'])[:80]
    print(f"  Q: {q_preview}{'...' if len(str(context['question'])) > 80 else ''}")
    print(f"  A.len: {len(context['answer_a'])}  B.len: {len(context['answer_b'])}")

    tokens = []
    scores = []

    for i in range(args.attempts):
        resp = interface.query_judge(context)
        token = resp.get('scores', {}).get('pairwise_comparison')
        num = resp.get('scores', {}).get('overall_score')
        if token is None or num is None:
            err = resp.get('parse_error', 'unparseable')
            print(f"  [{i+1}] parse_error: {err}")
            continue
        tokens.append(token)
        scores.append(float(num))
        print(f"  [{i+1}] {token}  -> {num}")

    if not scores:
        print("\nNo successful parses; cannot compute variance.")
        return 1

    # Summary stats (RMS sensitivity w.r.t first successful attempt)
    counts = Counter(tokens)
    mode_token, mode_count = counts.most_common(1)[0]
    baseline = scores[0]
    diffs = [s - baseline for s in scores]
    rms_sensitivity = rms_from_differences(diffs)
    token_disagreement = sum(1 for t in tokens if t != tokens[0])

    print("\nSummary:")
    print(f"  Verdict counts: {dict(counts)}")
    print(f"  Mode verdict: {mode_token} ({mode_count}/{len(tokens)})")
    print(f"  RMS sensitivity (vs first): {rms_sensitivity:.4f}")
    print(f"  Verdict disagreement vs first: {token_disagreement}/{len(tokens)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

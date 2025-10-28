#!/usr/bin/env python3
"""
Quick test harness for Arena-Hard parsing utilities.

- Extracts verdict tokens and maps to numeric scores
- Parses raw Arena-Hard prompts into (question, answer_a, answer_b)

Usage:
  python bias-bounded-evaluation/scripts/dev/test_arena_parser.py \
      --file sos-addl-data/InDepthAnalysis/QwQ-32B-setting1/base_processed/gpt-4-0314.jsonl \
      --lines 5

If --file is omitted, the script searches under sos-addl-data/InDepthAnalysis/*/base_processed/*.jsonl
and picks the first file it finds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def add_repo_root_to_path() -> None:
    # scripts/dev/test_arena_parser.py -> scripts/dev -> scripts -> bias-bounded-evaluation
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def find_default_jsonl() -> Optional[Path]:
    base = Path("sos-addl-data/InDepthAnalysis")
    if not base.exists():
        return None
    for setting in base.glob("*-setting*/base_processed/*.jsonl"):
        return setting
    # Fallback: 'base' directories
    for setting in base.glob("*-setting*/base/*.jsonl"):
        return setting
    return None


def main() -> int:
    add_repo_root_to_path()

    parser = argparse.ArgumentParser(description="Test Arena-Hard prompt and verdict parsing")
    parser.add_argument("--file", type=str, default=None, help="Path to a base_processed JSONL file")
    parser.add_argument("--lines", type=int, default=5, help="Number of lines to test")
    args = parser.parse_args()

    # Lazy import after path setup
    from differential_debiasing.interfaces.arena_hard_utils import (
        extract_verdict_token,
        verdict_token_to_score,
        parse_arena_hard_prompt,
    )

    jsonl_path = Path(args.file) if args.file else find_default_jsonl()
    if not jsonl_path or not jsonl_path.exists():
        print("❌ Could not locate a JSONL file. Provide --file explicitly.")
        return 1

    print(f"📄 Testing file: {jsonl_path}")

    extracted = 0
    parsed_ok = 0
    total = 0
    saw_any_judgment_like = False

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.lines:
                break
            total += 1
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[{i}] ⚠️  JSON decode error: {e}")
                continue

            qid = rec.get("question_id")
            token = extract_verdict_token(rec)
            score = verdict_token_to_score(token) if token else None
            if token:
                extracted += 1
            # Heuristic: note if this record appears to contain judgment data
            if (isinstance(rec, dict) and (
                'score' in rec or 'judgment' in rec or
                (isinstance(rec.get('games'), list) and any(isinstance(g, dict) and ('score' in g or 'judgment' in g) for g in rec['games']))
            )):
                saw_any_judgment_like = True
            print(f"[{i}] question_id={qid} token={token} score={score}")

            try:
                user_prompt = rec["games"][0]["user_prompt"]
                q, a, b = parse_arena_hard_prompt(user_prompt)
                parsed_ok += 1
                print(f"     prompt parsed: q={len(q)} chars, A={len(a)} chars, B={len(b)} chars")
            except Exception as e:
                print(f"     ⚠️  prompt parse failed: {e}")

    print("\n📊 Summary:")
    print(f"  Lines tested: {total}")
    print(f"  Verdict tokens extracted: {extracted}")
    print(f"  Prompts parsed OK: {parsed_ok}")

    if extracted == 0 and not saw_any_judgment_like:
        print("\n💡 Hint: This looks like a 'base' file without judgments.")
        print("   Try the matching base_processed/*.jsonl for verdict extraction.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

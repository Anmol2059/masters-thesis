#!/usr/bin/env python3
"""Evaluator ablation: LLM judge vs human ratings."""
import argparse
import json
from pathlib import Path

from src.evaluator import LLMJudge, aggregate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", type=Path, required=True, help="Dir of session JSON files")
    p.add_argument("--output", type=Path, default=Path("results/evaluator_comparison.json"))
    args = p.parse_args()

    judge = LLMJudge()
    all_results = []
    for session_file in sorted(args.sessions.glob("*.json")):
        session = json.loads(session_file.read_text())
        for seg in session.get("segments", []):
            result = judge.evaluate(seg["source"], seg["reference"], seg["interpretation"])
            all_results.append(result)

    summary = aggregate(all_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

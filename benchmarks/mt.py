#!/usr/bin/env python3
"""MT model comparison benchmark."""
import argparse
import csv
import json
from pathlib import Path

from src.translation import Translator, NLLBTranslator, load_glossary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-transcripts", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--glossary", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("results/mt_comparison.csv"))
    args = p.parse_args()

    glossary = load_glossary(args.glossary) if args.glossary else {}
    sources = json.loads(args.source_transcripts.read_text())
    refs = json.loads(args.references.read_text())

    models = {
        "qwen2.5-7b-domain": Translator(glossary=glossary),
        "nllb-600m-baseline": NLLBTranslator(),
    }
    rows = []
    for name, model in models.items():
        for src, ref in zip(sources, refs):
            hyp = model.translate(src)
            rows.append({"model": name, "source": src[:60], "hypothesis": hyp[:60], "reference": ref[:60]})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "source", "hypothesis", "reference"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results → {args.output}")


if __name__ == "__main__":
    main()

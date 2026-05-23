"""Experiment 2: Translation vanilla vs glossary-injected."""
import argparse
import json
from pathlib import Path

from src.translator import Translator
from src.metrics import compute_bleu, compute_chrf, compute_comet, compute_term_accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", required=True)
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--model", choices=["nllb", "qwen"], required=True)
    p.add_argument("--condition", choices=["vanilla", "domain"], default="vanilla")
    p.add_argument("--glossary", default=None)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.source_dir)
    ref_dir = Path(args.reference_dir)

    translator = Translator(model=args.model)
    if args.condition == "domain" and args.glossary:
        translator.load_glossary(args.glossary)

    hypotheses, references = [], []
    samples = []

    for src_file in sorted(src_dir.glob("*.txt")):
        ref_file = ref_dir / src_file.name
        if not ref_file.exists():
            continue

        source = src_file.read_text().strip()
        reference = ref_file.read_text().strip()
        hypothesis = translator.translate(source)

        hypotheses.append(hypothesis)
        references.append(reference)
        samples.append({"id": src_file.stem, "source": source,
                         "hypothesis": hypothesis, "reference": reference})

    glossary_terms = translator.glossary if args.glossary else {}
    output = {
        "model": args.model,
        "condition": args.condition,
        "bleu": compute_bleu(hypotheses, references),
        "chrf": compute_chrf(hypotheses, references),
        "comet": compute_comet(
            [s["source"] for s in samples], hypotheses, references
        ),
        "term_accuracy": compute_term_accuracy(hypotheses, glossary_terms),
        "samples": samples,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"BLEU: {output['bleu']:.2f}  chrF: {output['chrf']:.2f}  "
          f"COMET: {output['comet']:.4f}  TermAcc: {output['term_accuracy']:.2%}")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()

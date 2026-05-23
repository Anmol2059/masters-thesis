"""Experiment 4: Compare glossary construction methods."""
import argparse
import json
from pathlib import Path

from src.translator import Translator
from src.metrics import compute_bleu, compute_chrf, compute_comet, compute_term_accuracy

GLOSSARY_FILES = {
    "manual": "glossaries/eu_parliament_es_en.json",
    "tfidf":  "glossaries/eu_parliament_es_en_tfidf.json",
    "llm":    "glossaries/eu_parliament_es_en_llm.json",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", required=True)
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--glossaries", nargs="+",
                   choices=["manual", "tfidf", "llm"], default=["manual", "tfidf", "llm"])
    p.add_argument("--output", required=True)
    return p.parse_args()


def evaluate_glossary(method, src_dir, ref_dir):
    glossary_path = GLOSSARY_FILES[method]
    translator = Translator(model="qwen")
    translator.load_glossary(glossary_path)

    hyps, refs, srcs = [], [], []
    for src_file in sorted(src_dir.glob("*.txt")):
        ref_file = ref_dir / src_file.name
        if not ref_file.exists():
            continue
        source = src_file.read_text().strip()
        hyps.append(translator.translate(source))
        refs.append(ref_file.read_text().strip())
        srcs.append(source)

    return {
        "method": method,
        "n_terms": len(translator.glossary),
        "bleu": compute_bleu(hyps, refs),
        "chrf": compute_chrf(hyps, refs),
        "comet": compute_comet(srcs, hyps, refs),
        "term_accuracy": compute_term_accuracy(hyps, translator.glossary),
    }


def main():
    args = parse_args()
    src_dir = Path(args.source_dir)
    ref_dir = Path(args.reference_dir)

    rows = [evaluate_glossary(m, src_dir, ref_dir) for m in args.glossaries]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    print(f"{'Method':<12} {'#Terms':>7} {'BLEU':>7} {'chrF':>7} {'COMET':>7} {'TermAcc':>9}")
    for r in rows:
        print(f"{r['method']:<12} {r['n_terms']:>7} {r['bleu']:>7.2f} "
              f"{r['chrf']:>7.2f} {r['comet']:>7.4f} {r['term_accuracy']:>8.2%}")
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()

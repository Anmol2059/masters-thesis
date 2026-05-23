"""Experiment 4: Compare glossary construction methods (manual vs TF-IDF vs LLM-generated)."""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.translator import Translator
from src.metrics import compute_bleu, compute_chrf, compute_comet, compute_term_accuracy
from src.utils import setup_logging, make_output_dir, load_cached, save_cached, auto_output_dir

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
    p.add_argument("--output-dir", default=None)
    p.add_argument("--log-dir", default="logs")
    return p.parse_args()


def evaluate_glossary(method, src_dir, ref_dir, transcripts_dir, logger):
    translator = Translator(model="qwen")
    translator.load_glossary(GLOSSARY_FILES[method])

    src_files = sorted(src_dir.glob("*.txt"))
    hyps, refs, srcs = [], [], []

    for src_file in tqdm(src_files, desc=f"Glossary [{method}]", unit="file"):
        ref_file = ref_dir / src_file.name
        if not ref_file.exists():
            continue

        cache_key = f"{method}_{src_file.stem}"
        cached = load_cached(transcripts_dir, cache_key)
        if cached:
            hyps.append(cached["hypothesis_en"])
            refs.append(cached["reference_en"])
            srcs.append(cached["source_es"])
            continue

        source = src_file.read_text().strip()
        reference = ref_file.read_text().strip()
        hypothesis = translator.translate(source)

        entry = {"id": src_file.stem, "source_es": source,
                 "hypothesis_en": hypothesis, "reference_en": reference}
        save_cached(transcripts_dir, cache_key, entry)
        hyps.append(hypothesis)
        refs.append(reference)
        srcs.append(source)
        logger.info("%s [%s] translated", src_file.stem, method)

    return {
        "method":        method,
        "n_terms":       len(translator.glossary),
        "n_files":       len(hyps),
        "bleu":          compute_bleu(hyps, refs),
        "chrf":          compute_chrf(hyps, refs),
        "comet":         compute_comet(srcs, hyps, refs),
        "term_accuracy": compute_term_accuracy(hyps, translator.glossary),
    }


def main():
    args = parse_args()
    logger, ts = setup_logging(args.log_dir, "glossary_comparison")

    out_dir = make_output_dir(args.output_dir or auto_output_dir("glossary_comparison", ts))
    transcripts_dir = out_dir / "transcripts"
    logger.info("Output dir: %s", out_dir)

    src_dir = Path(args.source_dir)
    ref_dir = Path(args.reference_dir)

    rows = [evaluate_glossary(m, src_dir, ref_dir, transcripts_dir, logger)
            for m in args.glossaries]

    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    tqdm.write(f"\n{'Method':<12} {'#Terms':>7} {'BLEU':>7} {'chrF':>7} {'COMET':>7} {'TermAcc':>9}")
    for r in rows:
        tqdm.write(
            f"{r['method']:<12} {r['n_terms']:>7} {r['bleu']:>7.2f} "
            f"{r['chrf']:>7.2f} {r['comet']:>7.4f} {r['term_accuracy']:>8.2%}"
        )
    tqdm.write(f"\nSaved → {out_dir}")


if __name__ == "__main__":
    main()

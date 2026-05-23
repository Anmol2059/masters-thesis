"""Experiment 2: Translation — NLLB-3.3B baseline vs Qwen3-8B vanilla vs Qwen3-8B + glossary."""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.translator import Translator
from src.metrics import compute_bleu, compute_chrf, compute_comet, compute_term_accuracy
from src.utils import setup_logging, make_output_dir, load_cached, save_cached, auto_output_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", required=True)
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--model", choices=["nllb", "qwen"], required=True)
    p.add_argument("--condition", choices=["vanilla", "domain"], default="vanilla")
    p.add_argument("--glossary", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--log-dir", default="logs")
    return p.parse_args()


def main():
    args = parse_args()
    tag = f"mt_{args.model}_{args.condition}"
    logger, ts = setup_logging(args.log_dir, tag)

    out_dir = make_output_dir(args.output_dir or auto_output_dir(tag, ts))
    transcripts_dir = out_dir / "transcripts"
    logger.info("Output dir: %s", out_dir)

    translator = Translator(model=args.model)
    if args.condition == "domain" and args.glossary:
        translator.load_glossary(args.glossary)

    src_files = sorted(Path(args.source_dir).glob("*.txt"))
    ref_dir = Path(args.reference_dir)

    all_results = []
    skipped = 0

    for src_file in tqdm(src_files, desc=f"MT [{args.model}/{args.condition}]", unit="file"):
        ref_file = ref_dir / src_file.name
        if not ref_file.exists():
            continue

        cached = load_cached(transcripts_dir, src_file.stem)
        if cached:
            all_results.append(cached)
            skipped += 1
            continue

        source = src_file.read_text().strip()
        reference = ref_file.read_text().strip()
        hypothesis = translator.translate(source)

        entry = {
            "id": src_file.stem,
            "source_es": source,
            "hypothesis_en": hypothesis,
            "reference_en": reference,
        }
        save_cached(transcripts_dir, src_file.stem, entry)
        all_results.append(entry)
        logger.info("%s  translated", src_file.stem)

    tqdm.write(f"Resumed {skipped}/{len(all_results)} files from cache.")

    if not all_results:
        tqdm.write("No results — check --source-dir and --reference-dir paths.")
        return

    sources = [r["source_es"]    for r in all_results]
    hyps    = [r["hypothesis_en"] for r in all_results]
    refs    = [r["reference_en"]  for r in all_results]

    glossary_terms = translator.glossary if args.glossary else {}
    summary = {
        "model":         args.model,
        "condition":     args.condition,
        "n_files":       len(all_results),
        "bleu":          compute_bleu(hyps, refs),
        "chrf":          compute_chrf(hyps, refs),
        "comet":         compute_comet(sources, hyps, refs),
        "term_accuracy": compute_term_accuracy(hyps, glossary_terms),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    tqdm.write(
        f"\nBLEU: {summary['bleu']:.2f}  chrF: {summary['chrf']:.2f}"
        f"  COMET: {summary['comet']:.4f}  TermAcc: {summary['term_accuracy']:.2%}"
        f"  ({len(all_results)} files)\nSaved → {out_dir}"
    )
    logger.info("Done. bleu=%.2f comet=%.4f", summary["bleu"], summary["comet"])


if __name__ == "__main__":
    main()

"""Experiment 3: End-to-end audio → English.

Two backends:
  cascaded  — Whisper large-v3 (ES→ES) + NLLB-3.3B (ES→EN)
  seamless  — SeamlessM4T v2 (ES audio → EN text directly)

Evaluation uses the multi-faceted protocol (docs/evaluation.md):
  - BLEU / chrF          : surface overlap with interpreter reference
  - COMET-DA             : interpreter alignment (semantic)
  - COMET-Kiwi           : faithfulness to source (reference-free)
  - TermAcc              : IATE EU terminology accuracy
  - Composite            : weighted aggregate
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.asr import Transcriber, SeamlessTranscriber
from src.translator import Translator
from src.metrics import (compute_bleu, compute_chrf, compute_comet,
                          compute_comet_kiwi, compute_term_accuracy, compute_composite)
from src.utils import setup_logging, make_output_dir, load_cached, save_cached, auto_output_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--reference-dir", required=True, help="Gold EN interpreter transcripts")
    p.add_argument("--source-dir", default=None,
                   help="Gold ES transcripts for COMET source (skip COMET if absent)")
    p.add_argument("--backend", choices=["cascaded", "seamless"], default="cascaded")
    p.add_argument("--asr-model", default="large-v3")
    p.add_argument("--glossary", default=None, help="Optional IATE glossary for TermAcc")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--log-dir", default="logs")
    return p.parse_args()


def main():
    args = parse_args()
    tag = f"pipeline_{args.backend}"
    logger, ts = setup_logging(args.log_dir, tag)

    out_dir = make_output_dir(args.output_dir or auto_output_dir(tag, ts))
    transcripts_dir = out_dir / "transcripts"
    logger.info("Output dir: %s", out_dir)

    if args.backend == "seamless":
        transcriber = SeamlessTranscriber()
        translator = None
        asr_model_id = SeamlessTranscriber.MODEL_ID
        # Note: SeamlessM4T v2 was trained on Europarl (EU Parliament data),
        # giving it an inherent domain advantage — documented in discussion.
    else:
        transcriber = Transcriber(model_size=args.asr_model)
        translator = Translator(model="nllb")
        asr_model_id = f"faster-whisper-{args.asr_model}"

    audio_files = sorted(Path(args.audio_dir).glob("*.wav"))
    ref_dir = Path(args.reference_dir)
    src_dir = Path(args.source_dir) if args.source_dir else None

    glossary: dict[str, str] = {}
    if args.glossary:
        glossary = json.loads(Path(args.glossary).read_text())

    all_results = []
    skipped = 0

    for audio_file in tqdm(audio_files, desc=f"Pipeline [{args.backend}]", unit="file"):
        stem = audio_file.stem
        ref_file = ref_dir / f"{stem}.txt"
        if not ref_file.exists():
            continue

        cached = load_cached(transcripts_dir, stem)
        if cached:
            all_results.append(cached)
            skipped += 1
            continue

        for attempt in range(3):
            try:
                if args.backend == "seamless":
                    hypothesis_en = transcriber.translate(str(audio_file))
                    asr_es = ""
                else:
                    asr_es = transcriber.transcribe(str(audio_file))
                    hypothesis_en = translator.translate(asr_es)
                break
            except RuntimeError as e:
                tqdm.write(f"  [retry {attempt+1}/3] {stem}: {e}")
                if attempt == 2:
                    tqdm.write(f"  [skip] {stem} failed after 3 attempts")
                    logger.error("skip %s after 3 attempts: %s", stem, e)
                    hypothesis_en = ""
                    asr_es = ""

        reference_en = ref_file.read_text().strip()
        source_es = ""
        if src_dir:
            src_file = src_dir / f"{stem}.txt"
            if src_file.exists():
                source_es = src_file.read_text().strip()

        entry = {
            "id": stem,
            "asr_es": asr_es,
            "hypothesis_en": hypothesis_en,
            "reference_en": reference_en,
            "source_es": source_es,
        }
        save_cached(transcripts_dir, stem, entry)
        all_results.append(entry)
        logger.info("%s  done", stem)

    tqdm.write(f"Resumed {skipped}/{len(all_results)} files from cache.")
    if not all_results:
        tqdm.write("No results — check --audio-dir and --reference-dir.")
        return

    hyps    = [r["hypothesis_en"] for r in all_results]
    refs    = [r["reference_en"]  for r in all_results]
    sources = [r["source_es"]     for r in all_results]
    has_src = any(sources)

    bleu = compute_bleu(hyps, refs)
    chrf = compute_chrf(hyps, refs)
    comet_da    = compute_comet(sources, hyps, refs)      if has_src else None
    comet_kiwi  = compute_comet_kiwi(sources, hyps)       if has_src else None
    term_acc    = compute_term_accuracy(hyps, glossary)   if glossary else None
    composite   = compute_composite(
        comet_da or 0, comet_kiwi or 0, term_acc or 0, bleu
    ) if has_src else None

    summary = {
        "backend":     args.backend,
        "asr_model":   asr_model_id,
        "mt_model":    "facebook/nllb-200-3.3B" if args.backend == "cascaded" else "n/a",
        "n_files":     len(all_results),
        "bleu":        bleu,
        "chrf":        chrf,
        "comet_da":    comet_da,
        "comet_kiwi":  comet_kiwi,
        "term_acc":    term_acc,
        "composite":   composite,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    tqdm.write(
        f"\nBLEU: {bleu:.2f}  chrF: {chrf:.2f}"
        + (f"  COMET-DA: {comet_da:.4f}  COMET-Kiwi: {comet_kiwi:.4f}" if has_src else "")
        + (f"  TermAcc: {term_acc:.2%}" if term_acc is not None else "")
        + (f"  Composite: {composite:.4f}" if composite is not None else "")
        + f"  ({len(all_results)} files)\nSaved → {out_dir}"
    )
    logger.info("Done. bleu=%.2f comet_da=%s composite=%s", bleu, comet_da, composite)


if __name__ == "__main__":
    main()

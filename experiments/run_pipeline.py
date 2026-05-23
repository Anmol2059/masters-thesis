"""Experiment 3: End-to-end audio → English.

Backends:
  cascaded  — Whisper (ES→ES) + Qwen3/NLLB (ES→EN), with optional domain adaptation
  seamless  — SeamlessM4T v2 (ES audio → EN text directly)
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.asr import Transcriber, SeamlessTranscriber
from src.translator import Translator
from src.metrics import compute_bleu, compute_chrf, compute_comet
from src.utils import setup_logging, make_output_dir, load_cached, save_cached, auto_output_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--reference-dir", required=True, help="Gold EN interpreter transcripts")
    p.add_argument("--source-dir", default=None,
                   help="Gold ES transcripts for COMET source (optional; skips COMET if absent)")
    p.add_argument("--backend", choices=["cascaded", "seamless"], default="cascaded")
    p.add_argument("--condition",
                   choices=["vanilla", "domain", "domain-asr-only", "domain-mt-only"],
                   default="vanilla", help="Domain adaptation (cascaded only)")
    p.add_argument("--glossary", default=None)
    p.add_argument("--asr-model", default="large-v3")
    p.add_argument("--mt-model", choices=["qwen", "nllb"], default="qwen")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--log-dir", default="logs")
    return p.parse_args()


def main():
    args = parse_args()
    condition_tag = args.condition if args.backend == "cascaded" else "e2e"
    tag = f"pipeline_{args.backend}_{condition_tag}"
    logger, ts = setup_logging(args.log_dir, tag)

    out_dir = make_output_dir(args.output_dir or auto_output_dir(tag, ts))
    transcripts_dir = out_dir / "transcripts"
    logger.info("Output dir: %s", out_dir)

    if args.backend == "seamless":
        transcriber = SeamlessTranscriber()
        translator = None
        asr_model_id = SeamlessTranscriber.MODEL_ID
    else:
        domain_asr = args.condition in ("domain", "domain-asr-only")
        domain_mt  = args.condition in ("domain", "domain-mt-only")

        transcriber = Transcriber(model_size=args.asr_model)
        if domain_asr and args.glossary:
            transcriber.load_domain_glossary(args.glossary)

        translator = Translator(model=args.mt_model)
        if domain_mt and args.glossary:
            translator.load_glossary(args.glossary)

        asr_model_id = f"faster-whisper-{args.asr_model}"

    audio_files = sorted(Path(args.audio_dir).glob("*.wav"))
    ref_dir = Path(args.reference_dir)
    src_dir = Path(args.source_dir) if args.source_dir else None

    all_results = []
    skipped = 0

    for audio_file in tqdm(audio_files, desc=f"Pipeline [{args.backend}/{condition_tag}]", unit="file"):
        stem = audio_file.stem
        ref_file = ref_dir / f"{stem}.txt"
        if not ref_file.exists():
            continue

        cached = load_cached(transcripts_dir, stem)
        if cached:
            all_results.append(cached)
            skipped += 1
            continue

        if args.backend == "seamless":
            hypothesis_en = transcriber.translate(str(audio_file))
            asr_es = ""
        else:
            asr_es = transcriber.transcribe(str(audio_file))
            hypothesis_en = translator.translate(asr_es)

        reference_en = ref_file.read_text().strip()
        source_es = (src_dir / f"{stem}.txt").read_text().strip() if src_dir and (src_dir / f"{stem}.txt").exists() else ""

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
        tqdm.write("No results — check --audio-dir and --reference-dir paths.")
        return

    hyps    = [r["hypothesis_en"] for r in all_results]
    refs    = [r["reference_en"]  for r in all_results]
    sources = [r["source_es"]     for r in all_results]

    bleu = compute_bleu(hyps, refs)
    chrf = compute_chrf(hyps, refs)
    comet = compute_comet(sources, hyps, refs) if any(sources) else None

    summary = {
        "backend":    args.backend,
        "condition":  condition_tag,
        "asr_model":  asr_model_id,
        "mt_model":   args.mt_model if args.backend == "cascaded" else "n/a",
        "n_files":    len(all_results),
        "bleu":       bleu,
        "chrf":       chrf,
        "comet":      comet,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    comet_str = f"  COMET: {comet:.4f}" if comet is not None else ""
    tqdm.write(
        f"\nBLEU: {bleu:.2f}  chrF: {chrf:.2f}{comet_str}"
        f"  ({len(all_results)} files)\nSaved → {out_dir}"
    )
    logger.info("Done. bleu=%.2f chrf=%.2f comet=%s", bleu, chrf, comet)


if __name__ == "__main__":
    main()

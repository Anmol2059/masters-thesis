"""Experiment 1: ASR — Whisper vanilla vs domain-prompted, and SeamlessM4T v2 baseline."""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.asr import Transcriber, SeamlessTranscriber
from src.metrics import compute_wer, compute_cer
from src.utils import setup_logging, make_output_dir, load_cached, save_cached, auto_output_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--gold-dir", required=True)
    p.add_argument("--backend", choices=["whisper", "seamless"], default="whisper")
    p.add_argument("--model", default="large-v3", help="Whisper model size (ignored for seamless)")
    p.add_argument("--condition", choices=["vanilla", "domain"], default="vanilla")
    p.add_argument("--glossary", default=None)
    p.add_argument("--output-dir", default=None, help="Output directory (auto-named if omitted)")
    p.add_argument("--log-dir", default="logs")
    return p.parse_args()


def main():
    args = parse_args()
    backend_tag = f"{args.backend}_{args.condition}" if args.backend == "whisper" else "seamless"
    logger, ts = setup_logging(args.log_dir, f"asr_{backend_tag}")

    out_dir = make_output_dir(args.output_dir or auto_output_dir(f"asr_{backend_tag}", ts))
    transcripts_dir = out_dir / "transcripts"
    logger.info("Output dir: %s", out_dir)

    if args.backend == "seamless":
        transcriber = SeamlessTranscriber()
        model_id = SeamlessTranscriber.MODEL_ID
    else:
        transcriber = Transcriber(model_size=args.model)
        if args.condition == "domain" and args.glossary:
            transcriber.load_domain_glossary(args.glossary)
        model_id = f"faster-whisper-{args.model}"

    audio_files = sorted(Path(args.audio_dir).glob("*.wav"))
    gold_dir = Path(args.gold_dir)

    all_results = []
    skipped = 0

    for audio_file in tqdm(audio_files, desc=f"ASR [{backend_tag}]", unit="file"):
        stem = audio_file.stem
        gold_file = gold_dir / f"{stem}.txt"
        if not gold_file.exists():
            continue

        cached = load_cached(transcripts_dir, stem)
        if cached:
            all_results.append(cached)
            skipped += 1
            continue

        hypothesis = transcriber.transcribe(str(audio_file))
        reference = gold_file.read_text().strip()
        entry = {
            "id": stem,
            "hypothesis_es": hypothesis,
            "reference_es": reference,
            "wer": compute_wer(reference, hypothesis),
            "cer": compute_cer(reference, hypothesis),
        }
        save_cached(transcripts_dir, stem, entry)
        all_results.append(entry)
        logger.info("%s  WER=%.4f  CER=%.4f", stem, entry["wer"], entry["cer"])

    tqdm.write(f"Resumed {skipped}/{len(all_results)} files from cache.")

    if not all_results:
        tqdm.write("No results — check audio-dir and gold-dir paths.")
        return

    # Corpus-level WER/CER
    from jiwer import wer as jwer, cer as jcer
    refs = [r["reference_es"] for r in all_results]
    hyps = [r["hypothesis_es"] for r in all_results]
    corpus_wer = jwer(refs, hyps)
    corpus_cer = jcer(refs, hyps)

    summary = {
        "backend": args.backend,
        "model": model_id,
        "condition": args.condition if args.backend == "whisper" else "n/a",
        "n_files": len(all_results),
        "corpus_wer": corpus_wer,
        "corpus_cer": corpus_cer,
        "mean_wer": sum(r["wer"] for r in all_results) / len(all_results),
        "mean_cer": sum(r["cer"] for r in all_results) / len(all_results),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    tqdm.write(
        f"\nCorpus WER: {corpus_wer:.4f}  Corpus CER: {corpus_cer:.4f}"
        f"  ({len(all_results)} files)\nSaved → {out_dir}"
    )
    logger.info("Done. corpus_wer=%.4f  corpus_cer=%.4f", corpus_wer, corpus_cer)


if __name__ == "__main__":
    main()

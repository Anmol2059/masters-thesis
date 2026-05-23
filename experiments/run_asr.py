"""Experiment 1: ASR vanilla vs domain-prompted."""
import argparse
import json
from pathlib import Path

from src.asr import Transcriber
from src.metrics import compute_wer, compute_cer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--gold-dir", required=True)
    p.add_argument("--model", default="large-v3")
    p.add_argument("--condition", choices=["vanilla", "domain"], default="vanilla")
    p.add_argument("--glossary", default=None)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    gold_dir = Path(args.gold_dir)

    transcriber = Transcriber(model_size=args.model)
    if args.condition == "domain" and args.glossary:
        transcriber.load_domain_glossary(args.glossary)

    results = []
    for audio_file in sorted(audio_dir.glob("*.wav")):
        stem = audio_file.stem
        gold_file = gold_dir / f"{stem}.txt"
        if not gold_file.exists():
            continue

        hypothesis = transcriber.transcribe(str(audio_file))
        reference = gold_file.read_text().strip()

        results.append({
            "id": stem,
            "hypothesis": hypothesis,
            "reference": reference,
            "wer": compute_wer(reference, hypothesis),
            "cer": compute_cer(reference, hypothesis),
        })

    output = {
        "condition": args.condition,
        "model": args.model,
        "mean_wer": sum(r["wer"] for r in results) / len(results) if results else 0,
        "mean_cer": sum(r["cer"] for r in results) / len(results) if results else 0,
        "samples": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"WER: {output['mean_wer']:.4f}  CER: {output['mean_cer']:.4f}")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()

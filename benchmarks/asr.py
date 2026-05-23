#!/usr/bin/env python3
"""ASR model comparison benchmark."""
import argparse
import csv
import time
from pathlib import Path

from src.asr import Transcriber


MODELS = {
    "whisper-tiny": "tiny",
    "whisper-small": "small",
    "whisper-medium": "medium",
    "whisper-large-v3": "large-v3",
    "whisper-large-v3-turbo": "large-v3-turbo",
}


def wer(hyp: str, ref: str) -> float:
    from evaluate import load
    metric = load("wer")
    return metric.compute(predictions=[hyp], references=[ref])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True, help="Directory of (audio, ref) pairs")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    p.add_argument("--output", type=Path, default=Path("results/asr_comparison.csv"))
    args = p.parse_args()

    audio_files = sorted(args.dataset.glob("*.wav"))
    ref_files = {f.stem: f.with_suffix(".txt") for f in audio_files}
    rows = []

    for model_name in args.models:
        model_size = MODELS.get(model_name, model_name)
        asr = Transcriber(model_size=model_size, language="es")
        for audio in audio_files:
            ref_path = ref_files[audio.stem]
            if not ref_path.exists():
                continue
            ref_text = ref_path.read_text().strip()
            t0 = time.perf_counter()
            segs = asr.transcribe_file(str(audio))
            latency = time.perf_counter() - t0
            hyp_text = " ".join(s.text for s in segs)
            score = wer(hyp_text, ref_text)
            rows.append({"model": model_name, "file": audio.name, "wer": score, "latency_s": latency})
            print(f"{model_name} | {audio.name} | WER={score:.3f} | {latency:.2f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "file", "wer", "latency_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults → {args.output}")


if __name__ == "__main__":
    main()

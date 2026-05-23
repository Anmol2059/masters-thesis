"""Experiment 3: End-to-end audio → English translation."""
import argparse
import json
from pathlib import Path

from src.asr import Transcriber
from src.translator import Translator
from src.metrics import compute_bleu, compute_chrf, compute_comet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--condition",
                   choices=["vanilla", "domain", "domain-asr-only", "domain-mt-only"],
                   default="vanilla")
    p.add_argument("--glossary", default=None)
    p.add_argument("--asr-model", default="large-v3")
    p.add_argument("--mt-model", default="qwen")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    ref_dir = Path(args.reference_dir)

    domain_asr = args.condition in ("domain", "domain-asr-only")
    domain_mt = args.condition in ("domain", "domain-mt-only")

    transcriber = Transcriber(model_size=args.asr_model)
    if domain_asr and args.glossary:
        transcriber.load_domain_glossary(args.glossary)

    translator = Translator(model=args.mt_model)
    if domain_mt and args.glossary:
        translator.load_glossary(args.glossary)

    hypotheses, references, sources, samples = [], [], [], []

    for audio_file in sorted(audio_dir.glob("*.wav")):
        ref_file = ref_dir / f"{audio_file.stem}.txt"
        if not ref_file.exists():
            continue

        asr_output = transcriber.transcribe(str(audio_file))
        hypothesis = translator.translate(asr_output)
        reference = ref_file.read_text().strip()

        sources.append(asr_output)
        hypotheses.append(hypothesis)
        references.append(reference)
        samples.append({"id": audio_file.stem, "asr": asr_output,
                         "hypothesis": hypothesis, "reference": reference})

    output = {
        "condition": args.condition,
        "asr_model": args.asr_model,
        "mt_model": args.mt_model,
        "bleu": compute_bleu(hypotheses, references),
        "chrf": compute_chrf(hypotheses, references),
        "comet": compute_comet(sources, hypotheses, references),
        "samples": samples,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"BLEU: {output['bleu']:.2f}  chrF: {output['chrf']:.2f}  "
          f"COMET: {output['comet']:.4f}")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()

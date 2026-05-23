#!/usr/bin/env python3
"""Offline evaluation of pre-recorded interpretation sessions."""
import argparse
import json
from pathlib import Path

from src.asr import Transcriber, DomainAdapter
from src.translation import Translator, load_glossary
from src.evaluator import LLMJudge, aggregate


def main() -> None:
    p = argparse.ArgumentParser(description="Offline interpretation evaluation")
    p.add_argument("--source-audio", type=Path, required=True)
    p.add_argument("--interpreter-audio", type=Path, required=True)
    p.add_argument("--domain", default="general")
    p.add_argument("--glossary", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    args = p.parse_args()

    glossary = load_glossary(args.glossary) if args.glossary else {}
    adapter = DomainAdapter(glossary) if glossary else None

    source_asr = Transcriber(
        model_size="large-v3",
        language="es",
        initial_prompt=adapter.build_prompt() if adapter else "",
    )
    interp_asr = Transcriber(model_size="large-v3-turbo", language="en")
    translator = Translator(domain=args.domain, glossary=glossary)
    judge = LLMJudge(glossary=glossary)

    src_segments = source_asr.transcribe_file(str(args.source_audio))
    itp_segments = interp_asr.transcribe_file(str(args.interpreter_audio))

    results = []
    for src, itp in zip(src_segments, itp_segments):
        ref = translator.translate(src.text)
        result = judge.evaluate(src.text, ref, itp.text)
        results.append(result)

    summary = aggregate(results)
    output = {
        "segments": [r.to_dict() for r in results],
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Results written to {args.output}")
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

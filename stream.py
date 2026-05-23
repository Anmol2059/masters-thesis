#!/usr/bin/env python3
"""Real-time streaming entry point."""
import argparse
from pathlib import Path

from src.asr import Transcriber, DomainAdapter
from src.translation import Translator, load_glossary
from src.evaluator import LLMJudge
from src.streaming import Pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="InterpretBench live streaming")
    p.add_argument("--source-device", type=int, default=0)
    p.add_argument("--interp-device", type=int, default=1)
    p.add_argument("--domain", default="general")
    p.add_argument("--glossary", type=Path, default=None)
    p.add_argument("--ui", action="store_true")
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

    pipeline = Pipeline(
        source_asr=source_asr,
        interp_asr=interp_asr,
        translator=translator,
        judge=judge,
        source_device=args.source_device,
        interp_device=args.interp_device,
    )

    if args.ui:
        from src.ui import launch
        launch(pipeline=pipeline)
    else:
        pipeline.on_result = lambda r: print(r.to_dict())
        pipeline.start()
        try:
            import time; time.sleep(3600)
        except KeyboardInterrupt:
            pipeline.stop()


if __name__ == "__main__":
    main()

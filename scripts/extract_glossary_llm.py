"""Auto-extract and translate domain glossary via Qwen LLM."""
import argparse
import json
from pathlib import Path

from src.translator import Translator


EXTRACTION_PROMPT = """You are a terminology expert for EU Parliament discourse.
Given the following Spanish text, extract the top domain-specific terms and
their correct English translations as a JSON object {es_term: en_term}.
Focus on institutional names, procedures, and technical vocabulary.
Return ONLY valid JSON, no explanation.

Text:
{text}"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="Dir of Spanish .txt transcripts")
    p.add_argument("--output", required=True)
    p.add_argument("--sample-docs", type=int, default=20,
                   help="Number of transcripts to sample for extraction")
    return p.parse_args()


def main():
    args = parse_args()
    docs = sorted(Path(args.corpus).glob("*.txt"))[: args.sample_docs]

    translator = Translator(model="qwen")
    combined: dict[str, str] = {}

    for doc in docs:
        text = doc.read_text(encoding="utf-8")[:2000]  # keep prompt short
        prompt = EXTRACTION_PROMPT.format(text=text)
        try:
            raw = translator.translate(prompt)
            parsed = json.loads(raw)
            combined.update(parsed)
        except (json.JSONDecodeError, Exception):
            pass  # skip malformed responses

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    print(f"Extracted {len(combined)} terms → {args.output}")


if __name__ == "__main__":
    main()

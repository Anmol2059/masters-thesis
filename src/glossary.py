"""Glossary loading, formatting, and injection utilities."""
import json
from pathlib import Path


def load(path: str | Path) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_prompt_block(glossary: dict[str, str]) -> str:
    lines = "\n".join(f"- {src} → {tgt}" for src, tgt in glossary.items())
    return f"Use this glossary strictly:\n{lines}"


def to_asr_prompt(glossary: dict[str, str]) -> str:
    return ", ".join(glossary.keys())

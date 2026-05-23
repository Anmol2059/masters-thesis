"""Load and format domain glossaries for prompt injection."""
import json
from pathlib import Path


def load_glossary(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_glossary_for_prompt(glossary: dict, src_lang: str = "ES", tgt_lang: str = "EN") -> str:
    lines = [f"DOMAIN GLOSSARY ({src_lang}→{tgt_lang}):"]
    for src_term, entry in glossary.items():
        tgt_term = entry if isinstance(entry, str) else entry.get("translation", "")
        note = "" if isinstance(entry, str) else entry.get("note", "")
        line = f"  {src_term} → {tgt_term}"
        if note:
            line += f"  [{note}]"
        lines.append(line)
    return "\n".join(lines)

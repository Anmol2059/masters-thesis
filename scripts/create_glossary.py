#!/usr/bin/env python3
"""Interactive helper to build a domain glossary JSON."""
import json
from pathlib import Path


def main() -> None:
    print("InterpretBench — Glossary Builder")
    name = input("Glossary name (e.g. 'finance_es_en'): ").strip()
    output = Path(f"glossaries/{name}.json")
    glossary: dict = {}

    print("Enter terms (blank source term to finish):")
    while True:
        src = input("  Source term (ES): ").strip()
        if not src:
            break
        tgt = input(f"  Translation (EN) for '{src}': ").strip()
        note = input(f"  Note (optional): ").strip()
        glossary[src] = {"translation": tgt, "note": note} if note else {"translation": tgt}

    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(glossary, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(glossary)} terms to {output}")


if __name__ == "__main__":
    main()

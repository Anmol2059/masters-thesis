"""Prompt-based ASR domain adaptation (hotword boosting via Whisper initial_prompt)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union


class DomainAdapter:
    """Build an initial_prompt string from a domain glossary to bias Whisper recognition."""

    def __init__(self, glossary: Union[str, Path, dict]) -> None:
        if isinstance(glossary, dict):
            self._glossary = glossary
        else:
            with open(glossary, encoding="utf-8") as f:
                self._glossary = json.load(f)

    def build_prompt(self, max_terms: int = 30) -> str:
        """Return a comma-separated term list Whisper uses as context."""
        terms = list(self._glossary.keys())[:max_terms]
        return ", ".join(terms)

    @property
    def glossary(self) -> dict:
        return self._glossary

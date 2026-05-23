"""LLM-based domain-aware Spanish→English translator (Qwen2.5-7B-Instruct)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .glossary_loader import format_glossary_for_prompt

_SYSTEM_TEMPLATE = """\
You are a specialist Spanish→English translator for {domain} discourse.
Translate faithfully, preserving register, tone, and formality.

{glossary_block}

Return ONLY the English translation. No explanations."""


@dataclass
class Translator:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    domain: str = "general"
    glossary: dict = field(default_factory=dict)
    max_new_tokens: int = 512
    _pipe: object = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline
        self._pipe = pipeline(
            "text-generation",
            model=self.model_id,
            device_map="auto",
            max_new_tokens=self.max_new_tokens,
        )

    @property
    def _system_prompt(self) -> str:
        glossary_block = format_glossary_for_prompt(self.glossary) if self.glossary else ""
        return _SYSTEM_TEMPLATE.format(domain=self.domain, glossary_block=glossary_block)

    def translate(self, text: str) -> str:
        self._load()
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": text},
        ]
        out = self._pipe(messages)
        return out[0]["generated_text"][-1]["content"].strip()

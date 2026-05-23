"""LLM-as-judge: scores interpretation quality across 4 dimensions."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .metrics import EvaluationResult
from ..translation.glossary_loader import format_glossary_for_prompt

_JUDGE_SYSTEM = """\
You are an expert interpreter quality evaluator. Given:
  - source_text: original Spanish utterance
  - reference: machine English translation (domain-adapted)
  - interpretation: human interpreter's English output
  - glossary: domain terminology reference

Score the interpretation on 4 dimensions (0–100 each) and return a JSON object:
{
  "accuracy": <int>,        // semantic faithfulness to source
  "fluency": <int>,         // grammaticality and naturalness
  "pragmatic": <int>,       // register, tone, cultural adaptation
  "terminology": <int>,     // correct use of domain glossary terms
  "omissions": [<str>],     // list of omitted content
  "additions": [<str>],     // hallucinated content not in source
  "notes": "<str>"          // brief evaluator comment
}
Return ONLY the JSON object, no other text."""


@dataclass
class LLMJudge:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    glossary: dict = field(default_factory=dict)
    _pipe: object = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline
        self._pipe = pipeline(
            "text-generation",
            model=self.model_id,
            device_map="auto",
            max_new_tokens=512,
        )

    def evaluate(
        self,
        source_text: str,
        reference: str,
        interpretation: str,
    ) -> EvaluationResult:
        self._load()
        glossary_block = format_glossary_for_prompt(self.glossary) if self.glossary else "N/A"
        user_msg = json.dumps({
            "source_text": source_text,
            "reference": reference,
            "interpretation": interpretation,
            "glossary": glossary_block,
        }, ensure_ascii=False, indent=2)

        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        out = self._pipe(messages)
        raw = out[0]["generated_text"][-1]["content"].strip()

        return EvaluationResult.from_json(raw)

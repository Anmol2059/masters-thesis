"""NLLB-200 baseline translator (no domain adaptation)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NLLBTranslator:
    model_id: str = "facebook/nllb-200-distilled-600M"
    src_lang: str = "spa_Latn"
    tgt_lang: str = "eng_Latn"
    _pipe: object = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline
        self._pipe = pipeline(
            "translation",
            model=self.model_id,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            device_map="auto",
        )

    def translate(self, text: str) -> str:
        self._load()
        out = self._pipe(text, max_length=512)
        return out[0]["translation_text"].strip()

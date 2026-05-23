"""NLLB-200-3.3B translation wrapper (ES → EN)."""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

NLLB_MODEL = "facebook/nllb-200-3.3B"


class Translator:
    def __init__(self, device: int = 0):
        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
        self._pipe = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            src_lang="spa_Latn",
            tgt_lang="eng_Latn",
            device=device,
            max_length=512,
        )

    def translate(self, text: str) -> str:
        return self._pipe(text)[0]["translation_text"]

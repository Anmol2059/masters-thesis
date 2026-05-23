"""Qwen3-8B and NLLB-3.3B translation wrappers with glossary injection."""
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


QWEN_MODEL = "Qwen/Qwen3-8B"
NLLB_MODEL = "facebook/nllb-200-3.3B"

VANILLA_SYSTEM = "Translate the following Spanish text to English."


def _build_glossary_system(glossary: dict[str, str]) -> str:
    lines = "\n".join(f"- {src} → {tgt}" for src, tgt in glossary.items())
    return f"Translate Spanish to English. Use this glossary strictly:\n{lines}"


class Translator:
    def __init__(self, model: str = "qwen"):
        self.model_name = model
        self.glossary: dict[str, str] = {}
        self._pipe: Any = None
        self._load(model)

    def _load(self, model: str) -> None:
        if model == "nllb":
            tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
            m = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
            self._pipe = pipeline("translation", model=m, tokenizer=tokenizer,
                                   src_lang="spa_Latn", tgt_lang="eng_Latn",
                                   device=0, max_length=512)
        elif model == "qwen":
            import torch
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
            self._llm = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL, quantization_config=bnb, device_map="auto"
            )
        else:
            raise ValueError(f"Unknown model: {model}")

    def load_glossary(self, path: str) -> None:
        self.glossary = json.loads(Path(path).read_text())

    def translate(self, text: str) -> str:
        if self.model_name == "nllb":
            result = self._pipe(text)
            return result[0]["translation_text"]

        system = (_build_glossary_system(self.glossary)
                  if self.glossary else VANILLA_SYSTEM)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": text}]
        # enable_thinking=False disables Qwen3's chain-of-thought for faster inference
        ids = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", enable_thinking=False
        ).to(self._llm.device)
        out = self._llm.generate(ids, max_new_tokens=512, do_sample=False)
        decoded = self._tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        return decoded.strip()

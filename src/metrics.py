"""Evaluation metrics: WER, CER, BLEU, chrF, COMET, COMET-Kiwi, TermAcc, composite score."""
from __future__ import annotations
import os
import re

import sacrebleu
from jiwer import wer, cer
from comet import download_model, load_from_checkpoint

# Ensure HF token is visible even when HF_HOME is overridden to a non-home path
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)

_comet_da = None
_comet_kiwi = None


def _get_comet_da():
    global _comet_da
    if _comet_da is None:
        path = download_model("Unbabel/wmt22-comet-da")
        _comet_da = load_from_checkpoint(path)
    return _comet_da


def _get_comet_kiwi():
    global _comet_kiwi
    if _comet_kiwi is None:
        path = download_model("Unbabel/wmt22-cometkiwi-da")
        _comet_kiwi = load_from_checkpoint(path)
    return _comet_kiwi


def compute_wer(reference: str, hypothesis: str) -> float:
    return wer(reference, hypothesis)


def compute_cer(reference: str, hypothesis: str) -> float:
    return cer(reference, hypothesis)


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    return sacrebleu.corpus_bleu(hypotheses, [references]).score


def compute_chrf(hypotheses: list[str], references: list[str]) -> float:
    return sacrebleu.corpus_chrf(hypotheses, [references]).score


def compute_comet(sources: list[str], hypotheses: list[str],
                  references: list[str]) -> float:
    """Interpreter alignment — COMET-DA against interpreter reference (semantic)."""
    model = _get_comet_da()
    data = [{"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)]
    return float(model.predict(data, batch_size=8, gpus=1).system_score)


def compute_comet_kiwi(sources: list[str], hypotheses: list[str]) -> float:
    """Faithfulness score — COMET-Kiwi QE against source (no reference needed)."""
    model = _get_comet_kiwi()
    data = [{"src": s, "mt": h} for s, h in zip(sources, hypotheses)]
    return float(model.predict(data, batch_size=8, gpus=1).system_score)


def compute_term_accuracy(hypotheses: list[str], glossary: dict[str, str]) -> float:
    if not glossary:
        return 0.0
    hits = total = 0
    for hyp in hypotheses:
        hyp_lower = hyp.lower()
        for tgt in glossary.values():
            total += 1
            if re.search(r"\b" + re.escape(tgt.lower()) + r"\b", hyp_lower):
                hits += 1
    return hits / total if total else 0.0


def compute_composite(
    comet_da: float,
    comet_kiwi: float,
    term_accuracy: float,
    bleu: float,
    w_comet_da: float = 0.35,
    w_comet_kiwi: float = 0.30,
    w_term: float = 0.20,
    w_bleu: float = 0.15,
) -> float:
    """Composite score (0–1 scale). Weights documented in docs/evaluation.md."""
    bleu_norm = bleu / 100.0
    return (w_comet_da   * comet_da
          + w_comet_kiwi * comet_kiwi
          + w_term       * term_accuracy
          + w_bleu       * bleu_norm)

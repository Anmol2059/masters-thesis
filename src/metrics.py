"""WER, CER, BLEU, chrF, COMET, term accuracy."""
from __future__ import annotations

import re

import sacrebleu
from jiwer import wer, cer
from comet import download_model, load_from_checkpoint

_comet_model = None


def _get_comet():
    global _comet_model
    if _comet_model is None:
        path = download_model("Unbabel/wmt22-comet-da")
        _comet_model = load_from_checkpoint(path)
    return _comet_model


def compute_wer(reference: str, hypothesis: str) -> float:
    return wer(reference, hypothesis)


def compute_cer(reference: str, hypothesis: str) -> float:
    return cer(reference, hypothesis)


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return result.score


def compute_chrf(hypotheses: list[str], references: list[str]) -> float:
    result = sacrebleu.corpus_chrf(hypotheses, [references])
    return result.score


def compute_comet(sources: list[str], hypotheses: list[str],
                  references: list[str]) -> float:
    model = _get_comet()
    data = [{"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)]
    scores = model.predict(data, batch_size=8, gpus=1)
    return float(scores.system_score)


def compute_term_accuracy(hypotheses: list[str],
                          glossary: dict[str, str]) -> float:
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

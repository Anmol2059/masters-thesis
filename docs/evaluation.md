# Evaluation Protocol

## Motivation

Evaluating speech translation against professional interpreter references is fundamentally different from standard MT evaluation. Interpreters do not produce word-for-word translations — they re-package meaning for real-time delivery, making surface-level metrics like BLEU systematically underestimate quality. A machine translation that is more literal than a human interpreter will score low on BLEU while potentially being more faithful to the source.

We therefore propose a **multi-faceted evaluation protocol** that disentangles what each metric actually measures, with weights justified by the unique semantics of this task.

---

## Metrics

### 1. Interpreter Alignment — COMET-DA (`comet_da`)

```
COMET-DA(source_ES, hypothesis_EN, interpreter_EN)
Model: Unbabel/wmt22-comet-da
```

Measures semantic similarity between the machine output and the interpreter reference. This is the primary quality signal — it captures meaning-level agreement without penalising legitimate paraphrases. Used as the main quality metric throughout the literature.

**Caveat:** High COMET-DA does not guarantee source faithfulness — it only tells us how close we are to what the interpreter said, not whether the interpreter was accurate themselves.

---

### 2. Faithfulness — COMET-Kiwi (`comet_kiwi`)

```
COMET-Kiwi(source_ES, hypothesis_EN)   [reference-free / QE]
Model: Unbabel/wmt22-cometkiwi-da
```

Measures how faithfully the hypothesis preserves the meaning of the Spanish source, without using any reference. This is the only metric that directly evaluates source fidelity and is robust to interpreter paraphrase style.

**Why this matters:** If SeamlessM4T scores high on COMET-DA but low on COMET-Kiwi, it has learned to imitate interpreter style but may be missing source content. The inverse reveals the opposite failure mode.

---

### 3. Terminology Accuracy — TermAcc (`term_acc`)

```
TermAcc = proportion of IATE-listed target terms present in hypothesis
Glossary: glossaries/eu_parliament_es_en.json  (IATE-derived)
```

EU Parliament speech is dense with domain-specific terminology (rapporteur, codecision, comitology, ECB, GDP) where mistranslation carries disproportionate impact. TermAcc measures whether these high-stakes terms are rendered correctly, independently of overall fluency.

**Expected finding:** Cascaded + NLLB may tie or lose on overall fluency metrics against SeamlessM4T, but the term accuracy comparison is where the modularity of cascaded systems becomes legible. SeamlessM4T's prior EU Parliament training (Europarl corpus) may give it a baseline advantage here — this is explicitly documented in the discussion.

---

### 4. Surface Overlap — BLEU / chrF (`bleu`, `chrf`)

Standard corpus-level metrics included for comparability with prior work. Treated as secondary — interpreter references make BLEU structurally pessimistic for any system (including human translators compared against a different interpreter).

chrF is preferred over BLEU in low-resource or specialist-domain settings because it operates at character level and is more robust to morphological variation.

---

## Composite Score

```
Composite = 0.35 × COMET-DA
           + 0.30 × COMET-Kiwi
           + 0.20 × TermAcc
           + 0.15 × (BLEU / 100)
```

### Weight justification

| Metric | Weight | Rationale |
|--------|--------|-----------|
| COMET-DA (interpreter alignment) | 0.35 | Primary quality signal; semantic, robust to paraphrase |
| COMET-Kiwi (source faithfulness) | 0.30 | Only metric tying output to source; critical for MT eval |
| TermAcc (terminology) | 0.20 | Domain-critical; errors here have outsized real-world impact |
| BLEU/100 (surface overlap) | 0.15 | Comparability; down-weighted due to interpreter reference bias |

Weights sum to 1.0. All sub-scores are in [0, 1] range (BLEU is divided by 100). The composite is not used for primary claims — it is a diagnostic summary for ranking conditions when metrics disagree.

---

## SeamlessM4T Domain Advantage — Discussion Note

SeamlessM4T v2 (Meta, 2023) was trained on the **SeamlessAlign** corpus, which includes Europarl — the EU Parliament's multilingual proceedings. This gives SeamlessM4T an inherent domain advantage on EPIC v2.0 data that the cascaded system (Whisper + NLLB) does not have.

This must be acknowledged as a confound. It does not invalidate the comparison — both systems are evaluated as-is, without fine-tuning — but it means SeamlessM4T's scores reflect both architectural advantage (end-to-end joint training) and data advantage (in-domain pre-training). Disentangling these is left for future work.

---

## Thesis Contribution

This evaluation protocol is, to our knowledge, the first to:

1. Apply COMET-Kiwi (reference-free QE) alongside COMET-DA for speech translation against interpreter references
2. Explicitly separate **source faithfulness** (COMET-Kiwi) from **interpreter alignment** (COMET-DA) — a distinction that matters uniquely when references are interpretations rather than translations
3. Use IATE-derived terminology for TermAcc rather than a manually curated glossary
4. Propose a weighted composite score with documented, task-motivated weights for this specific evaluation setting

The protocol is designed to be reusable for any system evaluated against interpreter corpora.

# scripts/

Utility scripts for data preparation, model download, and glossary construction.

| Script | Purpose |
|--------|---------|
| `download_models.py` | Pull Whisper, Qwen, NLLB, COMET from HuggingFace Hub |
| `prepare_epic.py` | Clean EPIC transcripts (strip markup), pair with audio, write `epic_processed/` |
| `extract_glossary_tfidf.py` | Auto-build ES→EN glossary via TF-IDF over EPIC corpus |
| `extract_glossary_llm.py` | Auto-build ES→EN glossary by prompting Qwen |

Run order:

```bash
python data/download_epic.py        # 1. download corpus
python scripts/prepare_epic.py      # 2. clean + pair
python scripts/download_models.py   # 3. download models
# then run experiments/
```

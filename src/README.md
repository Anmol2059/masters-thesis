# src/

Core library modules shared across all experiments.

| Module | Purpose |
|--------|---------|
| `asr.py` | Whisper wrapper — loads `faster-whisper`, applies optional domain `initial_prompt` |
| `translator.py` | Qwen2.5-7B and NLLB-600M wrappers — glossary injected into Qwen system prompt |
| `glossary.py` | Load JSON glossary, format as prompt block or ASR prompt string |
| `metrics.py` | WER, CER (jiwer), BLEU, chrF (sacrebleu), COMET (unbabel-comet), term accuracy |
| `epic_parser.py` | Strip EPIC v2.0 transcript markup → clean plain text |

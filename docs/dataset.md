# Dataset: EPIC v2.0

**European Parliament Interpreting Corpus v2.0**
Lobascio, Liu & Russo (2024) — [Zenodo record 13856205](https://zenodo.org/records/13856205)

---

## What We Use (ES→EN direction only)

| Component | Files | Total duration | Avg per file |
|-----------|-------|---------------|--------------|
| Spanish source audio (WAV, 16 kHz mono) | 130 | ~6.3 h (377 min) | ~2.9 min |
| Spanish source transcripts (gold) | 130 | — | — |
| English interpreter transcripts (gold) | 130 | — | — |

Duration range: 0.15 min – 25.2 min per speech.

We do **not** use: Italian data, POS tags, alignment files, or video files (audio extracted via ffmpeg).

---

## EPIC File Naming Convention

```
epic_st_{date}-{session}-{id}-org-{lang}.mp4   ← source speech video
epic_tt_{date}-{session}-{id}-int-{src}-{tgt}.wav  ← interpreter audio
```

- `st` = source talk, `tt` = target talk (interpretation)
- `org` = original, `int` = interpretation
- `{lang}` = `es` / `en` / `it`

For this thesis we select all files where `lang=es` (source) and `src-tgt=es-en` (interpreter).

---

## Raw Directory Layout (after download)

```
data/epic_raw/
├── audio/                          # 130 × WAV extracted from source MP4s
│   └── epic_st_*-org-es.wav
├── transcripts/
│   └── 05_transcripts_v2.0/
│       ├── source/                 # Gold Spanish transcripts (with EPIC markup)
│       │   └── epic_st_*-org-es.txt
│       └── target/                 # Gold English interpreter transcripts
│           └── epic_tt_*-int-es-en.txt
├── recordings/
│   └── 06_recordings_v2.0/
│       ├── source/                 # Original MP4 videos (Spanish speeches)
│       └── target/                 # Interpreter WAVs (not used directly)
├── manifest.json                   # Pairs audio ↔ transcript by speech ID
├── 04_metadata_v2.0.zip
├── 05_transcripts_v2.0.zip
└── 06_recordings_v2.0.zip
```

---

## EPIC Transcript Markup

Raw transcripts contain spoken-language annotation that must be stripped before use:

| Markup | Meaning | Example |
|--------|---------|---------|
| `(..)` / `(1.02)` | Pause (seconds) | `(.)` short pause |
| `//` | False start / self-interruption | `I think // I believe` |
| `-ehm-` | Filled hesitation | `-eeh-`, `-mmm-` |
| `~word </correct/>` | Self-correction (keep correction) | `~webs_site </website/>` |

Cleaned by `scripts/prepare_epic.py` → stored in `data/epic_processed/`.

---

## Processed Directory Layout (after prepare_epic.py)

```
data/epic_processed/
├── audio/                          # Symlinked or copied from epic_raw/audio/
├── transcripts_es/                 # Cleaned Spanish transcripts (plain text)
├── transcripts_en_interp/          # Cleaned English interpreter transcripts
└── manifest.json                   # Full metadata + file paths
```

---

## Download

```bash
python data/download_epic.py --data-dir data/epic_raw
```

Resumable — safe to re-run after any interruption. See `data/download_epic.py` for details.

Manual download (browser required for large files):
- Transcripts (3.5 MB): https://zenodo.org/records/13856205/files/05_transcripts_v2.0.zip
- Recordings (7.5 GB): https://zenodo.org/records/13856205/files/06_recordings_v2.0.zip

---

## License

EPIC v2.0 is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Please cite the corpus if you use it:

```bibtex
@dataset{lobascio2024epic,
  title     = {EPIC v2.0: European Parliament Interpreting Corpus},
  author    = {Lobascio, Tomaso and Liu, Xuankai and Russo, Mariachiara},
  year      = {2024},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.13856205}
}
```

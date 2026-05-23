"""Clean EPIC v2.0 transcripts and build processed dataset for ES→EN experiments.

Expected epic_raw layout (after download):
  epic_raw/
    audio/                              ← 130 WAV files already extracted
      epic_st_{id}-org-es.wav
    transcripts/
      05_transcripts_v2.0/
        source/
          epic_st_{id}-org-es.txt       ← gold Spanish transcripts
        target/
          epic_tt_{id}-int-es-en.txt    ← gold English interpreter transcripts

Output (epic_processed):
  audio/                 ← symlinks to epic_raw/audio WAVs
  transcripts_es/        ← cleaned Spanish transcripts  (stem = audio stem)
  transcripts_en_interp/ ← cleaned English transcripts  (stem = audio stem)
  manifest.json
"""
import argparse
import json
from pathlib import Path

from src.epic_parser import parse_epic_transcript


def extract_speech_id(audio_stem: str) -> str:
    """epic_st_01-04-04-m-006-org-es  →  01-04-04-m-006"""
    return audio_stem.removeprefix("epic_st_").removesuffix("-org-es")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epic-dir", required=True, help="Path to epic_raw/")
    p.add_argument("--output-dir", required=True, help="Path to epic_processed/")
    return p.parse_args()


def main():
    args = parse_args()
    epic = Path(args.epic_dir)
    out = Path(args.output_dir)

    audio_out = out / "audio"
    es_out    = out / "transcripts_es"
    en_out    = out / "transcripts_en_interp"
    for d in (audio_out, es_out, en_out):
        d.mkdir(parents=True, exist_ok=True)

    src_transcript_dir = epic / "transcripts" / "05_transcripts_v2.0" / "source"
    tgt_transcript_dir = epic / "transcripts" / "05_transcripts_v2.0" / "target"

    if not src_transcript_dir.exists():
        raise FileNotFoundError(f"Source transcripts not found: {src_transcript_dir}")
    if not tgt_transcript_dir.exists():
        raise FileNotFoundError(f"Target transcripts not found: {tgt_transcript_dir}")

    audio_files = sorted((epic / "audio").glob("epic_st_*-org-es.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {epic / 'audio'}")

    manifest = []
    skipped = 0

    for wav in audio_files:
        speech_id = extract_speech_id(wav.stem)
        stem = wav.stem  # e.g. epic_st_01-04-04-m-006-org-es

        es_src  = src_transcript_dir / f"epic_st_{speech_id}-org-es.txt"
        en_src  = tgt_transcript_dir / f"epic_tt_{speech_id}-int-es-en.txt"

        if not es_src.exists():
            print(f"  [skip] no ES transcript: {es_src.name}")
            skipped += 1
            continue
        if not en_src.exists():
            print(f"  [skip] no EN interpreter: {en_src.name}")
            skipped += 1
            continue

        # Symlink audio (avoid copying 130 WAVs)
        link = audio_out / wav.name
        if not link.exists():
            link.symlink_to(wav.resolve())

        # Clean and write transcripts — use audio stem as filename so
        # run_pipeline.py can match audio → transcript by stem
        es_dst = es_out / f"{stem}.txt"
        en_dst = en_out / f"{stem}.txt"
        if not es_dst.exists():
            es_dst.write_text(parse_epic_transcript(es_src), encoding="utf-8")
        if not en_dst.exists():
            en_dst.write_text(parse_epic_transcript(en_src), encoding="utf-8")

        manifest.append({
            "id":            stem,
            "speech_id":     speech_id,
            "audio":         str(link),
            "transcript_es": str(es_dst),
            "transcript_en": str(en_dst),
        })

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Prepared {len(manifest)} speech pairs → {out}  (skipped {skipped})")


if __name__ == "__main__":
    main()

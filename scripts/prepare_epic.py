"""Extract EPIC v2.0 audio and clean transcripts for ES→EN experiments."""
import argparse
import json
import subprocess
from pathlib import Path

from src.epic_parser import parse_epic_transcript


def extract_audio(video_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vn", "-ac", "1", "-ar", "16000", str(out_path)],
        check=True, capture_output=True,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epic-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--direction", default="es-en")
    return p.parse_args()


def main():
    args = parse_args()
    epic = Path(args.epic_dir)
    out = Path(args.output_dir)

    audio_out = out / "audio"
    es_out = out / "transcripts_es"
    en_out = out / "transcripts_en_interp"
    for d in (audio_out, es_out, en_out):
        d.mkdir(parents=True, exist_ok=True)

    manifest = []

    # Locate Spanish transcripts — adjust glob to match actual EPIC layout
    for es_txt in sorted((epic / "transcripts").glob("*_ES_*.txt")):
        speech_id = es_txt.stem
        en_txt = epic / "transcripts" / f"{speech_id}_EN.txt"
        video = next((epic / "recordings").glob(f"{speech_id}*"), None)

        if not en_txt.exists() or video is None:
            continue

        wav = audio_out / f"{speech_id}.wav"
        extract_audio(video, wav)

        (es_out / f"{speech_id}.txt").write_text(
            parse_epic_transcript(es_txt), encoding="utf-8"
        )
        (en_out / f"{speech_id}.txt").write_text(
            parse_epic_transcript(en_txt), encoding="utf-8"
        )

        manifest.append({"id": speech_id, "audio": str(wav),
                          "transcript_es": str(es_out / f"{speech_id}.txt"),
                          "transcript_en": str(en_out / f"{speech_id}.txt")})

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Prepared {len(manifest)} speech pairs → {out}")


if __name__ == "__main__":
    main()

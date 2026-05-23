#!/usr/bin/env python3
"""
Download EPIC v2.0 from Zenodo and extract ES audio.

Resumable: re-run at any point after a failure and it picks up where it left off.
  - Partial zip downloads resume via HTTP Range requests
  - Already-extracted audio files are skipped
  - Already-unzipped transcript dirs are skipped

Usage:
    python data/download_epic.py [--data-dir data/epic_raw] [--direction es-en]
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ZENODO_BASE = "https://zenodo.org/records/13856205/files"

FILES = {
    "transcripts": {
        "url": f"{ZENODO_BASE}/05_transcripts_v2.0.zip",
        "filename": "05_transcripts_v2.0.zip",
        "size_mb": 3.5,
    },
    "recordings": {
        "url": f"{ZENODO_BASE}/06_recordings_v2.0.zip",
        "filename": "06_recordings_v2.0.zip",
        "size_mb": 7500,
    },
    "metadata": {
        "url": f"{ZENODO_BASE}/04_metadata_v2.0.zip",
        "filename": "04_metadata_v2.0.zip",
        "size_mb": 0.5,
    },
}

# EPIC v2.0 source language codes used in filenames (adjust if layout differs)
ES_PATTERNS = ["_ES_", "/ES/", "ES-EN", "es_en"]


def log(msg: str) -> None:
    print(f"[download_epic] {msg}", flush=True)


def resume_download(url: str, dest: Path) -> None:
    """Download with resume support via wget -c (falls back to curl -C -)."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Try wget first (cleaner progress output)
    wget = subprocess.run(["which", "wget"], capture_output=True)
    if wget.returncode == 0:
        cmd = ["wget", "-c", "--show-progress", "-q", "-O", str(dest), url]
        tool = "wget"
    else:
        cmd = ["curl", "-L", "-C", "-", "--progress-bar", "-o", str(dest), url]
        tool = "curl"

    log(f"Downloading {dest.name} via {tool} (resumable) ...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log(f"ERROR: {tool} exited with code {result.returncode}")
        sys.exit(1)
    log(f"Done: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB on disk)")


def unzip_if_needed(zip_path: Path, extract_to: Path, marker: str) -> None:
    """Unzip only if the marker directory/file doesn't already exist."""
    marker_path = extract_to / marker
    if marker_path.exists():
        log(f"Already extracted: {zip_path.name} → skipping")
        return
    extract_to.mkdir(parents=True, exist_ok=True)
    log(f"Extracting {zip_path.name} ...")
    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(extract_to)], check=True)
    log(f"Extracted → {extract_to}")


def is_es_source(path: Path) -> bool:
    """Return True if this recording looks like a Spanish source file."""
    name = path.name.upper()
    parent = str(path.parent).upper()
    return any(pat.upper() in name or pat.upper() in parent for pat in ES_PATTERNS)


def extract_audio(video: Path, out_wav: Path) -> None:
    """Extract 16 kHz mono WAV from video using ffmpeg."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            str(out_wav),
        ],
        check=True,
        capture_output=True,
    )


def extract_all_audio(recordings_dir: Path, audio_out: Path, direction: str) -> None:
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    videos = [p for p in recordings_dir.rglob("*") if p.suffix.lower() in video_exts]

    if not videos:
        log(f"WARNING: no video files found under {recordings_dir}")
        return

    log(f"Found {len(videos)} video file(s) — extracting ES audio only ...")
    skipped = done = 0

    for vid in sorted(videos):
        if not is_es_source(vid):
            continue
        wav = audio_out / f"{vid.stem}.wav"
        if wav.exists():
            skipped += 1
            continue
        try:
            extract_audio(vid, wav)
            done += 1
            log(f"  ✓ {wav.name}")
        except subprocess.CalledProcessError as e:
            log(f"  ✗ ffmpeg failed for {vid.name}: {e}")

    log(f"Audio extraction complete: {done} new, {skipped} already existed")


def write_manifest(data_dir: Path) -> None:
    transcript_dir = data_dir / "transcripts"
    audio_dir = data_dir / "audio"

    entries = []
    if audio_dir.exists():
        for wav in sorted(audio_dir.glob("*.wav")):
            stem = wav.stem
            es_txt = next(transcript_dir.rglob(f"{stem}*.txt"), None) if transcript_dir.exists() else None
            entries.append({
                "id": stem,
                "audio": str(wav),
                "transcript_es": str(es_txt) if es_txt else None,
            })

    manifest = data_dir / "manifest.json"
    manifest.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    log(f"Manifest written: {len(entries)} entries → {manifest}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/epic_raw",
                   help="Where to store raw downloads (default: data/epic_raw)")
    p.add_argument("--direction", default="es-en",
                   help="Language direction (currently only es-en supported)")
    p.add_argument("--skip-recordings", action="store_true",
                   help="Download transcripts + metadata only (no 7.5 GB recordings)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    log(f"Data directory: {data_dir.resolve()}")
    log(f"Direction: {args.direction}")

    # ── 1. Download ──────────────────────────────────────────────────────────
    for key, info in FILES.items():
        if key == "recordings" and args.skip_recordings:
            log("Skipping recordings (--skip-recordings set)")
            continue
        dest = data_dir / info["filename"]
        if dest.exists() and dest.stat().st_size > 1_000:
            log(f"Already downloaded: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        else:
            log(f"Need: {dest.name} (~{info['size_mb']} MB)")
            resume_download(info["url"], dest)

    # ── 2. Extract transcripts ───────────────────────────────────────────────
    transcripts_zip = data_dir / FILES["transcripts"]["filename"]
    if transcripts_zip.exists():
        unzip_if_needed(transcripts_zip, data_dir / "transcripts", marker=".")

    # ── 3. Extract recordings + pull audio ──────────────────────────────────
    recordings_zip = data_dir / FILES["recordings"]["filename"]
    if recordings_zip.exists() and not args.skip_recordings:
        recordings_dir = data_dir / "recordings"
        unzip_if_needed(recordings_zip, recordings_dir, marker=".")
        extract_all_audio(recordings_dir, data_dir / "audio", args.direction)

    # ── 4. Write manifest ────────────────────────────────────────────────────
    write_manifest(data_dir)

    log("All done.")


if __name__ == "__main__":
    main()

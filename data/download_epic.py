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
import json
import subprocess
import sys
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_BASE = "https://zenodo.org/records/13856205/files"

FILES = {
    "metadata": {
        "url": f"{ZENODO_BASE}/04_metadata_v2.0.zip",
        "filename": "04_metadata_v2.0.zip",
        "size_mb": 0.5,
    },
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
}

# EPIC v2.0 naming: source speeches are epic_st_*-org-{lang}.mp4 under source/
# We want Spanish originals: org-es.mp4

CHUNK = 1 << 20  # 1 MB chunks


def log(msg: str) -> None:
    print(f"[download_epic] {msg}", flush=True)


def resume_download(url: str, dest: Path) -> None:
    """Streaming download with byte-range resume and tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    r = requests.get(url, headers=headers, stream=True, timeout=30)

    # 416 = server says "range not satisfiable" → file already complete
    if r.status_code == 416:
        log(f"Already complete: {dest.name}")
        return
    r.raise_for_status()

    total = existing + int(r.headers.get("Content-Length", 0))
    mode = "ab" if existing else "wb"

    log(f"{'Resuming' if existing else 'Downloading'} {dest.name} "
        f"({total / 1e9:.2f} GB total, {existing / 1e6:.1f} MB already done)")

    with open(dest, mode) as f, tqdm(
        total=total,
        initial=existing,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
        dynamic_ncols=True,
    ) as bar:
        for chunk in r.iter_content(chunk_size=CHUNK):
            f.write(chunk)
            bar.update(len(chunk))

    log(f"Saved: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def unzip_if_needed(zip_path: Path, extract_to: Path) -> None:
    done_marker = extract_to / ".unzipped"
    if done_marker.exists():
        log(f"Already extracted: {zip_path.name} — skipping")
        return
    extract_to.mkdir(parents=True, exist_ok=True)
    log(f"Extracting {zip_path.name} ...")
    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(extract_to)], check=True)
    done_marker.touch()
    log(f"Extracted → {extract_to}")


def is_es_source(path: Path) -> bool:
    # EPIC source files: epic_st_*-org-es.mp4 inside source/
    return path.name.endswith("-org-es.mp4")


def extract_audio(video: Path, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(out_wav)],
        check=True, capture_output=True,
    )


def extract_all_audio(recordings_dir: Path, audio_out: Path) -> None:
    # ES source videos live under source/ subdirectory
    source_dir = recordings_dir / "06_recordings_v2.0" / "source"
    search_root = source_dir if source_dir.exists() else recordings_dir
    videos = [p for p in search_root.rglob("*.mp4") if is_es_source(p)]

    if not videos:
        log(f"WARNING: no ES video files found under {recordings_dir}")
        return

    log(f"Found {len(videos)} ES video(s) — extracting audio ...")
    skipped = errors = 0

    for vid in tqdm(sorted(videos), desc="ffmpeg audio", unit="file", dynamic_ncols=True):
        wav = audio_out / f"{vid.stem}.wav"
        if wav.exists():
            skipped += 1
            continue
        try:
            extract_audio(vid, wav)
        except subprocess.CalledProcessError as e:
            log(f"  ffmpeg failed: {vid.name} — {e}")
            errors += 1

    done = len(videos) - skipped - errors
    log(f"Audio done: {done} extracted, {skipped} skipped, {errors} errors")


def write_manifest(data_dir: Path) -> None:
    transcript_dir = data_dir / "transcripts"
    audio_dir = data_dir / "audio"
    entries = []
    if audio_dir.exists():
        for wav in sorted(audio_dir.glob("*.wav")):
            es_txt = next(transcript_dir.rglob(f"{wav.stem}*.txt"), None) \
                if transcript_dir.exists() else None
            entries.append({
                "id": wav.stem,
                "audio": str(wav),
                "transcript_es": str(es_txt) if es_txt else None,
            })
    manifest = data_dir / "manifest.json"
    manifest.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    log(f"Manifest: {len(entries)} entries → {manifest}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/epic_raw")
    p.add_argument("--direction", default="es-en")
    p.add_argument("--skip-recordings", action="store_true",
                   help="Download transcripts + metadata only (skips 7.5 GB recordings)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    log(f"Data dir : {data_dir.resolve()}")
    log(f"Direction: {args.direction}")

    # ── 1. Download ──────────────────────────────────────────────────────────
    for key, info in FILES.items():
        if key == "recordings" and args.skip_recordings:
            log("Skipping recordings (--skip-recordings)")
            continue
        dest = data_dir / info["filename"]
        # Consider complete if size looks right (within 1 MB of expected)
        if dest.exists() and dest.stat().st_size >= (info["size_mb"] - 1) * 1e6:
            log(f"Already downloaded: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        else:
            resume_download(info["url"], dest)

    # ── 2. Extract transcripts ───────────────────────────────────────────────
    transcripts_zip = data_dir / FILES["transcripts"]["filename"]
    if transcripts_zip.exists():
        unzip_if_needed(transcripts_zip, data_dir / "transcripts")

    # ── 3. Extract recordings + pull audio ──────────────────────────────────
    recordings_zip = data_dir / FILES["recordings"]["filename"]
    if recordings_zip.exists() and not args.skip_recordings:
        recordings_dir = data_dir / "recordings"
        unzip_if_needed(recordings_zip, recordings_dir)
        extract_all_audio(recordings_dir, data_dir / "audio")

    # ── 4. Manifest ──────────────────────────────────────────────────────────
    write_manifest(data_dir)
    log("All done.")


if __name__ == "__main__":
    main()

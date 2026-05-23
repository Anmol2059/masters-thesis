#!/usr/bin/env python3
"""Prepare Fisher/Callhome Spanish dataset for benchmarking.

Download the LDC corpus manually and point --data-dir at the extracted folder.
"""
import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True, help="Raw Fisher/Callhome directory")
    p.add_argument("--output-dir", type=Path, default=Path("data/fisher_callhome_es"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = list(args.data_dir.rglob("*.sph")) + list(args.data_dir.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files")

    for src in audio_files:
        dst = args.output_dir / src.name
        shutil.copy2(src, dst)

    print(f"Copied to {args.output_dir}")


if __name__ == "__main__":
    main()

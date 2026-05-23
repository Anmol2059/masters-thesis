#!/usr/bin/env python3
"""Download all required models for the thesis pipeline."""
import os
import sys
from pathlib import Path

HF_HOME = Path(os.environ.get("HF_HOME", "models/hf_cache"))
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)

MODELS = {
    "faster-whisper-large-v3":   "Systran/faster-whisper-large-v3",
    "seamless-m4t-v2-large":     "facebook/seamless-m4t-v2-large",
    "nllb-200-3.3b":             "facebook/nllb-200-3.3B",
    "wmt22-comet-da":            "Unbabel/wmt22-comet-da",
    "wmt22-cometkiwi-da":        "Unbabel/wmt22-cometkiwi-da",
}

IGNORE = ["*.gguf", "flax_model*", "tf_model*", "rust_model*"]


def is_cached(repo_id: str) -> bool:
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir(HF_HOME)
        return any(r.repo_id == repo_id for r in info.repos)
    except Exception:
        return False


def download(name: str, repo_id: str) -> None:
    from huggingface_hub import snapshot_download
    print(f"  Downloading {name}  ({repo_id}) ...")
    snapshot_download(repo_id=repo_id, cache_dir=HF_HOME, ignore_patterns=IGNORE)
    print(f"  ✓ {name}")


def main() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("[download_models] huggingface_hub not installed — skipping model check.")
        sys.exit(0)

    print("[download_models] Checking required models ...")
    for name, repo_id in MODELS.items():
        if is_cached(repo_id):
            print(f"  ✓ {name} (cached)")
        else:
            download(name, repo_id)

    print("[download_models] All models ready.\n")


if __name__ == "__main__":
    main()

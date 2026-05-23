#!/usr/bin/env python3
"""Download all required models into the system HuggingFace cache.

Models are stored in $HF_HOME (default: ~/.cache/huggingface/hub).
All loading code (transformers, faster-whisper, comet) reads from the same
cache automatically — no extra config needed.

To use a custom location:
    export HF_HOME=/path/to/shared/cache
    python scripts/download_models.py
"""
import sys

MODELS = {
    "faster-whisper-large-v3": "Systran/faster-whisper-large-v3",
    "seamless-m4t-v2-large":   "facebook/seamless-m4t-v2-large",
    "nllb-200-3.3b":           "facebook/nllb-200-3.3B",
    "wmt22-comet-da":          "Unbabel/wmt22-comet-da",
    "wmt22-cometkiwi-da":      "Unbabel/wmt22-cometkiwi-da",
}

IGNORE = ["*.gguf", "flax_model*", "tf_model*", "rust_model*"]


def is_cached(repo_id: str) -> bool:
    try:
        from huggingface_hub import scan_cache_dir
        return any(r.repo_id == repo_id for r in scan_cache_dir().repos)
    except Exception:
        return False


def download(name: str, repo_id: str) -> None:
    from huggingface_hub import snapshot_download
    print(f"  Downloading {name}  ({repo_id}) ...")
    snapshot_download(repo_id=repo_id, ignore_patterns=IGNORE)
    print(f"  ✓ {name}")


def main() -> None:
    try:
        import huggingface_hub
        from huggingface_hub import constants
        print(f"[download_models] Cache: {constants.HF_HUB_CACHE}")
    except ImportError:
        print("[download_models] huggingface_hub not installed.")
        sys.exit(1)

    for name, repo_id in MODELS.items():
        if is_cached(repo_id):
            print(f"  ✓ {name} (already cached)")
        else:
            download(name, repo_id)

    print("[download_models] All models ready.")


if __name__ == "__main__":
    main()

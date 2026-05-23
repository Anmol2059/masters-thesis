"""Shared utilities: logging setup and output directory helpers."""
import json
import logging
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str | Path, experiment_name: str) -> tuple[logging.Logger, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{experiment_name}_{timestamp}.log"

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
        logger.addHandler(fh)

    return logger, timestamp


def make_output_dir(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    return out


def load_cached(transcripts_dir: Path, stem: str) -> dict | None:
    cache = transcripts_dir / f"{stem}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    return None


def save_cached(transcripts_dir: Path, stem: str, data: dict) -> None:
    cache = transcripts_dir / f"{stem}.json"
    cache.write_text(json.dumps(data, ensure_ascii=False))


def auto_output_dir(base_name: str, timestamp: str) -> str:
    return f"results/{base_name}_{timestamp}"

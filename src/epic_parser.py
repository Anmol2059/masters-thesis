"""Parse EPIC v2.0 transcript format into clean text."""
import re
from pathlib import Path


# EPIC markup patterns to strip
_MARKUP = re.compile(
    r"\(\.+\)"          # (.) (pause markers)
    r"|//+"             # // (false starts)
    r"|-\w+-"           # -ehm- -eeh- (hesitations)
    r"|~\w+\s*</correct/>"  # ~word </correct/> (self-corrections, keep correction)
    r"|<[^>]+>"         # any remaining XML-like tags
)
_SPACES = re.compile(r"\s{2,}")


def clean(text: str) -> str:
    text = _MARKUP.sub(" ", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


def parse_epic_transcript(path: str | Path) -> str:
    """Read an EPIC .txt transcript and return cleaned plain text."""
    raw = Path(path).read_text(encoding="utf-8")
    return clean(raw)


def parse_epic_dir(directory: str | Path, output_dir: str | Path) -> None:
    """Clean all transcripts in a directory, writing results to output_dir."""
    src = Path(directory)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for txt in sorted(src.glob("*.txt")):
        (dst / txt.name).write_text(parse_epic_transcript(txt), encoding="utf-8")

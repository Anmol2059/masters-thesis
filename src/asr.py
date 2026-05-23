"""Whisper wrapper with optional domain prompting."""
import json
from pathlib import Path

from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda",
                 compute_type: str = "int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._initial_prompt: str | None = None

    def load_domain_glossary(self, glossary_path: str) -> None:
        data = json.loads(Path(glossary_path).read_text())
        terms = list(data.keys()) if isinstance(data, dict) else data
        self._initial_prompt = ", ".join(terms)

    def transcribe(self, audio_path: str) -> str:
        kwargs = {"language": "es"}
        if self._initial_prompt:
            kwargs["initial_prompt"] = self._initial_prompt
        segments, _ = self.model.transcribe(audio_path, **kwargs)
        return " ".join(s.text.strip() for s in segments)

"""Whisper-based streaming ASR using faster-whisper."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np


@dataclass
class Segment:
    text: str
    start: float
    end: float
    language: str = ""
    avg_logprob: float = 0.0


@dataclass
class Transcriber:
    model_size: str = "large-v3"
    device: str = "auto"
    compute_type: str = "int8"
    language: Optional[str] = None
    initial_prompt: str = ""
    _model: object = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        device = self.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = WhisperModel(
            self.model_size, device=device, compute_type=self.compute_type
        )

    def transcribe_file(self, audio_path: str) -> list[Segment]:
        self._load()
        segments, _ = self._model.transcribe(
            audio_path,
            language=self.language,
            initial_prompt=self.initial_prompt or None,
            beam_size=5,
            vad_filter=True,
        )
        return [Segment(s.text.strip(), s.start, s.end, self.language or "", s.avg_logprob)
                for s in segments]

    def transcribe_chunk(self, audio: np.ndarray, sr: int = 16_000) -> list[Segment]:
        """Transcribe a raw audio chunk (float32, mono)."""
        self._load()
        segments, _ = self._model.transcribe(
            audio,
            language=self.language,
            initial_prompt=self.initial_prompt or None,
            beam_size=3,
            vad_filter=True,
        )
        return [Segment(s.text.strip(), s.start, s.end, self.language or "", s.avg_logprob)
                for s in segments]

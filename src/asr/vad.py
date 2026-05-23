"""Silero VAD wrapper for audio segmentation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpeechSegment:
    start: int  # sample index
    end: int    # sample index


class VAD:
    _model = None
    _utils = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", force_reload=False, onnx=False
        )
        self._model = model
        self._utils = utils

    def get_speech_segments(
        self, audio: np.ndarray, sr: int = 16_000, threshold: float = 0.5
    ) -> list[SpeechSegment]:
        self._load()
        import torch
        get_speech_ts = self._utils[0]
        tensor = torch.from_numpy(audio).float()
        segments = get_speech_ts(tensor, self._model, sampling_rate=sr, threshold=threshold)
        return [SpeechSegment(s["start"], s["end"]) for s in segments]

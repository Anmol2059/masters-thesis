"""Orchestrates the full real-time interpretation pipeline."""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..asr import Transcriber, DomainAdapter
from ..translation import Translator, NLLBTranslator
from ..evaluator import LLMJudge, EvaluationResult
from .audio_capture import DualStreamCapture
from .buffer import AudioBuffer

SEGMENT_INTERVAL = 3.0  # seconds between evaluation cycles


@dataclass
class Pipeline:
    source_asr: Transcriber
    interp_asr: Transcriber
    translator: Translator
    judge: LLMJudge
    source_device: int = 0
    interp_device: int = 1
    on_result: Optional[Callable[[EvaluationResult], None]] = None
    _capture: DualStreamCapture = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)

    def _eval_loop(self) -> None:
        while self._running:
            time.sleep(SEGMENT_INTERVAL)
            src_audio = self._capture.source_buffer.drain()
            itp_audio = self._capture.interp_buffer.drain()
            if src_audio is None or itp_audio is None:
                continue

            src_segs = self.source_asr.transcribe_chunk(src_audio)
            itp_segs = self.interp_asr.transcribe_chunk(itp_audio)
            if not src_segs or not itp_segs:
                continue

            source_text = " ".join(s.text for s in src_segs)
            interp_text = " ".join(s.text for s in itp_segs)
            reference = self.translator.translate(source_text)
            result = self.judge.evaluate(source_text, reference, interp_text)

            if self.on_result:
                self.on_result(result)

    def start(self) -> None:
        self._capture = DualStreamCapture(
            source_device=self.source_device,
            interp_device=self.interp_device,
        )
        self._capture.start()
        self._running = True
        self._thread = threading.Thread(target=self._eval_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._capture.stop()
        self._thread.join(timeout=5)

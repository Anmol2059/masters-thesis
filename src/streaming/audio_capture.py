"""Dual-stream audio capture (source speaker + interpreter)."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .buffer import AudioBuffer

SAMPLE_RATE = 16_000
CHUNK_SIZE = 1024


@dataclass
class DualStreamCapture:
    source_device: int = 0       # mic index for source speaker
    interp_device: int = 1       # mic index for interpreter
    on_source: Callable[[np.ndarray], None] | None = None
    on_interp: Callable[[np.ndarray], None] | None = None
    source_buffer: AudioBuffer = field(default_factory=AudioBuffer)
    interp_buffer: AudioBuffer = field(default_factory=AudioBuffer)
    _threads: list[threading.Thread] = field(default_factory=list, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def _capture_loop(self, device: int, buffer: AudioBuffer, cb: Callable | None) -> None:
        import sounddevice as sd
        with sd.InputStream(device=device, samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=CHUNK_SIZE) as stream:
            while not self._stop.is_set():
                data, _ = stream.read(CHUNK_SIZE)
                chunk = data[:, 0]
                buffer.push(chunk)
                if cb:
                    cb(chunk)

    def start(self) -> None:
        self._stop.clear()
        for device, buf, cb in [
            (self.source_device, self.source_buffer, self.on_source),
            (self.interp_device, self.interp_buffer, self.on_interp),
        ]:
            t = threading.Thread(target=self._capture_loop, args=(device, buf, cb), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2)
        self._threads.clear()

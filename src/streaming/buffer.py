"""Ring buffer for streaming audio chunks."""
from __future__ import annotations

import threading
from collections import deque

import numpy as np


class AudioBuffer:
    def __init__(self, maxlen: int = 50) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._buf.append(chunk)

    def drain(self) -> np.ndarray | None:
        with self._lock:
            if not self._buf:
                return None
            data = np.concatenate(list(self._buf))
            self._buf.clear()
            return data

    def __len__(self) -> int:
        return len(self._buf)

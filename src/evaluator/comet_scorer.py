"""COMET reference-based scoring wrapper."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class COMETScorer:
    model: str = "Unbabel/wmt22-comet-da"
    _model: object = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        from comet import load_from_checkpoint, download_model
        path = download_model(self.model)
        self._model = load_from_checkpoint(path)

    def score(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ) -> list[float]:
        self._load()
        data = [{"src": s, "mt": h, "ref": r}
                for s, h, r in zip(sources, hypotheses, references)]
        output = self._model.predict(data, batch_size=8, gpus=1)
        return output.scores

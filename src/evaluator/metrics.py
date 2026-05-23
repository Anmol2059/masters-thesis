"""Evaluation result dataclass and aggregation helpers."""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    accuracy: float = 0.0
    fluency: float = 0.0
    pragmatic: float = 0.0
    terminology: float = 0.0
    omissions: list[str] = field(default_factory=list)
    additions: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_json(cls, raw: str) -> "EvaluationResult":
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group() if match else raw)
        return cls(
            accuracy=float(data.get("accuracy", 0)),
            fluency=float(data.get("fluency", 0)),
            pragmatic=float(data.get("pragmatic", 0)),
            terminology=float(data.get("terminology", 0)),
            omissions=data.get("omissions", []),
            additions=data.get("additions", []),
            notes=data.get("notes", ""),
        )

    def overall(self, weights: dict | None = None) -> float:
        w = weights or {"accuracy": 0.4, "fluency": 0.2, "pragmatic": 0.2, "terminology": 0.2}
        return sum(getattr(self, k) * v for k, v in w.items())

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "fluency": self.fluency,
            "pragmatic": self.pragmatic,
            "terminology": self.terminology,
            "omissions": self.omissions,
            "additions": self.additions,
            "notes": self.notes,
            "overall": self.overall(),
        }


def aggregate(results: list[EvaluationResult]) -> dict:
    if not results:
        return {}
    dims = ["accuracy", "fluency", "pragmatic", "terminology"]
    return {
        d: {
            "mean": statistics.mean(getattr(r, d) for r in results),
            "stdev": statistics.stdev(getattr(r, d) for r in results) if len(results) > 1 else 0,
        }
        for d in dims
    } | {"overall": {"mean": statistics.mean(r.overall() for r in results)}}

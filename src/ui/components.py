"""Shared Gradio UI component builders."""
from __future__ import annotations


def score_bar(label: str, value: float) -> str:
    filled = int(value / 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"{label}: {bar} {value:.0f}%"


def format_result(result) -> str:
    lines = [
        score_bar("Accuracy  ", result.accuracy),
        score_bar("Fluency   ", result.fluency),
        score_bar("Pragmatic ", result.pragmatic),
        score_bar("Terminology", result.terminology),
    ]
    if result.omissions:
        lines.append("\n⚠ Omissions: " + " | ".join(result.omissions))
    if result.additions:
        lines.append("⚠ Additions: " + " | ".join(result.additions))
    if result.notes:
        lines.append(f"ℹ {result.notes}")
    return "\n".join(lines)

"""Gradio dashboard for real-time interpretation monitoring."""
from __future__ import annotations

import gradio as gr

from .components import format_result
from ..evaluator import EvaluationResult


_latest: EvaluationResult | None = None


def update_result(result: EvaluationResult) -> None:
    global _latest
    _latest = result


def _get_display() -> str:
    if _latest is None:
        return "Waiting for first segment..."
    return format_result(_latest)


def launch(pipeline=None, share: bool = False) -> None:
    with gr.Blocks(title="InterpretBench") as demo:
        gr.Markdown("# InterpretBench — Real-Time Interpreter Quality")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Source Transcript")
                src_box = gr.Textbox(label="Spanish (source)", lines=4, interactive=False)
            with gr.Column():
                gr.Markdown("### Interpreter Output")
                itp_box = gr.Textbox(label="English (interpreter)", lines=4, interactive=False)

        with gr.Row():
            scores_box = gr.Textbox(label="Quality Scores", lines=8, interactive=False)

        refresh_btn = gr.Button("Refresh")
        refresh_btn.click(fn=_get_display, outputs=scores_box)

        demo.load(fn=_get_display, outputs=scores_box, every=3)

    if pipeline:
        pipeline.on_result = update_result
        pipeline.start()

    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)

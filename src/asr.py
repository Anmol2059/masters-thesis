"""ASR wrappers: Whisper (cascaded pipeline) and SeamlessM4T v2 (end-to-end ST)."""
import json
from pathlib import Path

from faster_whisper import WhisperModel


class Transcriber:
    """Whisper large-v3: ES audio → ES text."""

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


class SeamlessTranscriber:
    """SeamlessM4T v2: ES audio → ES text (ASR) or EN text (end-to-end ST)."""

    MODEL_ID = "facebook/seamless-m4t-v2-large"

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2Model
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = SeamlessM4Tv2Model.from_pretrained(
            self.MODEL_ID, dtype=torch.float16
        ).to(device)

    def _load_audio(self, audio_path: str):
        import librosa
        # librosa has no CUDA dependency — safer than torchaudio on this system
        waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        return waveform, sample_rate

    def transcribe(self, audio_path: str) -> str:
        """ES audio → ES text (ASR only, for WER evaluation)."""
        waveform, sr = self._load_audio(audio_path)
        inputs = self.processor(
            audio=waveform, sampling_rate=sr, return_tensors="pt"
        ).to(self.device)
        output_tokens = self.model.generate(**inputs, tgt_lang="spa", generate_speech=False)
        return self.processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

    def translate(self, audio_path: str) -> str:
        """ES audio → EN text (end-to-end speech translation)."""
        waveform, sr = self._load_audio(audio_path)
        inputs = self.processor(
            audio=waveform, sampling_rate=sr, return_tensors="pt"
        ).to(self.device)
        output_tokens = self.model.generate(**inputs, tgt_lang="eng", generate_speech=False)
        return self.processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

"""Local speech-to-text with OpenAI Whisper (base.en by default).

The browser sends 16 kHz mono 16-bit WAV, so no ffmpeg is needed here.
Whisper only TRANSCRIBES. A matching transcript does not prove the
pronunciation was accurate (stated in the proposal).
"""
from __future__ import annotations

import io
import threading
import wave

import numpy as np

from . import config

MIN_SECONDS = 0.25
SILENCE_RMS = 0.004      # below this the clip is treated as silence


class AudioError(ValueError):
    pass


def wav_to_float32(data: bytes) -> np.ndarray:
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            if wf.getsampwidth() != 2:
                raise AudioError("Expected 16-bit WAV audio.")
            rate, ch = wf.getframerate(), wf.getnchannels()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    except (wave.Error, EOFError) as e:
        raise AudioError(f"Could not read the recording: {e}") from e
    audio = pcm.astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if rate != 16000:   # simple linear resample fallback
        n = int(len(audio) * 16000 / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio


class Transcriber:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        if self._model is None:
            import whisper    # openai-whisper; downloads weights on first use
            self._model = whisper.load_model(config.WHISPER_MODEL)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def transcribe(self, wav_bytes: bytes) -> dict:
        audio = wav_to_float32(wav_bytes)
        seconds = len(audio) / 16000
        if seconds < MIN_SECONDS:
            raise AudioError("The recording was too short. Hold Record while you say the word.")
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < SILENCE_RMS:
            return {"text": "", "silent": True, "seconds": round(seconds, 2), "rms": rms}
        self.load()
        with self._lock:
            result = self._model.transcribe(
                audio, language="en", fp16=False,
                condition_on_previous_text=False, temperature=0.0)
        return {"text": result["text"].strip(), "silent": False,
                "seconds": round(seconds, 2), "rms": rms}

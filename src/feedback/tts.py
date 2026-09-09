from __future__ import annotations

import base64
import hashlib
import io
import shutil
import subprocess
import sys
import wave
from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16_000
DEFAULT_VOICE = "Samantha" 
_ESPEAK_VOICE = "en-us"     
_SPEAKING_RATE = 150        
_HAS_SAY = sys.platform == "darwin" and shutil.which("say") is not None


def _render_aiff(text: str, voice: str, rate: int) -> bytes:
    """Render text to AIFF bytes via the macOS `say` command."""
    proc = subprocess.run(
        ["say", "-v", voice, "-r", str(rate), "-o", "/dev/stdout",
         "--data-format=LEF32@16000", text],
        capture_output=True, timeout=15,
    )
    if proc.returncode == 0 and proc.stdout:
        return proc.stdout
    # Fallback: render to a temp file (some macOS versions dislike /dev/stdout)
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", tmp_path, text],
                       capture_output=True, timeout=15)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _render_espeak(text: str, rate: int) -> bytes:
    proc = subprocess.run(
        ["espeak-ng", "-v", _ESPEAK_VOICE, "-s", str(rate), "--stdout", text],
        capture_output=True, timeout=15,
    )
    return proc.stdout if proc.returncode == 0 else b""


def _render_speech(text: str, voice: str, rate: int) -> bytes:
    if _HAS_SAY:
        return _render_aiff(text, voice, rate)
    return _render_espeak(text, rate)


def _aiff_to_wave(raw: bytes) -> np.ndarray:
    """Decode arbitrary `say` output to float32 mono 16 kHz via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", "pipe:0", "-f", "f32le", "-ar", str(SAMPLE_RATE),
         "-ac", "1", "pipe:1"],
        input=raw, capture_output=True, timeout=15,
    )
    if proc.returncode == 0 and len(proc.stdout) >= 4:
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return np.zeros(SAMPLE_RATE // 2, dtype=np.float32)


@lru_cache(maxsize=256)
def synthesize(text: str, voice: str = DEFAULT_VOICE, rate: int = _SPEAKING_RATE) -> np.ndarray:
    """Return the reference pronunciation of `text` as a float32 16 kHz waveform."""
    text = (text or "").strip()
    if not text:
        return np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    raw = _render_speech(text, voice, rate)
    if not raw:
        return np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    return _aiff_to_wave(raw)


def pcm_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode a float32 [-1, 1] mono waveform as WAV (PCM16) bytes."""
    pcm16 = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def wav_data_uri(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Encode a waveform as a base64 ``data:audio/wav`` URI for inline playback."""
    if audio is None or len(audio) == 0:
        return ""
    b64 = base64.b64encode(pcm_to_wav_bytes(audio, sample_rate)).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def synthesize_wav_bytes(text: str, voice: str = DEFAULT_VOICE, rate: int = _SPEAKING_RATE) -> bytes:
    """Return the reference pronunciation of `text` as WAV (PCM16) bytes for playback."""
    return pcm_to_wav_bytes(synthesize(text, voice, rate))


def cache_key(text: str, voice: str = DEFAULT_VOICE) -> str:
    """Stable short key for a (text, voice) pair -- handy for HTTP caching/etags."""
    return hashlib.sha1(f"{voice}:{text}".encode("utf-8")).hexdigest()[:16]

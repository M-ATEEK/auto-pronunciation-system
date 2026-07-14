"""Audio decoding: raw upload bytes = float32 mono waveform at 16 kHz.
"""

from __future__ import annotations

import io
import subprocess

import numpy as np

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

SAMPLE_RATE = 16_000


def decode_audio(raw: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode arbitrary audio bytes to a float32 mono waveform at ``sr``."""
    if _HAS_SF:
        try:
            audio, orig_sr = sf.read(io.BytesIO(raw), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if orig_sr != sr:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(int(orig_sr), sr)
                audio = resample_poly(audio, sr // g, int(orig_sr) // g).astype(np.float32)
            return audio.astype(np.float32)
        except Exception:
            pass

    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", "pipe:0", "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1"],
            input=raw, capture_output=True, timeout=15,
        )
        if proc.returncode == 0 and len(proc.stdout) >= 4:
            return np.frombuffer(proc.stdout, dtype=np.float32).copy()
    except Exception:
        pass

    return np.zeros(sr // 2, dtype=np.float32)

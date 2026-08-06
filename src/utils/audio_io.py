"""Audio decoding and silence trimming: raw upload bytes -> float32 mono 16 kHz.
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

TRIM_TOP_DB = 25
SILENCE_RMS_FLOOR = 0.05
_TRIM_FRAME = 400   # 25 ms at 16 kHz
_TRIM_HOP = 160     # 10 ms at 16 kHz
_TRIM_PAD_S = 0.02  # keep 20 ms either side so onsets are not clipped
_ABS_THRESHOLD_FLOOR = 0.01  # backstop for near-silent frames
# Speech lasts longer than a click or a gain-control artefact. Boundaries must
# be backed by at least this many consecutive active frames (50 ms).
MIN_SPEECH_RUN_FRAMES = 5
ANCHOR_PEAK_FRACTION = 0.15


def frame_rms(audio: np.ndarray) -> np.ndarray:
    """Per-frame RMS energy (25 ms window, 10 ms hop)."""
    n = max(0, len(audio) - _TRIM_FRAME)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.array(
        [np.sqrt(np.mean(audio[i:i + _TRIM_FRAME] ** 2) + 1e-12)
         for i in range(0, n, _TRIM_HOP)],
        dtype=np.float32,
    )


def has_speech(audio: np.ndarray) -> bool:
    """True if any frame is loud enough to plausibly be speech."""
    rms = frame_rms(audio)
    return bool(len(rms) and rms.max() >= SILENCE_RMS_FLOOR)


def trim_silence(audio: np.ndarray, top_db: int = TRIM_TOP_DB,
                 min_run_frames: int = MIN_SPEECH_RUN_FRAMES) -> np.ndarray:
    """Remove leading/trailing silence from a waveform.

    """
    if audio.size == 0:
        return audio

    rms = frame_rms(audio)
    if len(rms) == 0 or rms.max() < SILENCE_RMS_FLOOR:
        return audio[:0]

    threshold = max(rms.max() * (10.0 ** (-top_db / 20.0)), _ABS_THRESHOLD_FLOOR)
    active = rms >= threshold

    # Keep only sustained runs; isolated blips are artefacts, not speech.
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        j = i
        while j < len(active) and active[j]:
            j += 1
        if j - i >= min_run_frames:
            runs.append((i, j))
        i = j
    if not runs:
        return audio[:0]

    peak = rms.max()
    anchors = [(a, b) for a, b in runs if rms[a:b].max() >= ANCHOR_PEAK_FRACTION * peak]
    if anchors:
        runs = anchors

    pad = int(_TRIM_PAD_S * SAMPLE_RATE)
    lo = max(0, runs[0][0] * _TRIM_HOP - pad)
    hi = min(len(audio), runs[-1][1] * _TRIM_HOP + _TRIM_FRAME + pad)
    return audio[lo:hi].astype(np.float32)


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

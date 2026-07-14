"""17-dimensional acoustic feature extraction (Stage 2).
"""

from __future__ import annotations

import numpy as np
import librosa

SAMPLE_RATE = 16_000
N_MFCC = 13
HOP_LENGTH = 160    # 10 ms at 16 kHz
WIN_LENGTH = 400    # 25 ms at 16 kHz
FMIN = 60.0
FMAX = 400.0
PAUSE_ENERGY_THRESHOLD = 0.01
FEATURE_DIM = 17


def extract_features(audio: np.ndarray, sr: int = SAMPLE_RATE, n_phones: int = 1) -> np.ndarray:
    """Return a 17-dimensional feature vector for one audio segment."""
    if len(audio) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    audio = audio.astype(np.float32)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    mfcc_mean = mfcc.mean(axis=1)

    try:
        f0, voiced_flag, _ = librosa.pyin(audio, fmin=FMIN, fmax=FMAX, sr=sr, hop_length=HOP_LENGTH)
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None and voiced_flag.any() else np.array([])
        f0_mean = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        if np.isnan(f0_mean):
            f0_mean = 0.0
    except Exception:
        f0_mean = 0.0

    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
    rms_mean = float(rms.mean())

    duration = len(audio) / sr
    speech_rate = float(n_phones / max(duration, 0.01))
    pause_ratio = float((rms < PAUSE_ENERGY_THRESHOLD).mean())

    vec = np.concatenate([mfcc_mean, [f0_mean, rms_mean, speech_rate, pause_ratio]])
    return vec.astype(np.float32)


def extract_mfcc_sequence(audio: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Return the full MFCC frame matrix — shape (T, n_mfcc) — for DTW comparison."""
    if len(audio) == 0:
        return np.zeros((1, n_mfcc), dtype=np.float32)
    audio = audio.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    return mfcc.T.astype(np.float32)

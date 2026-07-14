"""Unit tests for the Stage-2 feature extractor."""

import numpy as np

from src.features.extractor import (
    FEATURE_DIM,
    N_MFCC,
    SAMPLE_RATE,
    extract_features,
    extract_mfcc_sequence,
)


def _sine(freq: float = 150.0, seconds: float = 0.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_feature_vector_shape():
    vec = extract_features(_sine())
    assert vec.shape == (FEATURE_DIM,)
    assert vec.dtype == np.float32


def test_empty_audio_returns_zeros():
    vec = extract_features(np.array([], dtype=np.float32))
    assert vec.shape == (FEATURE_DIM,)
    assert np.allclose(vec, 0.0)


def test_tone_has_energy_silence_does_not():
    tone_rms = extract_features(_sine(freq=150.0))[14]
    silence_rms = extract_features(np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32))[14]
    assert tone_rms > 0.0            # a tone has energy
    assert silence_rms < 0.01        # silence has ~zero energy
    assert tone_rms > silence_rms


def test_f0_is_non_negative():
    f0 = extract_features(_sine(freq=150.0))[13]
    assert f0 >= 0.0                 # F0 is a valid, non-negative value


def test_mfcc_sequence_shape():
    seq = extract_mfcc_sequence(_sine(seconds=0.4))
    assert seq.ndim == 2
    assert seq.shape[1] == N_MFCC
    assert seq.shape[0] > 1          # multiple frames for 0.4 s of audio

"""Utterance-level content verification: did the learner say THIS sentence
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from src.feedback.tts import synthesize
from src.features.extractor import SAMPLE_RATE, extract_mfcc_sequence
from src.personalization.dtw import dtw_distance
from src.utils.audio_io import trim_silence
from src.utils.logger import get_logger

_LOG = get_logger(__name__)

MISMATCH_RATIO_THRESHOLD = 0.94


MIN_WORDS = 3
_MIN_FRAMES = 12
_TARGET_FRAMES = 180


def _utterance_mfcc(audio: np.ndarray) -> np.ndarray:
    """Cepstral-mean-normalised MFCC envelope for a whole utterance.
    """
    audio = trim_silence(audio)
    if audio.size == 0:
        return np.zeros((0, 1), dtype=np.float32)
    seq = extract_mfcc_sequence(audio, sr=SAMPLE_RATE)
    if len(seq) == 0:
        return seq
    step = max(1, len(seq) // _TARGET_FRAMES)
    return (seq - seq.mean(axis=0, keepdims=True))[::step]


def _shuffle_words(text: str, seed: int = 0) -> str:
    """Reorder the words, guaranteeing a different order where possible."""
    words = text.split()
    rng = random.Random(seed)
    for _ in range(8):
        candidate = words[:]
        rng.shuffle(candidate)
        if candidate != words:
            return " ".join(candidate)
    return " ".join(reversed(words))


def _make_decoy(text: str) -> Optional[str]:
    """A same-length, different-content competing hypothesis, or None."""
    if len(text.split()) < MIN_WORDS:
        return None
    return _shuffle_words(text)


def _reference_envelope(text: str) -> np.ndarray:
    """TTS the text and return its MFCC envelope (TTS itself is cached).

    ``_utterance_mfcc`` trims, so both sides of the comparison get identical
    treatment.
    """
    return _utterance_mfcc(synthesize(text))


def verify_content(audio: np.ndarray, transcript: str) -> tuple[Optional[float], bool]:
    decoy_text = _make_decoy(transcript)
    if decoy_text is None:
        return None, False

    learner = _utterance_mfcc(audio)
    if len(learner) < _MIN_FRAMES:
        return None, False

    try:
        # Both TTS calls are independent subprocesses; run them concurrently so
        # a first-time sentence costs one synthesis, not two.
        with ThreadPoolExecutor(max_workers=2) as pool:
            ordered_f = pool.submit(_reference_envelope, transcript)
            decoy_f = pool.submit(_reference_envelope, decoy_text)
            ordered, decoy = ordered_f.result(), decoy_f.result()
    except Exception:
        _LOG.exception("content verification: TTS failed for %r", transcript[:60])
        return None, False

    if len(ordered) < _MIN_FRAMES or len(decoy) < _MIN_FRAMES:
        return None, False

    d_ordered = dtw_distance(learner, ordered)
    d_decoy = dtw_distance(learner, decoy)
    if d_decoy <= 1e-6:
        return None, False

    ratio = float(d_ordered / d_decoy)
    return round(ratio, 4), ratio >= MISMATCH_RATIO_THRESHOLD

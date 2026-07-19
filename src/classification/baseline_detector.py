"""Random-Forest baseline detector (Stage 3).

Wraps a per-phone ``PhonemeRandomForest`` trained on REAL LibriSpeech features
(see ``src/data/librispeech.py``).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.classification.random_forest import PhonemeRandomForest
from src.data.aligner import DictionaryAligner
from src.features.extractor import N_MFCC, extract_features, extract_mfcc_sequence

SAMPLE_RATE = 16_000


@dataclass
class BaselinePhoneResult:
    phone: str
    word: str
    prob_mispronounced: float
    is_mispronounced: bool
    learner_mfcc: Optional[np.ndarray] = None  # segment MFCC sequence (for future visuals)
    audio_clip: Optional[np.ndarray] = None    # padded segment audio (for future playback)


class BaselineDetector:
    """Native-only-trained RF detector using hand-crafted acoustic features."""

    def __init__(self, rf: PhonemeRandomForest, cmn: bool = True,
                 threshold: float = 0.5) -> None:
        self._rf = rf
        self._cmn = cmn
        self._threshold = threshold
        self._aligner = DictionaryAligner()

    # -- persistence ----------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"rf": self._rf, "cmn": self._cmn,
                         "threshold": self._threshold}, fh)

    @classmethod
    def load(cls, path: str) -> "BaselineDetector":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        return cls(blob["rf"], cmn=blob.get("cmn", True),
                   threshold=blob.get("threshold", 0.5))

    @property
    def trained_phones(self) -> list[str]:
        return self._rf.trained_phones

    # -- inference --------------------------------------------------------------
    def detect(self, audio: np.ndarray, transcript: str,
               sr: int = SAMPLE_RATE) -> list[BaselinePhoneResult]:
        """Return per-phone match/mismatch decisions for one utterance."""
        audio = audio.astype(np.float32)
        segments = self._aligner.align(audio, sr, transcript)
        if not segments:
            return []

        # Utterance-level CMN, matching src/data/librispeech.py's training-time
        # normalisation exactly.
        utt_mean = np.zeros(N_MFCC, dtype=np.float32)
        if self._cmn:
            seq = extract_mfcc_sequence(audio, sr=sr)
            if len(seq) > 0:
                utt_mean = seq.mean(axis=0).astype(np.float32)

        results: list[BaselinePhoneResult] = []
        pad = int(0.08 * sr)
        for seg in segments:
            if len(seg.audio) < 160:
                # Too little signal to feature-extract — treat as omission.
                results.append(BaselinePhoneResult(seg.phone, seg.word, 0.9, True))
                continue
            feat = extract_features(seg.audio, sr=sr, n_phones=1).copy()
            if self._cmn:
                feat[:N_MFCC] = feat[:N_MFCC] - utt_mean
            prob = self._rf.predict_proba_mispronounced(seg.phone, feat)
            lo = max(0, int(seg.start * sr) - pad)
            hi = min(len(audio), int(seg.end * sr) + pad)
            results.append(BaselinePhoneResult(
                phone=seg.phone,
                word=seg.word,
                prob_mispronounced=round(float(prob), 4),
                is_mispronounced=prob > self._threshold,
                learner_mfcc=extract_mfcc_sequence(seg.audio, sr=sr),
                audio_clip=audio[lo:hi],
            ))
        return results

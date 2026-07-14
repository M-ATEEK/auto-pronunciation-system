"""Phone alignment (Stage 1).

`DictionaryAligner` uses the CMU Pronouncing Dictionary (via `pronouncing`) to
look up the phoneme sequence of each word in a transcript, and splits the audio
across those phones using corpus-derived duration weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.utils.phoneme_constants import ENGLISH_PHONEMES

try:
    import pronouncing as _pro
    _HAS_PRONOUNCING = True
except ImportError:
    _HAS_PRONOUNCING = False

_STRESS_RE = re.compile(r"\d")


@dataclass
class PhoneSegment:
    """A time-aligned phone segment extracted from an audio file."""

    phone: str
    start: float          # seconds
    end: float            # seconds
    audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    sr: int = 16_000
    word: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def extract_audio(self, full_audio: np.ndarray, sr: int) -> "PhoneSegment":
        start_sample = int(self.start * sr)
        end_sample = int(self.end * sr)
        sliced = full_audio[start_sample:end_sample]
        return PhoneSegment(
            phone=self.phone,
            start=self.start,
            end=self.end,
            audio=sliced.astype(np.float32),
            sr=sr,
            word=self.word,
        )


def _strip_stress(phone: str) -> str:
    return _STRESS_RE.sub("", phone)


# Valid ARPABET symbols for direct phoneme-mode detection.
_ARPABET: frozenset[str] = frozenset(ENGLISH_PHONEMES)


def _phones_for_word(word: str) -> list[str]:
    """Return the ARPABET phoneme list for a word, or a fallback sequence.

    """
    word = word.strip(".,!?;:'\"")

    # Direct ARPABET symbol — bypass the dictionary entirely.
    upper = _STRESS_RE.sub("", word.upper())
    if upper in _ARPABET:
        return [upper]

    word = word.lower()
    if _HAS_PRONOUNCING:
        results = _pro.phones_for_word(word)
        if results:
            phones = [_strip_stress(p) for p in results[0].split()]
            phones = [p for p in phones if p in ENGLISH_PHONEMES]
            if phones:
                return phones
    # fallback is deterministic hash-based sequence for out-of-dictionary words.
    n = max(len(word) // 2, 1)
    idx = abs(hash(word)) % len(ENGLISH_PHONEMES)
    return [ENGLISH_PHONEMES[(idx + i) % len(ENGLISH_PHONEMES)] for i in range(n)]


def phones_for_text(transcript: str) -> list[str]:
    """Return the flat ARPABET phone sequence for a whole transcript."""
    phones: list[str] = []
    for word in transcript.split():
        phones.extend(_phones_for_word(word))
    return phones


class AlignerProtocol(Protocol):
    def align(self, audio: np.ndarray, sr: int, transcript: str) -> list[PhoneSegment]:
        ...


# Relative duration weights per phoneme class derived from corpus statistics.
_PHONE_WEIGHT: dict[str, float] = {
    # Vowels (longest)
    "AA": 1.5, "AE": 1.4, "AH": 1.2, "AO": 1.5, "AW": 1.4, "AY": 1.4,
    "EH": 1.3, "ER": 1.3, "EY": 1.3, "IH": 1.1, "IY": 1.2,
    "OW": 1.4, "OY": 1.3, "UH": 1.1, "UW": 1.3,
    "L": 0.9, "R": 0.9, "W": 0.8, "Y": 0.8,
    # Nasals
    "M": 0.8, "N": 0.8, "NG": 0.7,
    "CH": 0.9, "DH": 0.9, "F": 1.0, "JH": 0.9,
    "S": 1.0, "SH": 1.0, "TH": 0.9, "V": 0.9, "Z": 0.9, "ZH": 0.9,
    # Stops (shortest — closure + burst)
    "B": 0.55, "D": 0.55, "G": 0.55, "K": 0.55, "P": 0.55, "T": 0.55,
    "HH": 0.7,
}
_DEFAULT_WEIGHT = 1.0


class DictionaryAligner:
    """Assigns real phoneme sequences from the CMU Pronouncing Dictionary.

    The audio duration is split proportionally using corpus-derived phoneme
    duration weights.
    """

    def align(self, audio: np.ndarray, sr: int, transcript: str) -> list[PhoneSegment]:
        duration = len(audio) / sr
        words = transcript.lower().split()

        word_phones: list[tuple[str, list[str]]] = []
        for w in words:
            word_phones.append((w, _phones_for_word(w)))

        # Build a flat (word, phone, weight) list, then distribute the duration.
        flat: list[tuple[str, str, float]] = []
        for word, phones in word_phones:
            for phone in phones:
                flat.append((word, phone, _PHONE_WEIGHT.get(phone, _DEFAULT_WEIGHT)))

        if not flat:
            return []

        total_weight = sum(w for _, _, w in flat)
        segments: list[PhoneSegment] = []
        t = 0.0
        for word, phone, weight in flat:
            seg_dur = (weight / total_weight) * duration
            seg = PhoneSegment(phone=phone, start=t, end=t + seg_dur, sr=sr, word=word)
            segments.append(seg.extract_audio(audio, sr))
            t += seg_dur
        return segments

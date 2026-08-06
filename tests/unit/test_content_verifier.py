"""Unit tests for utterance-level content verification.

"""

from __future__ import annotations

import numpy as np

from src.classification.content_verifier import (
    MIN_WORDS,
    MISMATCH_RATIO_THRESHOLD,
    _make_decoy,
    _shuffle_words,
    verify_content,
)

LONG = "the quick brown fox jumps over the lazy dog every single morning"


class TestShuffleWords:
    def test_returns_a_different_order(self):
        assert _shuffle_words(LONG) != LONG

    def test_preserves_the_same_words(self):
        assert sorted(_shuffle_words(LONG).split()) == sorted(LONG.split())

    def test_is_deterministic_for_a_given_seed(self):
        assert _shuffle_words(LONG, seed=7) == _shuffle_words(LONG, seed=7)

    def test_single_word_cannot_be_reordered(self):
        # Must not hang or raise; a one-word "shuffle" is just the word back.
        assert _shuffle_words("hello") == "hello"


class TestVerifyContentGuards:
    """Verification must decline rather than guess when it cannot apply."""

    def test_too_few_words_is_inconclusive(self):
        """Single words cannot be verified: a decoy of 'cap' is 'pac', which
        contains the same phones and aligns under DTW's time warping."""
        audio = np.random.default_rng(0).normal(0, 0.2, 16000).astype(np.float32)
        ratio, mismatch = verify_content(audio, "cap")
        assert ratio is None
        assert mismatch is False

    def test_decoy_declined_below_min_words(self):
        assert _make_decoy("cap") is None
        assert _make_decoy("hello there") is None
        assert _make_decoy("what is this") is not None

    def test_min_words_boundary_is_declined(self):
        audio = np.random.default_rng(0).normal(0, 0.2, 16000).astype(np.float32)
        short = " ".join(["word"] * (MIN_WORDS - 1))
        assert verify_content(audio, short) == (None, False)

    def test_empty_audio_is_inconclusive(self):
        ratio, mismatch = verify_content(np.zeros(0, dtype=np.float32), LONG)
        assert ratio is None
        assert mismatch is False

    def test_very_short_audio_is_inconclusive(self):
        # Fewer frames than DTW needs to say anything meaningful.
        ratio, mismatch = verify_content(np.zeros(800, dtype=np.float32), LONG)
        assert ratio is None
        assert mismatch is False

    def test_inconclusive_never_reports_mismatch(self):
        """A failed check must not produce a false wrong-sentence warning."""
        for audio, text in [
            (np.zeros(0, dtype=np.float32), LONG),
            (np.zeros(800, dtype=np.float32), LONG),
            (np.random.default_rng(1).normal(0, 0.2, 16000).astype(np.float32), "two words"),
        ]:
            ratio, mismatch = verify_content(audio, text)
            assert mismatch is False, f"warned on inconclusive input: {text!r}"


class TestThreshold:
    def test_threshold_sits_between_measured_populations(self):
        """Measured worst cases: matching <= 0.932, non-matching >= 0.945."""
        assert 0.932 < MISMATCH_RATIO_THRESHOLD < 0.945

"""Unit tests for silence trimming and speech detection.

"""

from __future__ import annotations

import numpy as np

from src.utils.audio_io import SAMPLE_RATE, has_speech, trim_silence


def _tone(duration_s: float, amp: float = 0.5, freq: float = 200.0) -> np.ndarray:
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_s: float, amp: float = 0.0) -> np.ndarray:
    n = int(duration_s * SAMPLE_RATE)
    if amp == 0.0:
        return np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    return (rng.normal(0, amp, n)).astype(np.float32)


class TestHasSpeech:
    def test_detects_a_loud_tone(self):
        assert has_speech(_tone(0.5)) is True

    def test_rejects_digital_silence(self):
        assert has_speech(_silence(1.0)) is False

    def test_rejects_low_level_room_noise(self):
        # A "silent" browser recording is not digital silence -- it is faint
        # room noise. A purely relative trim would treat this as signal.
        assert has_speech(_silence(1.0, amp=0.002)) is False

    def test_rejects_browser_amplified_room_noise(self):
        # The browser's gain control amplifies room noise (measured ~0.027 RMS
        # peak). That must still count as silence, not speech.
        assert has_speech(_silence(1.0, amp=0.009)) is False

    def test_accepts_quiet_but_real_speech(self):
        # Quietest LibriSpeech utterance measured ~0.096 peak-frame RMS.
        assert has_speech(_tone(0.5, amp=0.14)) is True

    def test_empty_audio_has_no_speech(self):
        assert has_speech(np.zeros(0, dtype=np.float32)) is False


class TestTrimSilence:
    def test_removes_leading_silence(self):
        audio = np.concatenate([_silence(0.5), _tone(0.5)])
        trimmed = trim_silence(audio)
        # The speech is ~0.5 s; trimming should get close to that, not 1.0 s.
        assert len(trimmed) < len(audio) * 0.75
        assert len(trimmed) > 0.3 * SAMPLE_RATE

    def test_removes_trailing_silence(self):
        audio = np.concatenate([_tone(0.5), _silence(0.5)])
        trimmed = trim_silence(audio)
        assert len(trimmed) < len(audio) * 0.75

    def test_removes_silence_on_both_sides(self):
        audio = np.concatenate([_silence(0.4), _tone(0.4), _silence(0.4)])
        trimmed = trim_silence(audio)
        assert len(trimmed) < len(audio) * 0.7

    def test_keeps_speech_that_has_no_padding(self):
        audio = _tone(0.6)
        trimmed = trim_silence(audio)
        assert len(trimmed) >= 0.9 * len(audio)

    def test_all_silence_returns_empty(self):
        assert trim_silence(_silence(1.0)).size == 0

    def test_low_level_noise_returns_empty(self):
        assert trim_silence(_silence(1.0, amp=0.002)).size == 0

    def test_empty_input_returns_empty(self):
        assert trim_silence(np.zeros(0, dtype=np.float32)).size == 0

    def test_faint_blip_before_the_word_does_not_anchor_the_boundary(self):
        """Regression: a breath/click before a short word ate the first phones.

        A 100 ms faint blip passes the run-length rule, so without an energy
        anchor the trim started at the blip and kept ~0.5 s of silence before
        the actual word -- on a 3-phone word that consumed the first phone.
        """
        blip = _tone(0.10, amp=0.04)          # faint, but long enough to be a "run"
        audio = np.concatenate([
            _silence(0.5), blip, _silence(0.5), _tone(0.4, amp=0.45),
        ])
        trimmed = trim_silence(audio)
        # Must start at the real word, not the blip: the blip + gap is 1.0 s,
        # so anything close to the full length means the blip won.
        assert len(trimmed) < 0.7 * SAMPLE_RATE
        # And the first quarter must be real signal, not silence.
        q = len(trimmed) // 4
        assert np.abs(trimmed[:q]).max() > 0.1

    def test_quiet_sound_inside_an_utterance_is_kept(self):
        """Only the outer boundaries are anchored; interior content stays."""
        audio = np.concatenate([
            _tone(0.3, amp=0.45), _tone(0.1, amp=0.04), _tone(0.3, amp=0.45),
        ])
        trimmed = trim_silence(audio)
        # All three parts retained (~0.7 s), not just the loud ends.
        assert len(trimmed) >= 0.65 * SAMPLE_RATE

    def test_leading_silence_no_longer_dominates_a_short_clip(self):
        """The actual bug: on a short utterance, leading silence took a whole phone.

        A 1.5 s clip whose speech starts at 0.5 s means the first quarter of
        the clip (one phone of a 4-phone word) is pure silence. After trimming,
        the first quarter must contain real signal.
        """
        audio = np.concatenate([_silence(0.5), _tone(1.0)])
        quarter = len(audio) // 4
        assert np.abs(audio[:quarter]).max() < 0.01      # before: silent

        trimmed = trim_silence(audio)
        tq = len(trimmed) // 4
        assert np.abs(trimmed[:tq]).max() > 0.1          # after: real signal

"""CAPT system orchestrator (Stage 3 dual-detector dispatch).
"""

from __future__ import annotations

import os

import numpy as np

from src.classification.baseline_detector import BaselineDetector
from src.classification.phoneme_align import align, is_confusable
from src.classification.phoneme_recognizer import PhonemeRecognizer
from src.data.aligner import _phones_for_word
from src.data.librispeech import build_native_exemplars
from src.feedback.generator import generate_feedback
from src.features.extractor import SAMPLE_RATE, extract_mfcc_sequence
from src.personalization.discovery import NativeReferenceBank, discover_error_patterns, rules_to_dict
from src.personalization.dtw import dtw_distance
from src.personalization.profiler import LearnerProfile, ProfileManager
from src.utils.logger import get_logger

_LOG = get_logger(__name__)

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
_CORPUS_DIR = os.path.join(_BASE_DIR, "data", "librispeech")
_BASELINE_MODEL_PATH = os.path.join(_BASE_DIR, "models", "baseline_rf.pkl")
_EXEMPLAR_BANK_PATH = os.path.join(_BASE_DIR, "models", "native_exemplars.pkl")
_PROFILE_DIR = os.path.join(_BASE_DIR, "models", "profiles")

DETECTOR_BASELINE = "baseline"
DETECTOR_NEURAL = "neural"

# Once a discovered rule escalates a phone, this is the confidence floor applied.
_SYSTEMATIC_SCORE_FLOOR = 0.82

# Utterance-level plausibility guard no natural speaking rate exceeds this
_MAX_PHONES_PER_SEC = 16.0
_CLIP_PAD_S = 0.08  # padding around a phone segment so learner playback is audible


def _articulation_implausible(waveform: np.ndarray, transcript: str,
                               sr: int = SAMPLE_RATE) -> tuple[bool, float]:
    """Return (is_implausible, phones_per_second).

    Flags the case where the transcript has far more phones than the active
    (non silent) audio duration could hold at any natural speaking rate.
    """
    n_phones = 0
    for w in transcript.split():
        n_phones += len(_phones_for_word(w.strip(".,!?;:'\"")))
    if n_phones == 0 or waveform.size == 0:
        return False, 0.0
    win, hop = int(0.025 * sr), int(0.010 * sr)
    active_frames = 0
    for i in range(0, max(0, len(waveform) - win), hop):
        if np.sqrt(np.mean(waveform[i:i + win] ** 2) + 1e-9) > 0.01:
            active_frames += 1
    active_dur = active_frames * hop / sr
    rate = n_phones / max(active_dur, 0.05)
    return rate > _MAX_PHONES_PER_SEC, round(rate, 1)


class CAPTSystem:
    """Dispatches per-phone analysis to the classical or neural detector."""

    def __init__(self) -> None:
        self._profile_manager = ProfileManager(_PROFILE_DIR)
        self._baseline: BaselineDetector | None = None
        self._exemplars: dict | None = None
        self._ref_bank: NativeReferenceBank | None = None

    # -- lazy-loaded shared resources (each loaded once per process)
    def _get_baseline(self) -> BaselineDetector:
        if self._baseline is None:
            if not os.path.exists(_BASELINE_MODEL_PATH):
                raise RuntimeError(
                    "Baseline model not trained yet — run scripts/train_baseline.py first.")
            self._baseline = BaselineDetector.load(_BASELINE_MODEL_PATH)
        return self._baseline

    def _get_exemplars(self) -> dict:
        if self._exemplars is None:
            self._exemplars = build_native_exemplars(_CORPUS_DIR, per_phone=3, max_utterances=80)
        return self._exemplars

    def _native_reference_mfcc(self, phone: str):
        exemplars = self._get_exemplars().get(phone)
        return exemplars[0] if exemplars else None

    def _get_ref_bank(self) -> NativeReferenceBank:
        if self._ref_bank is None:
            if not os.path.exists(_EXEMPLAR_BANK_PATH):
                raise RuntimeError(
                    "Native exemplar bank not built yet — run scripts/build_exemplars.py first.")
            self._ref_bank = NativeReferenceBank.load(_EXEMPLAR_BANK_PATH)
        return self._ref_bank

    # -- public API
    def analyze(self, waveform, transcript: str, learner_id: str,
                detector: str = DETECTOR_BASELINE) -> dict:
        """Run one detector's full per-phone analysis and update the learner profile.

        Returns a dict with ``phones`` plus an utterance-level mismatch guard        """
        profile = self._profile_manager.get_or_create(learner_id)
        if detector == DETECTOR_NEURAL:
            phones_out = self._analyze_neural(waveform, transcript, profile)
        else:
            phones_out = self._analyze_baseline(waveform, transcript, profile)
        self._profile_manager.save(learner_id)

        total = max(len(phones_out), 1)
        correct = sum(1 for p in phones_out if not p["is_mispronounced"])
        match_confidence = round(correct / total, 3)
        match_warning = match_confidence < 0.5

        implausible, rate = _articulation_implausible(waveform, transcript)
        articulation_rate = None
        if implausible:
            match_warning = True
            match_confidence = min(match_confidence, 0.2)
            articulation_rate = rate

        return {
            "phones": phones_out,
            "match_confidence": match_confidence,
            "match_warning": match_warning,
            "articulation_rate": articulation_rate,
        }

    def profile_dict(self, learner_id: str) -> dict:
        return self._profile_manager.profile_dict(learner_id)

    # -- Stage 4 calibration: discover systematic errors from a few sentences --
    def _collect_phone_observations(self, waveform, transcript: str) -> list[tuple]:
        """Return [(intended_phone, learner_MFCC_sequence), ...] for one utterance.
        """
        expected: list[str] = []
        for w in transcript.split():
            w_clean = w.strip(".,!?;:'\"")
            expected.extend(_phones_for_word(w_clean))

        recognizer = PhonemeRecognizer.instance()
        recognized = recognizer.recognize(waveform, sr=SAMPLE_RATE)
        recog_phones = [r.phone for r in recognized]
        ops = align(expected, recog_phones)

        obs: list[tuple] = []
        for op in ops:
            if op.op in ("match", "sub") and op.recognized_index is not None:
                rp = recognized[op.recognized_index]
                seg = waveform[rp.start_sample:rp.end_sample]
                if len(seg) >= 160:
                    obs.append((op.expected, extract_mfcc_sequence(seg, sr=SAMPLE_RATE)))
        return obs

    def calibrate(self, learner_id: str, sessions: list[tuple]) -> list[dict]:
        """Discover the learner's systematic error patterns from a few sentences.

        Args:
            learner_id: learner identifier.
            sessions:   list of (waveform, transcript) calibration recordings.

        Returns:
            List of discovered rule dicts (also stored in the learner's profile).
        """
        bank = self._get_ref_bank()

        observations: list[tuple] = []
        for waveform, transcript in sessions:
            observations.extend(self._collect_phone_observations(waveform, transcript))

        rules = discover_error_patterns(observations, bank)
        rules_dict = rules_to_dict(rules)
        profile = self._profile_manager.get_or_create(learner_id)
        profile.set_discovered_rules(rules_dict)
        self._profile_manager.save(learner_id)
        _LOG.info("Calibration for %s discovered %d rule(s) from %d observations.",
                  learner_id, len(rules), len(observations))
        return [rules_dict[p] for p in rules_dict]

    # -- classical (Random Forest) path 
    def _analyze_baseline(self, waveform, transcript: str, profile: LearnerProfile) -> list[dict]:
        """Does this segment match the expected phone? (assumes the transcript is correct)"""
        detector = self._get_baseline()
        results = detector.detect(waveform, transcript, sr=SAMPLE_RATE)

        phones_out = []
        for r in results:
            dtw = None
            is_mispr = r.is_mispronounced
            score = r.prob_mispronounced
            if is_mispr and r.learner_mfcc is not None:
                native_mfcc = self._native_reference_mfcc(r.phone)
                if native_mfcc is not None:
                    dtw = round(dtw_distance(r.learner_mfcc, native_mfcc), 4)

            # Stage 4 -> Stage 3 loop: a previously discovered systematic error
            is_systematic = False
            rule = profile.rule_for(r.phone)
            if rule is not None:
                is_systematic = True
                is_mispr = True
                score = max(score, _SYSTEMATIC_SCORE_FLOOR)

            if is_mispr:
                profile.update(r.phone, dtw if dtw is not None else 30.0)

            text_hint, learner_audio_uri = None, None
            if is_mispr:
                feedback = generate_feedback(r.phone, r.audio_clip, sr=SAMPLE_RATE)
                text_hint = feedback.text_hint
                learner_audio_uri = feedback.learner_audio_uri or None

            phones_out.append({
                "phone": r.phone,
                "word": r.word,
                "prob_mispronounced": score,
                "is_mispronounced": is_mispr,
                "substitution": None,  # RF does not identify the substitute
                "dtw_distance": dtw,
                "is_systematic": is_systematic,
                "text_hint": text_hint,
                "learner_audio_uri": learner_audio_uri,
            })
        return phones_out

    # -- neural (wav2vec2) path 
    def _analyze_neural(self, waveform, transcript: str, profile: LearnerProfile) -> list[dict]:
        """Recognise what was actually said, align it to the expected phones."""
        expected: list[tuple[str, str]] = []  # (word, ARPABET phone)
        for w in transcript.split():
            w_clean = w.strip(".,!?;:'\"")
            for ph in _phones_for_word(w_clean):
                expected.append((w_clean, ph))
        exp_phones = [ph for _, ph in expected]

        recognizer = PhonemeRecognizer.instance()
        recognized = recognizer.recognize(waveform, sr=SAMPLE_RATE)
        recog_phones = [r.phone for r in recognized]

        ops = align(exp_phones, recog_phones)

        pad = int(_CLIP_PAD_S * SAMPLE_RATE)
        phones_out = []
        for op in ops:
            if op.op == "ins" or op.expected_index is None:
                continue
            word, phone = expected[op.expected_index]
            substitution = None
            dtw = None
            audio_clip = None

            if op.op == "match":
                is_mispr = False
                score = 0.2
            elif op.op == "sub":
                substitution = op.recognized
                if is_confusable(phone, op.recognized):
                    is_mispr = False
                    score = 0.55
                else:
                    is_mispr = True
                    score = 0.85
                if op.recognized_index is not None:
                    rp = recognized[op.recognized_index]
                    learner_mfcc = extract_mfcc_sequence(
                        waveform[rp.start_sample:rp.end_sample], sr=SAMPLE_RATE)
                    native_mfcc = self._native_reference_mfcc(phone)
                    if native_mfcc is not None:
                        dtw = round(dtw_distance(learner_mfcc, native_mfcc), 4)
                    lo = max(0, rp.start_sample - pad)
                    hi = min(len(waveform), rp.end_sample + pad)
                    audio_clip = waveform[lo:hi]
            else:  # "del" -- the expected phone was omitted entirely
                is_mispr = True
                score = 0.9

            # Stage 4 -> Stage 3 loop: a previously discovered systematic error
            is_systematic = False
            rule = profile.rule_for(phone)
            if rule is not None:
                is_systematic = True
                is_mispr = True
                score = max(score, _SYSTEMATIC_SCORE_FLOOR)

            if is_mispr:
                profile.update(phone, dtw if dtw is not None else 30.0)

            text_hint, learner_audio_uri = None, None
            if is_mispr:
                feedback = generate_feedback(phone, audio_clip, sr=SAMPLE_RATE)
                text_hint = feedback.text_hint
                learner_audio_uri = feedback.learner_audio_uri or None

            phones_out.append({
                "phone": phone,
                "word": word,
                "prob_mispronounced": score,
                "is_mispronounced": is_mispr,
                "substitution": substitution,
                "dtw_distance": dtw,
                "is_systematic": is_systematic,
                "text_hint": text_hint,
                "learner_audio_uri": learner_audio_uri,
            })
        return phones_out

"""CAPT system orchestrator (Stage 3 dual-detector dispatch).
"""

from __future__ import annotations

import os

from src.classification.baseline_detector import BaselineDetector
from src.classification.phoneme_align import align, is_confusable
from src.classification.phoneme_recognizer import PhonemeRecognizer
from src.data.aligner import _phones_for_word
from src.data.librispeech import build_native_exemplars
from src.features.extractor import SAMPLE_RATE, extract_mfcc_sequence
from src.personalization.dtw import dtw_distance
from src.personalization.profiler import LearnerProfile, ProfileManager
from src.utils.logger import get_logger

_LOG = get_logger(__name__)

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
_CORPUS_DIR = os.path.join(_BASE_DIR, "data", "librispeech")
_BASELINE_MODEL_PATH = os.path.join(_BASE_DIR, "models", "baseline_rf.pkl")
_PROFILE_DIR = os.path.join(_BASE_DIR, "models", "profiles")

DETECTOR_BASELINE = "baseline"
DETECTOR_NEURAL = "neural"


class CAPTSystem:
    """Dispatches per-phone analysis to the classical or neural detector."""

    def __init__(self) -> None:
        self._profile_manager = ProfileManager(_PROFILE_DIR)
        self._baseline: BaselineDetector | None = None
        self._exemplars: dict | None = None

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

    # -- public API 
    def analyze(self, waveform, transcript: str, learner_id: str,
                detector: str = DETECTOR_BASELINE) -> list[dict]:
        """Run one detector's full per-phone analysis and update the learner profile."""
        profile = self._profile_manager.get_or_create(learner_id)
        if detector == DETECTOR_NEURAL:
            phones_out = self._analyze_neural(waveform, transcript, profile)
        else:
            phones_out = self._analyze_baseline(waveform, transcript, profile)
        self._profile_manager.save(learner_id)
        return phones_out

    def profile_dict(self, learner_id: str) -> dict:
        return self._profile_manager.profile_dict(learner_id)

    # -- classical (Random Forest) path 
    def _analyze_baseline(self, waveform, transcript: str, profile: LearnerProfile) -> list[dict]:
        """Does this segment match the expected phone? (assumes the transcript is correct)"""
        detector = self._get_baseline()
        results = detector.detect(waveform, transcript, sr=SAMPLE_RATE)

        phones_out = []
        for r in results:
            dtw = None
            if r.is_mispronounced and r.learner_mfcc is not None:
                native_mfcc = self._native_reference_mfcc(r.phone)
                if native_mfcc is not None:
                    dtw = round(dtw_distance(r.learner_mfcc, native_mfcc), 4)
            if r.is_mispronounced:
                profile.update(r.phone, dtw if dtw is not None else 30.0)
            phones_out.append({
                "phone": r.phone,
                "word": r.word,
                "prob_mispronounced": r.prob_mispronounced,
                "is_mispronounced": r.is_mispronounced,
                "substitution": None,  # RF does not identify the substitute
                "dtw_distance": dtw,
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

        phones_out = []
        for op in ops:
            if op.op == "ins" or op.expected_index is None:
                continue
            word, phone = expected[op.expected_index]
            substitution = None
            dtw = None

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
            else:  # "del" -- the expected phone was omitted entirely
                is_mispr = True
                score = 0.9

            if is_mispr:
                profile.update(phone, dtw if dtw is not None else 30.0)

            phones_out.append({
                "phone": phone,
                "word": word,
                "prob_mispronounced": score,
                "is_mispronounced": is_mispr,
                "substitution": substitution,
                "dtw_distance": dtw,
            })
        return phones_out

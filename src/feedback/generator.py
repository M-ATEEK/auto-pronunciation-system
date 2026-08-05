"""Feedback assembler for Stage 5 of the CAPT pipeline.

For each flagged (mispronounced) phone, produces:
  1. Text feedback articulatory hint for the target phone
  2. Audio feedback the learner's OWN production, as a base64 data URI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.feedback.hints import get_hint
from src.feedback.tts import wav_data_uri


@dataclass
class FeedbackItem:
    """Text + audio feedback for one mispronounced phone."""

    phone: str
    text_hint: str
    learner_audio_uri: str = ""


def generate_feedback(phone: str, learner_audio: np.ndarray | None = None,
                       sr: int = 16000) -> FeedbackItem:
    """Produce a FeedbackItem for a mispronounced phone.    """
    learner_audio_uri = ""
    if learner_audio is not None and len(learner_audio) > 0:
        learner_audio_uri = wav_data_uri(learner_audio, sr)

    return FeedbackItem(
        phone=phone,
        text_hint=get_hint(phone),
        learner_audio_uri=learner_audio_uri,
    )

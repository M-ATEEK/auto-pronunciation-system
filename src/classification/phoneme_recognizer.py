"""Neural phoneme recogniser  Stage 3 acoustic model (alternative to the RF baseline).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

_MODEL_NAME = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"

# wav2vec2 downsamples 16 kHz audio by 320x -> one CTC frame per 20 ms.
_SAMPLES_PER_FRAME = 320

# IPA (model vocabulary) -> ARPABET (CMU dictionary / rest of pipeline).
_IPA_TO_ARPABET = {
    "ɑ": "AA", "æ": "AE", "ə": "AH", "aʊ": "AW", "aɪ": "AY",
    "b": "B", "ʧ": "CH", "d": "D", "ð": "DH", "ɾ": "T",
    "ɛ": "EH", "ɝ": "ER", "eɪ": "EY", "f": "F", "g": "G",
    "h": "HH", "ɪ": "IH", "i": "IY", "ʤ": "JH", "k": "K",
    "l": "L", "m": "M", "n": "N", "ŋ": "NG", "oʊ": "OW",
    "ɔɪ": "OY", "p": "P", "ɹ": "R", "s": "S", "ʃ": "SH",
    "t": "T", "θ": "TH", "ʊ": "UH", "u": "UW", "v": "V",
    "w": "W", "j": "Y", "z": "Z",
}
_NON_PHONE = {"|", " ", "[UNK]", "[PAD]", "<s>", "</s>", "<pad>", "<unk>"}


@dataclass
class RecognizedPhone:
    """One phone the acoustic model detected in the learner audio."""

    phone: str            # ARPABET symbol
    start_sample: int     # inclusive
    end_sample: int       # exclusive
    confidence: float     # mean posterior probability over the phone's frames


class PhonemeRecognizer:
    """Lazy-loaded singleton wrapper around the wav2vec2 phoneme model."""

    _instance: Optional["PhonemeRecognizer"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._id_to_token: dict[int, str] = {}
        self._blank_id: int = 0

    @classmethod
    def instance(cls) -> "PhonemeRecognizer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def load(self) -> "PhonemeRecognizer":
        """Load model + processor into memory (idempotent). Downloads on first use."""
        if self._model is not None:
            return self
        import torch  # local import so the module is importable without torch
        from transformers import AutoModelForCTC, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(_MODEL_NAME)
        self._model = AutoModelForCTC.from_pretrained(_MODEL_NAME)
        self._model.eval()
        vocab = self._processor.tokenizer.get_vocab()
        self._id_to_token = {idx: tok for tok, idx in vocab.items()}
        self._blank_id = self._model.config.pad_token_id or vocab.get("[PAD]", 0)
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def recognize(self, audio: np.ndarray, sr: int = 16000) -> list[RecognizedPhone]:
        """Return the ARPABET phones detected in ``audio`` with timing + confidence."""
        self.load()
        torch = self._torch

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return []

        inputs = self._processor(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(inputs.input_values).logits  # (1, T, V)
        probs = torch.softmax(logits, dim=-1)[0]               # (T, V)
        ids = torch.argmax(probs, dim=-1).tolist()             # (T,)
        frame_conf = probs.max(dim=-1).values.tolist()         # (T,)

        return self._collapse(ids, frame_conf)

    def _collapse(self, ids: list[int], frame_conf: list[float]) -> list[RecognizedPhone]:
        """CTC-collapse frame labels into timed ARPABET phones."""
        out: list[RecognizedPhone] = []
        run_id: Optional[int] = None
        run_start = 0
        run_confs: list[float] = []

        def flush(end_frame: int) -> None:
            if run_id is None:
                return
            tok = self._id_to_token.get(run_id, "")
            arp = _IPA_TO_ARPABET.get(tok)
            if arp is not None and run_confs:
                out.append(RecognizedPhone(
                    phone=arp,
                    start_sample=run_start * _SAMPLES_PER_FRAME,
                    end_sample=end_frame * _SAMPLES_PER_FRAME,
                    confidence=float(np.mean(run_confs)),
                ))

        for f, (cid, conf) in enumerate(zip(ids, frame_conf)):
            if cid == run_id:
                run_confs.append(conf)
                continue
            flush(f)
            run_id = cid
            run_start = f
            run_confs = [conf]
        flush(len(ids))
        return out

    def phones(self, audio: np.ndarray, sr: int = 16000) -> list[str]:
        """Convenience: just the ARPABET phone strings, in order."""
        return [p.phone for p in self.recognize(audio, sr)]

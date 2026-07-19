"""Real LibriSpeech training-data loader (Stage 1 → Stage 3 baseline).

Features are extracted from **real** LibriSpeech native English read speech, so
the classical baseline trains on genuine data.

Native-only augmentation
------------------------
For a target phone *P*:
  * **positive** examples = real feature vectors of segments that ARE *P*;
  * **negative** examples = real feature vectors of segments of OTHER phones
    (native audio that is a *mismatch* for *P*).

No non-native or hand-labelled mispronunciation data is used — the negative
class is manufactured purely from native speech.

Expected directory layout (standard LibriSpeech):
    <root>/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac
    <root>/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt
The loader searches recursively for ``*.trans.txt`` so it works with dev-clean,
test-clean, or train-clean-100 unchanged.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.aligner import DictionaryAligner, phones_for_text
from src.features.extractor import N_MFCC, extract_features, extract_mfcc_sequence
from src.utils.phoneme_constants import ENGLISH_PHONEMES

import soundfile as sf

SAMPLE_RATE = 16_000
FEATURE_DIM = 17


def find_librispeech_root(base: str) -> Optional[str]:
    """Return the directory under ``base`` that contains *.trans.txt files."""
    base_path = Path(base)
    if not base_path.exists():
        return None
    for trans in base_path.rglob("*.trans.txt"):
        return str(trans.parent.parent.parent) if trans.parent.parent.parent else str(base_path)
    return None


def _iter_utterances(root: str):
    """Yield (flac_path, transcript) for every utterance under ``root``."""
    for trans_file in Path(root).rglob("*.trans.txt"):
        chapter_dir = trans_file.parent
        with open(trans_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                utt_id, _, text = line.partition(" ")
                flac = chapter_dir / f"{utt_id}.flac"
                if flac.exists():
                    yield str(flac), text


def _load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    return audio.astype(np.float32)


def _utterance_mfcc_mean(audio: np.ndarray) -> np.ndarray:
    """Mean of the 13 MFCC coefficients over a whole utterance (for CMN)."""
    seq = extract_mfcc_sequence(audio, sr=SAMPLE_RATE)  # (T, N_MFCC)
    if len(seq) == 0:
        return np.zeros(N_MFCC, dtype=np.float32)
    return seq.mean(axis=0).astype(np.float32)


def _segment_features(audio: np.ndarray, transcript: str, aligner: DictionaryAligner,
                      cmn: bool) -> list[tuple[str, np.ndarray]]:
    """Return [(phone, 17-dim feature vector), …] for one utterance."""
    segments = aligner.align(audio, SAMPLE_RATE, transcript)
    if not segments:
        return []

    # Utterance-level Cepstral Mean Normalisation (channel/speaker robustness),
    # applied identically here and at inference.
    utt_mean = _utterance_mfcc_mean(audio) if cmn else np.zeros(N_MFCC, dtype=np.float32)

    out: list[tuple[str, np.ndarray]] = []
    for seg in segments:
        if len(seg.audio) < 160:  # < 10 ms — too short to feature-extract
            continue
        feat = extract_features(seg.audio, sr=SAMPLE_RATE, n_phones=1).copy()
        if cmn:
            feat[:N_MFCC] = feat[:N_MFCC] - utt_mean
        out.append((seg.phone, feat))
    return out


def corpus_stats(corpus_dir: str, sample_utterances: int = 200, seed: int = 0) -> dict:
    """Return a quick, feature-free overview of the corpus and its phone coverage.

    Phone counts come from aligning transcripts to phones (no audio decoding or
    feature extraction), so this is fast enough for an API call.
    """
    root = find_librispeech_root(corpus_dir)
    if root is None:
        return {"available": False, "corpus_dir": corpus_dir}

    utts = list(_iter_utterances(root))
    speakers = {Path(flac).name.split("-")[0] for flac, _ in utts}

    rng = random.Random(seed)
    sample = utts[:]
    rng.shuffle(sample)
    sample = sample[:sample_utterances]

    phone_counts: Counter = Counter()
    for _flac, text in sample:
        phone_counts.update(phones_for_text(text))

    counts = dict(sorted(phone_counts.items()))
    values = list(counts.values())
    return {
        "available": True,
        "corpus_root": root,
        "total_utterances": len(utts),
        "speakers": len(speakers),
        "sampled_utterances": len(sample),
        "phones_covered": len(counts),
        "min_per_phone": min(values) if values else 0,
        "max_per_phone": max(values) if values else 0,
        "phone_counts": counts,
        "augmentation": (
            "native-only: for each phone, positives = its own segments; "
            "negatives = other phones' segments (all native speech)"
        ),
    }


def build_native_exemplars(corpus_dir: str, per_phone: int = 6, max_utterances: int = 200,
                           cmn: bool = True, seed: int = 0) -> dict[str, list[np.ndarray]]:
    """Return {phone: [MFCC-sequence, …]} of REAL native LibriSpeech phones.

    These per-phone exemplars are the native reference against which a learner's
    phone production is DTW-compared during error-pattern discovery (Stage 4).
    """
    root = find_librispeech_root(corpus_dir) or corpus_dir
    aligner = DictionaryAligner()
    rng = random.Random(seed)

    bank: dict[str, list[np.ndarray]] = {}
    utts = list(_iter_utterances(root))
    rng.shuffle(utts)
    processed = 0
    for flac, text in utts:
        if processed >= max_utterances:
            break
        try:
            audio = _load_audio(flac)
        except Exception:
            continue
        for seg in aligner.align(audio, SAMPLE_RATE, text):
            if len(seg.audio) < 160:
                continue
            slot = bank.setdefault(seg.phone, [])
            if len(slot) >= per_phone:
                continue
            seq = extract_mfcc_sequence(seg.audio, sr=SAMPLE_RATE)
            if cmn and len(seq) > 0:
                seq = seq - seq.mean(axis=0, keepdims=True)
            slot.append(seq.astype(np.float32))
        processed += 1
        if all(len(v) >= per_phone for v in bank.values()) and len(bank) >= 35:
            break
    return bank


def build_training_data(corpus_dir: str, phones: list[str] | None = None,
                        max_utterances: int = 400, min_per_phone: int = 20,
                        neg_ratio: float = 1.0, cmn: bool = True, seed: int = 0
                        ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build {phone: (X_pos, X_neg)} of REAL 17-dim features from LibriSpeech."""
    root = find_librispeech_root(corpus_dir) or corpus_dir
    phones = phones or ENGLISH_PHONEMES
    aligner = DictionaryAligner()
    rng = random.Random(seed)

    # 1) Collect real feature vectors grouped by phone.
    by_phone: dict[str, list[np.ndarray]] = {p: [] for p in phones}
    utts = list(_iter_utterances(root))
    rng.shuffle(utts)
    processed = 0
    for flac, text in utts:
        if processed >= max_utterances:
            break
        try:
            audio = _load_audio(flac)
        except Exception:
            continue
        for phone, feat in _segment_features(audio, text, aligner, cmn):
            if phone in by_phone:
                by_phone[phone].append(feat)
        processed += 1

    # 2) Native-only augmentation: positives = this phone; negatives = a random
    #    sample of OTHER phones' real segments.
    all_phones_with_data = [p for p, v in by_phone.items() if len(v) >= min_per_phone]
    data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for phone in all_phones_with_data:
        pos = np.stack(by_phone[phone]).astype(np.float32)
        n_neg = max(int(len(pos) * neg_ratio), 1)
        pool: list[np.ndarray] = []
        for other in all_phones_with_data:
            if other != phone:
                pool.extend(by_phone[other])
        if not pool:
            continue
        neg_idx = rng.choices(range(len(pool)), k=n_neg)
        neg = np.stack([pool[i] for i in neg_idx]).astype(np.float32)
        data[phone] = (pos, neg)
    return data

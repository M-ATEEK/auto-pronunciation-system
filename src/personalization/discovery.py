"""Unsupervised error-pattern discovery via DTW clustering (Stage 4).

Discovers a learner's systematic pronunciation errors from a few read-aloud
calibration sentences, using DTW clustering no labels, no non-native
training data.

Method
------
1. During calibration the learner reads several sentences. For every intended
   phone occurrence, the learner's MFCC segment and the intended phone are
   held.
2. Each occurrence is DTW - compared against a bank of real native exemplars
   (``NativeReferenceBank``) for the intended phone and for a set of likely
   substitute phones. This yields, per occurrence:
       d_intended    DTW distance to the intended phone
       d_best_other  DTW distance to the nearest other candidate phone
       margin        d_intended  d_best_other  (positive => sounds like another phone)
3. All occurrences are clustered (k-means, k=2) in this DTW-feature space. The
   cluster with the larger mean d_intended is the "deviant" cluster the
   learner's systematically mis-produced sounds, discovered from the data
   itself rather than from a fixed threshold.
4. Within the deviant cluster, occurrences are grouped by intended phone. A
   phone with enough deviant occurrences becomes a discovered rule: a
   substitution P->Q when a dominant nearest-other phone Q emerges, otherwise
   a distortion of P.

The discovered rules are stored in the learner profile and used to adjust the
detector's confidence for that learner (Stage 3 <-> Stage 4 loop).
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from src.personalization.dtw import dtw_distance

# L1-transfer-informed candidate substitutes per phone (ARPABET). Kept small so
_SUBSTITUTES: dict[str, list[str]] = {
    "TH": ["S", "T", "F", "DH"], "DH": ["D", "Z", "V", "TH"],
    "R": ["L", "W", "ER"], "L": ["R", "W"],
    "V": ["W", "F", "B"], "W": ["V"], "F": ["P", "V"],
    "Z": ["S"], "S": ["SH", "TH", "Z"], "SH": ["S", "CH"],
    "IH": ["IY", "EH"], "IY": ["IH"], "EH": ["AE", "IH"], "AE": ["EH", "AA"],
    "AA": ["AH", "AO"], "AO": ["AA", "OW"], "UH": ["UW"], "UW": ["UH"],
    "NG": ["N"], "N": ["NG", "M"], "M": ["N"],
    "B": ["P", "V"], "P": ["B", "F"], "D": ["T"], "T": ["D", "TH"],
    "G": ["K"], "K": ["G"], "JH": ["CH", "ZH"], "CH": ["SH", "JH"],
    "ER": ["R", "AH"], "Y": ["IY"], "ZH": ["SH", "Z"],
    "AH": ["AA", "EH"], "EY": ["EH", "IH"], "OW": ["AO"], "AY": ["AE"],
    "AW": ["AA"], "OY": ["OW"], "HH": ["F"],
}

_MIN_DEVIANT_COUNT = 2      # deviant occurrences of a phone to call it systematic
_SUB_DOMINANCE = 0.5        # fraction of deviant occ. sharing one Q -> substitution
_DEVIANT_RATIO = 1.5        # deviant cluster mean must exceed correct mean by this
                            # factor, else no genuine error population exists


@dataclass
class ErrorRule:
    """A discovered systematic error for one intended phone."""

    phone: str
    type: str                     # "substitution" | "distortion"
    substituted_with: Optional[str]
    support: int                  # number of deviant occurrences
    confidence: float             # fraction of the phone's occurrences that were deviant
    mean_dtw: float


class NativeReferenceBank:
    """Bank of real native per-phone MFCC exemplars for DTW comparison."""

    def __init__(self, exemplars: dict[str, list[np.ndarray]]) -> None:
        self._exemplars = exemplars

    @classmethod
    def load(cls, path: str) -> "NativeReferenceBank":
        with open(path, "rb") as fh:
            return cls(pickle.load(fh))

    @property
    def phones(self) -> list[str]:
        return list(self._exemplars.keys())

    def has(self, phone: str) -> bool:
        return bool(self._exemplars.get(phone))

    def min_dtw(self, phone: str, seq: np.ndarray) -> Optional[float]:
        """Minimum DTW distance from ``seq`` to any native exemplar of ``phone``."""
        exs = self._exemplars.get(phone)
        if not exs or seq is None or len(seq) == 0:
            return None
        s = seq - seq.mean(axis=0, keepdims=True)  # CMN, matching exemplar prep
        return min(dtw_distance(s, ex) for ex in exs)


def _candidates(phone: str, bank: NativeReferenceBank) -> list[str]:
    cands = list(_SUBSTITUTES.get(phone, []))
    return [c for c in cands if c != phone and bank.has(c)]


def discover_error_patterns(
    observations: list[tuple[str, np.ndarray]],
    bank: NativeReferenceBank,
    min_deviant_count: int = _MIN_DEVIANT_COUNT,
) -> list[ErrorRule]:
    """Discover systematic error rules from calibration observations.

    Args:
        observations: [(intended_phone, learner_MFCC_sequence), ...]
        bank:         native reference exemplar bank.

    Returns:
        List of ErrorRule, one per discovered systematic error.
    """
    feats: list[list[float]] = []
    meta: list[tuple[str, Optional[str]]] = []  # (intended, best_other)
    for phone, seq in observations:
        d_int = bank.min_dtw(phone, seq)
        if d_int is None:
            continue
        best_other, best_d = None, np.inf
        for q in _candidates(phone, bank):
            dq = bank.min_dtw(q, seq)
            if dq is not None and dq < best_d:
                best_other, best_d = q, dq
        d_best_other = float(best_d) if best_other is not None else d_int
        feats.append([float(d_int), d_best_other, float(d_int - d_best_other)])
        meta.append((phone, best_other))

    if not feats:
        return []

    X = np.asarray(feats, dtype=np.float64)
    deviant_mask = _cluster_deviant(X)

    by_phone: dict[str, dict] = {}
    for (phone, best_other), is_dev, row in zip(meta, deviant_mask, X):
        agg = by_phone.setdefault(phone, {"total": 0, "dev": 0, "others": {}, "dsum": 0.0})
        agg["total"] += 1
        if is_dev:
            agg["dev"] += 1
            agg["dsum"] += row[0]
            if best_other is not None and row[2] > 0:  # margin>0 -> sounds like Q
                agg["others"][best_other] = agg["others"].get(best_other, 0) + 1

    rules: list[ErrorRule] = []
    for phone, agg in by_phone.items():
        if agg["dev"] < min_deviant_count:
            continue
        dominant_q, dom_count = None, 0
        for q, c in agg["others"].items():
            if c > dom_count:
                dominant_q, dom_count = q, c
        is_sub = dominant_q is not None and dom_count >= _SUB_DOMINANCE * agg["dev"]
        rules.append(ErrorRule(
            phone=phone,
            type="substitution" if is_sub else "distortion",
            substituted_with=dominant_q if is_sub else None,
            support=int(agg["dev"]),
            confidence=round(agg["dev"] / max(agg["total"], 1), 3),
            mean_dtw=round(agg["dsum"] / max(agg["dev"], 1), 3),
        ))
    rules.sort(key=lambda r: (-r.support, -r.mean_dtw))
    return rules


def _cluster_deviant(X: np.ndarray) -> np.ndarray:
    """Return a boolean mask marking the 'deviant' occurrences.
    """
    n = len(X)
    d_int = X[:, 0]
    if np.allclose(d_int, d_int[0]):
        return np.zeros(n, dtype=bool)

    if n < 4:
        med = float(np.median(d_int))
        return d_int > _DEVIANT_RATIO * max(med, 1e-6)

    try:
        from sklearn.cluster import KMeans
        Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xs).labels_
        mean0 = d_int[labels == 0].mean() if (labels == 0).any() else -np.inf
        mean1 = d_int[labels == 1].mean() if (labels == 1).any() else -np.inf
        deviant_label = 0 if mean0 > mean1 else 1
        correct_mean = min(mean0, mean1)
        deviant_mean = max(mean0, mean1)
        if correct_mean > 0 and deviant_mean < _DEVIANT_RATIO * correct_mean:
            return np.zeros(n, dtype=bool)
        return labels == deviant_label
    except Exception:
        med = float(np.median(d_int))
        return d_int > _DEVIANT_RATIO * max(med, 1e-6)


def rules_to_dict(rules: list[ErrorRule]) -> dict[str, dict]:
    """Serialise discovered rules keyed by phone (for the learner profile)."""
    return {r.phone: asdict(r) for r in rules}

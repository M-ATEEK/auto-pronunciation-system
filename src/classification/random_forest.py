"""Per-phoneme binary Random Forest mispronunciation detector.

One RandomForestClassifier is trained per phoneme.
Binary labels: 0 = correct pronunciation, 1 = mispronounced.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = None
RANDOM_STATE = 42


class PhonemeRandomForest:
    """One RF binary classifier per phoneme."""

    def __init__(self, n_estimators: int = DEFAULT_N_ESTIMATORS,
                 max_depth: int | None = DEFAULT_MAX_DEPTH,
                 random_state: int = RANDOM_STATE) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._random_state = random_state
        self._classifiers: dict[str, RandomForestClassifier] = {}

    def _make_clf(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            min_samples_split=2,
            class_weight="balanced",
            random_state=self._random_state,
        )

    def fit(self, phone: str, X_pos: np.ndarray, X_neg: np.ndarray) -> "PhonemeRandomForest":
        """Train a binary classifier for a single phoneme.

        Args:
            phone: ARPABET phoneme symbol.
            X_pos: Positive examples (this phone), shape (N, 17).
            X_neg: Negative examples (other phones), shape (M, 17).
        """
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.zeros(len(X_pos)), np.ones(len(X_neg))])
        clf = self._make_clf()
        clf.fit(X, y)
        self._classifiers[phone] = clf
        return self

    def fit_all(self, data: dict[str, tuple[np.ndarray, np.ndarray]]) -> "PhonemeRandomForest":
        """Train one classifier per phoneme from a {phone: (X_pos, X_neg)} mapping."""
        for phone, (X_pos, X_neg) in data.items():
            self.fit(phone, X_pos, X_neg)
        return self

    def predict_proba_mispronounced(self, phone: str, x: np.ndarray) -> float:
        """Return P(mispronounced | phone, features).

        Returns 0.5 if no classifier has been trained for this phone.
        """
        clf = self._classifiers.get(phone)
        if clf is None:
            return 0.5
        x2d = x.reshape(1, -1) if x.ndim == 1 else x
        proba = clf.predict_proba(x2d)
        classes = list(clf.classes_)
        if 1 in classes:
            return float(proba[0, classes.index(1)])
        return float(proba[0, -1])

    def is_mispronounced(self, phone: str, x: np.ndarray, threshold: float = 0.5) -> bool:
        """Return True if P(mispronounced) > threshold."""
        return self.predict_proba_mispronounced(phone, x) > threshold

    @property
    def trained_phones(self) -> list[str]:
        return list(self._classifiers.keys())

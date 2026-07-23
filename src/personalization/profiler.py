"""Learner profile: per-phone DTW error statistics (Stage 4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

SYSTEMATIC_COUNT_THRESHOLD = 3
SYSTEMATIC_DTW_THRESHOLD = 25.0


class PhoneStats(TypedDict):
    count: int
    mean_dtw: float
    is_systematic: bool


class LearnerProfile:
    """Per-phone DTW error statistics for one learner."""

    def __init__(self, learner_id: str, dtw_threshold: float = SYSTEMATIC_DTW_THRESHOLD,
                 count_threshold: int = SYSTEMATIC_COUNT_THRESHOLD) -> None:
        self.learner_id = learner_id
        self._dtw_threshold = dtw_threshold
        self._count_threshold = count_threshold
        self._phone_stats: dict[str, PhoneStats] = {}
        # Populated by Stage-4 DTW-clustering discovery (calibration step).
        self._discovered_rules: dict[str, dict] = {}

    def update(self, phone: str, dtw_distance: float) -> None:
        """Record one DTW observation for the given phone."""
        if phone not in self._phone_stats:
            self._phone_stats[phone] = {"count": 0, "mean_dtw": 0.0, "is_systematic": False}
        stats = self._phone_stats[phone]
        n = stats["count"]
        new_mean = (stats["mean_dtw"] * n + dtw_distance) / (n + 1)
        stats["count"] = n + 1
        stats["mean_dtw"] = round(new_mean, 4)
        stats["is_systematic"] = (
            stats["count"] >= self._count_threshold and stats["mean_dtw"] > self._dtw_threshold
        )

    def get_stats(self, phone: str) -> PhoneStats | None:
        return self._phone_stats.get(phone)

    def systematic_errors(self) -> list[str]:
        return [p for p, s in self._phone_stats.items() if s["is_systematic"]]

    def set_discovered_rules(self, rules: dict[str, dict]) -> None:
        self._discovered_rules = dict(rules)

    @property
    def discovered_rules(self) -> dict[str, dict]:
        return self._discovered_rules

    def rule_for(self, phone: str) -> dict | None:
        return self._discovered_rules.get(phone)

    def to_dict(self) -> dict:
        return {
            "learner_id": self.learner_id,
            "phone_stats": dict(self._phone_stats),
            "discovered_rules": dict(self._discovered_rules),
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LearnerProfile":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        profile = cls(learner_id=data["learner_id"])
        for phone, stats in data.get("phone_stats", {}).items():
            profile._phone_stats[phone] = PhoneStats(
                count=int(stats["count"]),
                mean_dtw=float(stats["mean_dtw"]),
                is_systematic=bool(stats["is_systematic"]),
            )
        profile._discovered_rules = dict(data.get("discovered_rules", {}))
        return profile


class ProfileManager:
    """In-memory store of LearnerProfile objects, with optional directory persistence."""

    def __init__(self, profile_dir: str | Path | None = None) -> None:
        self._profiles: dict[str, LearnerProfile] = {}
        self._dir = Path(profile_dir) if profile_dir else None

    def get_or_create(self, learner_id: str) -> LearnerProfile:
        if learner_id in self._profiles:
            return self._profiles[learner_id]
        if self._dir is not None:
            p = self._dir / f"{learner_id}.json"
            if p.exists():
                profile = LearnerProfile.load(p)
                self._profiles[learner_id] = profile
                return profile
        profile = LearnerProfile(learner_id)
        self._profiles[learner_id] = profile
        return profile

    def save(self, learner_id: str) -> None:
        if self._dir is None:
            return
        profile = self._profiles.get(learner_id)
        if profile is None:
            return
        profile.save(self._dir / f"{learner_id}.json")

    def profile_dict(self, learner_id: str) -> dict:
        return self.get_or_create(learner_id).to_dict()

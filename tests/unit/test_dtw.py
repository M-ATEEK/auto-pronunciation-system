"""Unit tests for DTW distance computation."""

from __future__ import annotations

import numpy as np
import pytest

from src.personalization.dtw import dtw_distance, dtw_path


def _seq(n_frames: int, n_dim: int = 13, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_frames, n_dim)).astype(np.float32)


class TestDTWDistance:
    def test_returns_positive_float(self):
        a = _seq(10)
        b = _seq(12, seed=1)
        d = dtw_distance(a, b)
        assert isinstance(d, float)
        assert d >= 0.0

    def test_identical_sequences_near_zero(self):
        a = _seq(15, seed=7)
        d = dtw_distance(a, a)
        assert d == pytest.approx(0.0, abs=1e-4)

    def test_different_sequences_positive(self):
        a = _seq(15, seed=0)
        b = _seq(15, seed=42)
        d = dtw_distance(a, b)
        assert d > 0.0

    def test_different_lengths(self):
        a = _seq(10)
        b = _seq(25)
        d = dtw_distance(a, b)
        assert d >= 0.0 and np.isfinite(d)

    def test_single_frame(self):
        a = np.ones((1, 13), dtype=np.float32)
        b = np.zeros((1, 13), dtype=np.float32)
        d = dtw_distance(a, b)
        assert d >= 0.0

    def test_empty_sequences(self):
        a = _seq(0)
        b = _seq(5)
        d = dtw_distance(a, b)
        assert d == pytest.approx(0.0)

    def test_scale_independence(self):
        """Larger distance between sequences means larger DTW distance."""
        base = _seq(15, seed=0)
        close = base + 0.01
        far = base + 10.0
        d_close = dtw_distance(base, close)
        d_far = dtw_distance(base, far)
        assert d_far > d_close

    def test_1d_input_accepted(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        d = dtw_distance(a, b)
        assert d >= 0.0


class TestDTWPath:
    def test_returns_tuple(self):
        a = _seq(8)
        b = _seq(10)
        result = dtw_path(a, b)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_path_starts_at_origin(self):
        a = _seq(8, seed=0)
        b = _seq(8, seed=1)
        dist, path = dtw_path(a, b)
        assert path[0] == (0, 0)

    def test_path_ends_at_last_frame(self):
        a = _seq(8, seed=0)
        b = _seq(10, seed=1)
        dist, path = dtw_path(a, b)
        assert path[-1] == (len(a) - 1, len(b) - 1)

    def test_path_distance_equals_distance_fn(self):
        a = _seq(6, seed=0)
        b = _seq(9, seed=1)
        d1 = dtw_distance(a, b)
        d2, _ = dtw_path(a, b)
        assert d1 == pytest.approx(d2, abs=1e-5)

    def test_empty_path(self):
        a = np.zeros((0, 13), dtype=np.float32)
        b = _seq(5)
        dist, path = dtw_path(a, b)
        assert dist == pytest.approx(0.0)
        assert path == []

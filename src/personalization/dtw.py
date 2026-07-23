"""Dynamic Time Warping (DTW) distance between MFCC sequences (Stage 4).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, metric: str = "euclidean") -> float:
    """Compute the DTW distance between two MFCC frame sequences.
    """
    if seq_a.ndim == 1:
        seq_a = seq_a.reshape(-1, 1)
    if seq_b.ndim == 1:
        seq_b = seq_b.reshape(-1, 1)

    T1, T2 = len(seq_a), len(seq_b)
    if T1 == 0 or T2 == 0:
        return 0.0

    cost = cdist(seq_a, seq_b, metric=metric).astype(np.float64)

    acc = np.full((T1, T2), np.inf, dtype=np.float64)
    acc[0, 0] = cost[0, 0]
    for i in range(1, T1):
        acc[i, 0] = acc[i - 1, 0] + cost[i, 0]
    for j in range(1, T2):
        acc[0, j] = acc[0, j - 1] + cost[0, j]
    for i in range(1, T1):
        for j in range(1, T2):
            acc[i, j] = cost[i, j] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])

    total_cost = acc[T1 - 1, T2 - 1]
    path_length = T1 + T2
    return float(total_cost / path_length)


def dtw_path(seq_a: np.ndarray, seq_b: np.ndarray,
             metric: str = "euclidean") -> tuple[float, list[tuple[int, int]]]:
    """Compute DTW distance and the optimal warping path.
    """
    if seq_a.ndim == 1:
        seq_a = seq_a.reshape(-1, 1)
    if seq_b.ndim == 1:
        seq_b = seq_b.reshape(-1, 1)

    T1, T2 = len(seq_a), len(seq_b)
    if T1 == 0 or T2 == 0:
        return 0.0, []

    cost = cdist(seq_a, seq_b, metric=metric).astype(np.float64)

    acc = np.full((T1, T2), np.inf, dtype=np.float64)
    acc[0, 0] = cost[0, 0]
    for i in range(1, T1):
        acc[i, 0] = acc[i - 1, 0] + cost[i, 0]
    for j in range(1, T2):
        acc[0, j] = acc[0, j - 1] + cost[0, j]
    for i in range(1, T1):
        for j in range(1, T2):
            acc[i, j] = cost[i, j] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])

    path: list[tuple[int, int]] = []
    i, j = T1 - 1, T2 - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
            if step == 0:
                i -= 1
                j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
    path.append((0, 0))
    path.reverse()

    total_cost = acc[T1 - 1, T2 - 1]
    path_length = T1 + T2
    return float(total_cost / path_length), path

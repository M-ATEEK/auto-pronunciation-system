"""Global alignment between an expected phone sequence and a recognised one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_MATCH = 0.0
_SUB = 1.0
_GAP = 1.0   # insertion or deletion

# Phones that are commonly confusable / allophonic -- a substitution between
# members of the same group is a "near-miss" rather than a gross error, so it
# is charged a reduced cost during alignment and the caller may treat it as a
# soft error rather than a hard mispronunciation.
_CONFUSABLE_GROUPS = [
    frozenset({"AH", "AX", "IH", "IX"}),      # reduced vowels / schwa
    frozenset({"AA", "AO"}),                   # cot/caught merger
    frozenset({"EH", "AE"}),
    frozenset({"IH", "IY"}),
    frozenset({"UH", "UW"}),
    frozenset({"ER", "R"}),
    frozenset({"M", "N", "NG"}),               # nasal place
    frozenset({"S", "Z"}),
    frozenset({"T", "D", "DX"}),               # alveolar stop / flap
    frozenset({"SH", "ZH", "CH", "JH"}),
]


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return _MATCH
    for grp in _CONFUSABLE_GROUPS:
        if a in grp and b in grp:
            return _SUB * 0.5
    return _SUB


@dataclass
class AlignOp:
    """One aligned position between expected and recognised sequences."""

    op: str                        # "match" | "sub" | "del" | "ins"
    expected: Optional[str]        # expected phone (None for insertion)
    recognized: Optional[str]      # recognised phone (None for deletion)
    expected_index: Optional[int]  # position in the expected sequence
    recognized_index: Optional[int] = None  # position in the recognised sequence

    @property
    def is_correct(self) -> bool:
        return self.op == "match"


def is_confusable(a: str, b: str) -> bool:
    """True if a->b is a near-miss (allophonic / commonly confused) substitution."""
    if a == b:
        return False
    return any(a in grp and b in grp for grp in _CONFUSABLE_GROUPS)


def align(expected: list[str], recognized: list[str]) -> list[AlignOp]:
    """Needleman-Wunsch global alignment of two phone sequences.

    Returns a list of AlignOp in expected-sequence order (insertions are placed
    at the point they occur). Every expected phone appears exactly once as the
    ``expected`` field of a match/sub/del op.
    """
    n, m = len(expected), len(recognized)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * _GAP
    for j in range(1, m + 1):
        dp[0][j] = j * _GAP

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + _sub_cost(expected[i - 1], recognized[j - 1])
            up = dp[i - 1][j] + _GAP     # deletion (expected not recognised)
            left = dp[i][j - 1] + _GAP   # insertion (extra recognised phone)
            dp[i][j] = min(diag, up, left)

    ops: list[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diag = dp[i - 1][j - 1] + _sub_cost(expected[i - 1], recognized[j - 1])
            if abs(dp[i][j] - diag) < 1e-9:
                e, r = expected[i - 1], recognized[j - 1]
                op = "match" if e == r else "sub"
                ops.append(AlignOp(op, e, r, i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + _GAP)) < 1e-9:
            ops.append(AlignOp("del", expected[i - 1], None, i - 1, None))
            i -= 1
            continue
        ops.append(AlignOp("ins", None, recognized[j - 1], None, j - 1))
        j -= 1

    ops.reverse()
    return ops

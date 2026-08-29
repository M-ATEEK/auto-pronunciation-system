from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classification.baseline_detector import BaselineDetector
from src.classification.content_verifier import (MISMATCH_RATIO_THRESHOLD,
                                                 verify_content)
from src.data.librispeech import (_iter_utterances, _load_audio,
                                  find_librispeech_root)
from src.features.extractor import SAMPLE_RATE
from src.utils.audio_io import has_speech, trim_silence

SR = SAMPLE_RATE


def correct_fraction(detector, audio, transcript) -> float | None:
    """Fraction of phones the classical detector calls correct."""
    res = detector.detect(audio, transcript, sr=SR)
    if not res:
        return None
    return sum(1 for r in res if not r.is_mispronounced) / len(res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/librispeech")
    ap.add_argument("--model", default="models/baseline_rf.pkl")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--skip", type=int, default=100)
    ap.add_argument("--out", default="evaluation/guard_results.json")
    args = ap.parse_args()

    root = find_librispeech_root(args.corpus) or args.corpus
    utts = list(_iter_utterances(root))
    random.Random(0).shuffle(utts)
    utts = utts[args.skip:]
    detector = BaselineDetector.load(args.model)
    rng = random.Random(11)

    # Collect utterances of at least three words, which is where the guard applies.
    pool = [(f, t) for f, t in utts if len(t.split()) >= 5][: args.pairs * 3]

    matched, mismatched = [], []
    blind_match, blind_mismatch = [], []
    for i in range(min(args.pairs, len(pool) - 1)):
        flac, text = pool[i]
        # a different transcript of comparable length
        others = [t for _, t in pool
                  if t != text and abs(len(t.split()) - len(text.split())) <= 3]
        if not others:
            continue
        wrong = rng.choice(others)
        try:
            audio = trim_silence(_load_audio(flac))
        except Exception:
            continue

        r_ok, _ = verify_content(audio, text)
        r_bad, _ = verify_content(audio, wrong)
        if r_ok is not None:
            matched.append(r_ok)
        if r_bad is not None:
            mismatched.append(r_bad)

        cf_ok = correct_fraction(detector, audio, text)
        cf_bad = correct_fraction(detector, audio, wrong)
        if cf_ok is not None:
            blind_match.append(cf_ok)
        if cf_bad is not None:
            blind_mismatch.append(cf_bad)

    m, mm = np.array(matched), np.array(mismatched)
    thr = MISMATCH_RATIO_THRESHOLD
    res = {
        "content_guard": {
            "pairs": int(min(len(m), len(mm))),
            "threshold": thr,
            "matched_ratio_mean": round(float(m.mean()), 4),
            "matched_ratio_worst": round(float(m.max()), 4),
            "mismatched_ratio_mean": round(float(mm.mean()), 4),
            "mismatched_ratio_best": round(float(mm.min()), 4),
            "false_alarms": int((m > thr).sum()),
            "misses": int((mm <= thr).sum()),
            "separation": round(float(mm.min() - m.max()), 4),
        },
        "classical_content_blindness": {
            "correct_fraction_matched": round(float(np.mean(blind_match)), 4),
            "correct_fraction_mismatched": round(float(np.mean(blind_mismatch)), 4),
            "difference": round(float(np.mean(blind_match) - np.mean(blind_mismatch)), 4),
        },
        "speech_presence_guard": {
            "silence_rejected": not has_speech(np.zeros(SR, dtype=np.float32)),
            "quiet_noise_rejected": not has_speech(
                (np.random.RandomState(0).randn(SR) * 0.01).astype(np.float32)),
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)

    g = res["content_guard"]
    print(f"content guard on {g['pairs']} pairs (threshold {thr})")
    print(f"  matched    mean {g['matched_ratio_mean']}  worst {g['matched_ratio_worst']}")
    print(f"  mismatched mean {g['mismatched_ratio_mean']}  best  {g['mismatched_ratio_best']}")
    print(f"  false alarms {g['false_alarms']}, misses {g['misses']}")
    b = res["classical_content_blindness"]
    print(f"classical 'correct' fraction: matched {b['correct_fraction_matched']}, "
          f"mismatched {b['correct_fraction_mismatched']}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()

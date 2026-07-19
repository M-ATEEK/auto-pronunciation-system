"""Train the Random-Forest baseline on real LibriSpeech native speech (Stage 3).

Pipeline:
  Stage 1 Data      : real LibriSpeech read speech (native English)
  Stage 2 Features  : 13 MFCC + F0 + RMS + speech-rate + pause  (17-dim)
  Stage 3 Classify  : native-only augmentation, one Random Forest per phone

Usage:
    .venv312/bin/python scripts/train_baseline.py \
        --corpus data/librispeech --max-utterances 100 --out models/baseline_rf.pkl
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from src.classification.baseline_detector import BaselineDetector
from src.classification.random_forest import PhonemeRandomForest
from src.data.librispeech import build_training_data


def _per_phone_holdout_eval(data: dict, seed: int = 0) -> dict:
    """Held-out accuracy/F1/AUC per phone, then averaged.

    Each phone's own (X_pos, X_neg) is split train/test and a *fresh* RF is
    fit on the train part only, so this measures whether that phone's actual
    positives are distinguishable from its own negatives. unlike a global
    flattened split, this never mixes phones together (a real "K" segment is
    always a positive for K and never doubles as some other phone's
    negative in the same evaluation).
    """
    accs, prs, rcs, f1s, aucs = [], [], [], [], []
    for phone, (pos, neg) in data.items():
        if len(pos) < 6 or len(neg) < 6:
            continue
        X = np.vstack([pos, neg])
        y = np.concatenate([np.zeros(len(pos)), np.ones(len(neg))])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        clf = PhonemeRandomForest().fit(phone, Xtr[ytr == 0], Xtr[ytr == 1])._classifiers[phone]
        pred = clf.predict(Xte)
        classes = list(clf.classes_)
        score = clf.predict_proba(Xte)[:, classes.index(1)] if 1 in classes else clf.predict_proba(Xte)[:, -1]
        accs.append(accuracy_score(yte, pred))
        pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
        prs.append(pr); rcs.append(rc); f1s.append(f1)
        if len(set(yte)) == 2:
            aucs.append(roc_auc_score(yte, score))
    return {
        "phones_evaluated": len(accs),
        "accuracy": float(np.mean(accs)),
        "precision": float(np.mean(prs)),
        "recall": float(np.mean(rcs)),
        "f1": float(np.mean(f1s)),
        "auc": float(np.mean(aucs)) if aucs else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/librispeech")
    ap.add_argument("--max-utterances", type=int, default=100)
    ap.add_argument("--min-per-phone", type=int, default=15)
    ap.add_argument("--out", default="models/baseline_rf.pkl")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/3] Extracting real features from LibriSpeech at {args.corpus} ...", flush=True)
    data = build_training_data(
        args.corpus, max_utterances=args.max_utterances,
        min_per_phone=args.min_per_phone, cmn=True, seed=0,
    )
    counts = {k: len(v[0]) for k, v in sorted(data.items())}
    print(f"      {len(data)} phones with data; positives/phone: "
          f"min={min(counts.values())}, max={max(counts.values())}, "
          f"median={int(np.median(list(counts.values())))}", flush=True)
    print(f"      extraction took {time.time() - t0:.1f}s", flush=True)

    print("[2/3] Per-phone held out evaluation (each phone vs its own split) ...", flush=True)
    metrics = _per_phone_holdout_eval(data)
    print(f"      phones evaluated: {metrics['phones_evaluated']}  "
          f"acc={metrics['accuracy']:.3f} precision={metrics['precision']:.3f} "
          f"recall={metrics['recall']:.3f} F1={metrics['f1']:.3f} AUC={metrics['auc']:.3f}",
          flush=True)

    print("training final Random Forest on all data (native only augmentation) ...", flush=True)
    rf = PhonemeRandomForest()
    rf.fit_all(data)

    print(f"[3/3] Saving baseline detector -> {args.out}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    BaselineDetector(rf, cmn=True, threshold=0.5).save(args.out)
    print(f"Done in {time.time() - t0:.1f}s. Trained phones: {len(rf.trained_phones)}", flush=True)


if __name__ == "__main__":
    main()

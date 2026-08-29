from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classification.baseline_detector import BaselineDetector
from src.data.aligner import DictionaryAligner
from src.data.librispeech import (_iter_utterances, _load_audio,
                                  find_librispeech_root)
from src.features.extractor import (N_MFCC, SAMPLE_RATE, extract_features,
                                    extract_mfcc_sequence)
from src.personalization.discovery import NativeReferenceBank
from src.utils.audio_io import trim_silence

SR = SAMPLE_RATE
BUCKETS = [(1, 10), (11, 20), (21, 50), (51, 10_000)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/librispeech")
    ap.add_argument("--model", default="models/baseline_rf.pkl")
    ap.add_argument("--bank", default="models/native_exemplars.pkl")
    ap.add_argument("--utterances", type=int, default=40)
    ap.add_argument("--skip", type=int, default=100)
    ap.add_argument("--out", default="evaluation/identification_results.json")
    args = ap.parse_args()

    detector = BaselineDetector.load(args.model)
    rf = detector._rf
    RF_PHONES = list(detector.trained_phones)
    bank = NativeReferenceBank.load(args.bank)
    BANK_PHONES = [p for p in bank.phones if bank.has(p)]

    root = find_librispeech_root(args.corpus) or args.corpus
    utts = list(_iter_utterances(root))
    random.Random(0).shuffle(utts)
    utts = utts[args.skip:]
    aligner = DictionaryAligner()

    rf_ranks, rf_margins = [], []
    dtw_ranks = []
    length_rows = []          # (n_phones, fraction the detector calls correct)
    used = 0

    for flac, text in utts:
        if used >= args.utterances:
            break
        try:
            audio = trim_silence(_load_audio(flac))
        except Exception:
            continue
        segs = aligner.align(audio, SR, text)
        segs = [s for s in segs if len(s.audio) >= 320]
        if len(segs) < 4:
            continue
        used += 1

        seq = extract_mfcc_sequence(audio, sr=SR)
        mean = seq.mean(axis=0).astype(np.float32) if len(seq) else np.zeros(N_MFCC, np.float32)

        n_correct, n_total = 0, 0
        for s in segs:
            feat = extract_features(s.audio, sr=SR, n_phones=1).copy()
            feat[:N_MFCC] -= mean

            # verification accuracy on known-correct native speech
            if s.phone in RF_PHONES:
                p = float(rf.predict_proba_mispronounced(s.phone, feat))
                n_total += 1
                n_correct += int(p <= detector._threshold)

                # identification by ranking every trained phone
                scores = np.array([1.0 - rf.predict_proba_mispronounced(q, feat)
                                   for q in RF_PHONES])
                order = np.argsort(-scores)
                rf_ranks.append(int(np.where(order == RF_PHONES.index(s.phone))[0][0]))
                rf_margins.append(float(scores[order[0]] - scores[order[1]]))

            # identification by DTW template matching
            if s.phone in BANK_PHONES:
                mseq = extract_mfcc_sequence(s.audio, sr=SR)
                if len(mseq) >= 3:
                    d = np.array([bank.min_dtw(q, mseq) if bank.min_dtw(q, mseq) is not None
                                  else np.inf for q in BANK_PHONES])
                    order = np.argsort(d)
                    dtw_ranks.append(int(np.where(order == BANK_PHONES.index(s.phone))[0][0]))

        if n_total:
            length_rows.append((n_total, n_correct / n_total))

    def topk(ranks, n_classes):
        r = np.array(ranks)
        return {f"top{k}": round(float((r < k).mean()), 4) for k in (1, 3, 5)} | {
            "median_rank": int(np.median(r)), "classes": n_classes,
            "chance_top1": round(1.0 / n_classes, 4), "n": len(r)}

    rf_m = np.array(rf_margins)
    rf_r = np.array(rf_ranks)
    gated = {}
    for q in (0.5, 0.75, 0.95):
        sel = rf_m >= np.quantile(rf_m, q)
        gated[f"most_confident_{int((1-q)*100)}pct"] = {
            "top1": round(float((rf_r[sel] < 1).mean()), 4), "n": int(sel.sum())}

    by_bucket = {}
    for lo, hi in BUCKETS:
        vals = [f for n, f in length_rows if lo <= n <= hi]
        label = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
        by_bucket[label] = {"utterances": len(vals),
                            "mean_correct_fraction": round(float(np.mean(vals)), 4)
                            if vals else None}

    res = {
        "utterances": used,
        "identification": {
            "random_forest": topk(rf_ranks, len(RF_PHONES)) | {"confidence_gated": gated},
            "dtw_templates": topk(dtw_ranks, len(BANK_PHONES)),
        },
        "verification_accuracy_by_utterance_length": by_bucket,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)

    print(json.dumps(res, indent=2))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()

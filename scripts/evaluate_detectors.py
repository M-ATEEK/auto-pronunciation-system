from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from src.classification.baseline_detector import BaselineDetector
from src.classification.phoneme_align import align, is_confusable
from src.classification.phoneme_recognizer import PhonemeRecognizer
from src.data.aligner import DictionaryAligner
from src.data.librispeech import (_iter_utterances, _load_audio,
                                  find_librispeech_root)
from src.features.extractor import (N_MFCC, SAMPLE_RATE, extract_features,
                                    extract_mfcc_sequence)
from src.utils.audio_io import trim_silence

SR = SAMPLE_RATE


SUBSTITUTIONS: dict[str, list[str]] = {
    "TH": ["S", "T", "F"], "DH": ["D", "Z"],
    "V": ["W", "B", "F"], "W": ["V"],
    "R": ["L"], "L": ["R"],
    "Z": ["S"], "S": ["SH"], "SH": ["S"], "ZH": ["SH"],
    "CH": ["SH"], "JH": ["Z", "CH"],
    "IY": ["IH"], "IH": ["IY"],
    "AE": ["EH"], "EH": ["AE"],
    "UW": ["UH"], "UH": ["UW"],
    "NG": ["N"], "N": ["NG"],
    "P": ["B"], "B": ["P"], "T": ["D"], "D": ["T"], "K": ["G"], "G": ["K"],
    "F": ["P"],
}


def _pick_substitute(phone: str, allowed: set[str], rng: random.Random) -> str | None:
   
    cands = [s for s in SUBSTITUTIONS.get(phone, [])
             if s in allowed and s != phone and not is_confusable(phone, s)]
    return rng.choice(cands) if cands else None


def build_items(corpus: str, n_utterances: int, skip: int, rate: float, seed: int,
                trained: set[str]):
   
    root = find_librispeech_root(corpus) or corpus
    utts = list(_iter_utterances(root))
    random.Random(0).shuffle(utts)         
    utts = utts[skip:]                    
    rng = random.Random(seed)
    aligner = DictionaryAligner()

    out = []
    for flac, text in utts:
        if len(out) >= n_utterances:
            break
        try:
            audio = trim_silence(_load_audio(flac))
        except Exception:
            continue
        segs = aligner.align(audio, SR, text)
        segs = [s for s in segs if len(s.audio) >= 320 and s.phone in trained]
        if len(segs) < 8:
            continue

        expected, labels = [], []
        for s in segs:
            sub = _pick_substitute(s.phone, trained, rng) if rng.random() < rate else None
            expected.append(sub or s.phone)
            labels.append(1 if sub else 0)
        if sum(labels) == 0:
            continue
        out.append({"audio": audio, "segs": segs, "expected": expected,
                    "labels": labels, "text": text})
    return out


def run_classical(items, detector) -> tuple[list[int], list[float]]:
  
    rf = detector._rf
    y_pred, y_score = [], []
    for it in items:
        seq = extract_mfcc_sequence(it["audio"], sr=SR)
        mean = seq.mean(axis=0).astype(np.float32) if len(seq) else np.zeros(N_MFCC, np.float32)
        for seg, exp in zip(it["segs"], it["expected"]):
            feat = extract_features(seg.audio, sr=SR, n_phones=1).copy()
            feat[:N_MFCC] -= mean
            p = float(rf.predict_proba_mispronounced(exp, feat))
            y_score.append(p)
            y_pred.append(int(p > detector._threshold))
    return y_pred, y_score


def run_neural(items) -> tuple[list[int], list[float]]:
   
    rec = PhonemeRecognizer.instance()
    y_pred, y_score = [], []
    for it in items:
        recognized = [r.phone for r in rec.recognize(it["audio"], sr=SR)]
        ops = align(it["expected"], recognized)
        # one verdict per expected position, in order
        verdict = {}
        for op in ops:
            if op.op == "ins" or op.expected_index is None:
                continue
            if op.op == "match":
                verdict[op.expected_index] = (0, 0.2)
            elif op.op == "sub":
                if is_confusable(it["expected"][op.expected_index], op.recognized):
                    verdict[op.expected_index] = (0, 0.55)
                else:
                    verdict[op.expected_index] = (1, 0.85)
            else:                                    # deletion
                verdict[op.expected_index] = (1, 0.9)
        for i in range(len(it["expected"])):
            p, s = verdict.get(i, (1, 0.9))          # unaligned == not produced
            y_pred.append(p)
            y_score.append(s)
    return y_pred, y_score


def metrics(y_true, y_pred, y_score) -> dict:
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(pr), 4),
        "recall": round(float(rc), 4),
        "f1": round(float(f1), 4),
        "auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def mcnemar(y_true, a_pred, b_pred) -> dict:
  
    from scipy.stats import binomtest
    a_ok = np.array(a_pred) == np.array(y_true)
    b_ok = np.array(b_pred) == np.array(y_true)
    n01 = int(np.sum(~a_ok & b_ok))    # classical wrong, neural right
    n10 = int(np.sum(a_ok & ~b_ok))    # classical right, neural wrong
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    return {"only_neural_correct": n01, "only_classical_correct": n10,
            "p_value": float(p)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/librispeech")
    ap.add_argument("--model", default="models/baseline_rf.pkl")
    ap.add_argument("--utterances", type=int, default=80)
    ap.add_argument("--skip", type=int, default=100, help="utterances used in training")
    ap.add_argument("--rate", type=float, default=0.25, help="fraction of phones perturbed")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="evaluation/detection_results.json")
    args = ap.parse_args()

    detector = BaselineDetector.load(args.model)
    trained = set(detector.trained_phones)

    print(f"Building test set from held-out utterances (skipping first {args.skip})…")
    items = build_items(args.corpus, args.utterances, args.skip, args.rate,
                        args.seed, trained)
    y_true = [l for it in items for l in it["labels"]]
    print(f"  {len(items)} utterances, {len(y_true)} phone items, "
          f"{sum(y_true)} induced errors ({100*sum(y_true)/len(y_true):.1f}%)")

    t0 = time.time()
    c_pred, c_score = run_classical(items, detector)
    t_classical = time.time() - t0
    print(f"  classical done in {t_classical:.1f}s")

    t0 = time.time()
    n_pred, n_score = run_neural(items)
    t_neural = time.time() - t0
    print(f"  neural done in {t_neural:.1f}s")

    res = {
        "protocol": {
            "utterances": len(items), "items": len(y_true),
            "induced_errors": int(sum(y_true)),
            "perturbation_rate": args.rate, "seed": args.seed,
            "held_out_from": args.skip,
        },
        "classical": metrics(y_true, c_pred, c_score),
        "neural": metrics(y_true, n_pred, n_score),
        "mcnemar": mcnemar(y_true, c_pred, n_pred),
        "runtime_seconds": {"classical": round(t_classical, 1),
                            "neural": round(t_neural, 1)},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)

    print(f"\n{'metric':<12}{'classical':>12}{'neural':>12}")
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        print(f"{k:<12}{res['classical'][k]:>12.4f}{res['neural'][k]:>12.4f}")
    print(f"\nMcNemar: {res['mcnemar']}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()

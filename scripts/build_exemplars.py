"""Build the native per-phone MFCC exemplar bank used by Stage-4 discovery.

Usage:
    .venv312/bin/python scripts/build_exemplars.py \
        --corpus data/librispeech --per-phone 5 --max-utterances 150 \
        --out models/native_exemplars.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.librispeech import build_native_exemplars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/librispeech")
    ap.add_argument("--per-phone", type=int, default=5)
    ap.add_argument("--max-utterances", type=int, default=150)
    ap.add_argument("--out", default="models/native_exemplars.pkl")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Building native exemplars from {args.corpus} ...", flush=True)
    bank = build_native_exemplars(
        args.corpus, per_phone=args.per_phone, max_utterances=args.max_utterances,
    )
    counts = {k: len(v) for k, v in sorted(bank.items())}
    print(f"  {len(bank)} phones; exemplars/phone min={min(counts.values())} "
          f"max={max(counts.values())}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(bank, fh)
    print(f"Saved -> {args.out}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

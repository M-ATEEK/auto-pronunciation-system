from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.aligner import DictionaryAligner
from src.data.librispeech import (_iter_utterances, _load_audio,
                                  find_librispeech_root)
from src.features.extractor import SAMPLE_RATE, extract_mfcc_sequence
from src.personalization.discovery import (NativeReferenceBank,
                                            discover_from_recognition)
from src.utils.audio_io import trim_silence

SR = SAMPLE_RATE

# Contrasts to simulate, each a documented first-language transfer pattern.
CASES = [("TH", "S"), ("V", "W"), ("R", "L"), ("Z", "S"), ("IY", "IH")]


def observations_for(utts, target: str, substitute: str, aligner, n_utt: int):
    from src.classification.phoneme_align import align
    from src.classification.phoneme_recognizer import PhonemeRecognizer
    from src.data.aligner import _phones_for_word

    recognizer = PhonemeRecognizer.instance()
    obs, perturbed, used = [], 0, 0
    for flac, text in utts:
        if used >= n_utt:
            break
        try:
            audio = trim_silence(_load_audio(flac))
        except Exception:
            continue

        expected = []
        for w in text.split():
            for ph in _phones_for_word(w.strip(".,!?;:'\"")):
                # the simulated learner has merged the two phones, so the
                # expectation is wrong at every occurrence of either one
                if ph == target:
                    expected.append(substitute)
                    perturbed += 1
                elif ph == substitute:
                    expected.append(target)
                    perturbed += 1
                else:
                    expected.append(ph)
        if not expected:
            continue

        recognized = recognizer.recognize(audio, sr=SR)
        ops = align(expected, [r.phone for r in recognized])
        used += 1
        for op in ops:
            if op.op == "ins" or op.expected_index is None:
                continue
            obs.append((op.expected, op.recognized if op.op != "del" else None))
    return obs, perturbed, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/librispeech")
    ap.add_argument("--bank", default="models/native_exemplars.pkl")
    ap.add_argument("--calibration-utterances", type=int, default=12)
    ap.add_argument("--skip", type=int, default=100)
    ap.add_argument("--out", default="evaluation/personalization_results.json")
    args = ap.parse_args()

    root = find_librispeech_root(args.corpus) or args.corpus
    utts = list(_iter_utterances(root))
    random.Random(0).shuffle(utts)
    utts = utts[args.skip:]
    aligner = DictionaryAligner()
    bank = NativeReferenceBank.load(args.bank)

    rows = []
    for target, substitute in CASES:
        obs, perturbed, used = observations_for(
            utts, target, substitute, aligner, args.calibration_utterances)
        if perturbed == 0:
            continue
        rules = discover_from_recognition(obs)
        found = [r.phone for r in rules]
        rows.append({
            "simulated_error": f"{target}->{substitute}",
            "expected_phone_perturbed": substitute,
            "calibration_utterances": used,
            "perturbed_occurrences": perturbed,
            "total_observations": len(obs),
            "discovered": substitute in found and target in found,
            "rules_raised": len(found),
            "false_rules": len([p for p in found if p not in (substitute, target)]),
        })
        print(f"{target}->{substitute}: perturbed {perturbed} occurrences, "
              f"discovered={rows[-1]['discovered']}, "
              f"rules={len(found)}, false={rows[-1]['false_rules']}")

    # Control: no perturbation at all. Any rule raised here is a false alarm.
    obs, _, used = observations_for(utts, "@@none@@", "", aligner,
                                    args.calibration_utterances)
    control_rules = discover_from_recognition(obs)
    print(f"control (no induced error): {len(control_rules)} rule(s) raised")

    hits = sum(r["discovered"] for r in rows)
    res = {
        "cases": rows,
        "hit_rate": round(hits / len(rows), 4) if rows else None,
        "mean_false_rules": round(
            sum(r["false_rules"] for r in rows) / len(rows), 2) if rows else None,
        "control_rules_raised": len(control_rules),
        "control_utterances": used,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nhit rate {res['hit_rate']}, mean false rules {res['mean_false_rules']}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()

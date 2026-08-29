set -euo pipefail

PY="${PY:-.venv312/bin/python}"
export PYTHONPATH="${PYTHONPATH:-.}"

for f in models/baseline_rf.pkl models/native_exemplars.pkl; do
    [ -f "$f" ] || { echo "missing $f -- see the prerequisites in this script's header"; exit 1; }
done

mkdir -p evaluation

echo "== 1/5  Detection, sparse errors (Tables 2, 4, 5) ============================"
$PY scripts/evaluate_detectors.py --utterances 80 --rate 0.25 \
    --out evaluation/detection_results.json

echo "== 2/5  Detection, dense errors (Tables 3, 4, 5) ============================="
$PY scripts/evaluate_detectors.py --utterances 80 --rate 1.0 \
    --out evaluation/detection_results_balanced.json

echo "== 3/5  Phone identification and utterance length (Tables 6, 7) =============="
$PY scripts/evaluate_identification.py --utterances 40 \
    --out evaluation/identification_results.json

echo "== 4/5  Personalisation discovery (Table 8) =================================="
$PY scripts/evaluate_personalization.py --calibration-utterances 12 \
    --out evaluation/personalization_results.json

echo "== 5/5  Input validity guards (Tables 9, 10) ================================="
$PY scripts/evaluate_guards.py --pairs 30 \
    --out evaluation/guard_results.json

echo
echo "All results written to evaluation/. Regenerate Figure 1 with:"
echo "    python3 ../thesis/figures/figure_6_1_detection_comparison.py"

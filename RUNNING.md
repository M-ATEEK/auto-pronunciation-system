# Running the CAPT System

A Computer-Assisted Pronunciation Training (CAPT) system: record a sentence,
get per-sound pronunciation feedback. Built as a FastAPI backend + React
frontend, with two selectable detectors (classical Random Forest, neural
wav2vec2), DTW-based personalisation, and multi-modal feedback.

## Requirements

- Python 3.12
- Node 18 (`nvm use 18`)
- ffmpeg (audio decoding)
- macOS (the feedback TTS reference audio uses the `say` command)

## 1. Backend setup

```bash
cd implementation
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

`torch`/`transformers` (needed for the neural detector) are heavy — the first
`/api/recognize` or `/api/analyze?detector=neural` call downloads the
wav2vec2 model (~1.2 GB) automatically.

### Download LibriSpeech (dev-clean)

The classical detector, DTW reference bank, and calibration all need real
native speech. Point `data/librispeech` at an extracted LibriSpeech
`dev-clean` tree (a symlink works fine).

### Train the models (one-time, before first use)

```bash
python scripts/train_baseline.py     # -> models/baseline_rf.pkl  (~7 min)
python scripts/build_exemplars.py    # -> models/native_exemplars.pkl  (~5 s)
```

Both scripts accept `--corpus`, and have sensible defaults — see `--help`.

### Run the API

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```

```bash
curl http://127.0.0.1:8001/api/health
# {"status":"ok","version":"0.1.0"}
```

## 2. Frontend setup

```bash
cd implementation/frontend
nvm use 18
npm install
npm run dev
```

Open **http://localhost:5173** — the Vite dev server proxies `/api/*` to
the backend on port 8001.

## 3. Using it

1. Type the sentence you're about to say.
2. Optionally click **Listen to correct pronunciation** to hear it first.
3. **Record**, say it, **Stop**.
4. Pick a detector (**Classical** or **Neural**) and **Analyze Pronunciation**.
5. Read the per-sound table and feedback cards: articulatory hint, your own
   audio vs. the native reference, and a waveform/DTW comparison for
   mispronounced sounds.
6. Under **Calibrate Your Profile**, record the two calibration sentences and
   **Run Calibration** to discover systematic error patterns — future
   analyses for that learner will flag matching sounds as "Systematic".
7. **View My Profile** for the full per-phone history dashboard.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/health` | Liveness check |
| GET  | `/api/phones?text=` | ARPABET phone sequence for text (Stage 1) |
| POST | `/api/features` | 17-dim acoustic feature vector from a clip (Stage 2) |
| GET  | `/api/data/stats` | LibriSpeech corpus overview |
| POST | `/api/analyze?detector=baseline\|neural` | Per-phone mispronunciation verdicts + feedback (Stages 3-5) |
| GET  | `/api/profile/{learner_id}` | Learner's accumulated error profile |
| POST | `/api/calibrate` | Discover systematic errors from a few sentences (Stage 4) |
| POST | `/api/recognize` | Neural phoneme recognition, no transcript needed |
| GET  | `/api/reference?text=` | Native-quality TTS reference audio (Stage 5) |

## Tests

```bash
# Backend
source .venv312/bin/activate
pytest tests/ -q

# Frontend
cd frontend
npm run test
```

## Project layout

```
src/
  api/              FastAPI application and endpoints
  data/              Phone alignment + LibriSpeech loader     — Stage 1
  features/          Acoustic feature extraction               — Stage 2
  classification/     Random Forest, wav2vec2, alignment        — Stage 3
  personalization/    DTW, learner profiles, error discovery    — Stage 4
  feedback/           Hints, TTS, feedback assembly              — Stage 5
  pipeline/           CAPTSystem orchestrator (dual-detector dispatch)
  utils/              Shared utilities (logging, phoneme constants, audio decode)
scripts/            One-time training/build scripts
tests/              Backend unit tests (pytest)
frontend/           React + Vite UI
  src/components/    AudioRecorder, TranscriptInput, FeedbackCard,
                     WaveformView, ProfileDashboard, CalibrationPanel
  src/pages/         AnalysisPage
```

## Notes

- **Silence trimming is part of the contract.** `src/utils/audio_io.trim_silence`
  runs on *both* the LibriSpeech training audio and the learner's recording.
  The dictionary aligner splits an utterance across its phones by proportion of
  total duration, so leading/trailing silence shifts every phone boundary — on a
  short recording it can consume an entire phone. If you change the trimming,
  **retrain** (`train_baseline.py` and `build_exemplars.py`) so training and
  inference still see the same kind of audio.
- `models/*.pkl` and `models/profiles/` are generated artifacts (gitignored)
  — regenerate with the training scripts above.
- `data/librispeech` is expected to be a symlink or download (gitignored).
- Learner profiles persist to `models/profiles/<learner_id>.json` across
  server restarts.

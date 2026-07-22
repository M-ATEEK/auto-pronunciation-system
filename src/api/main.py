from __future__ import annotations

import logging
import os
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from src.classification.baseline_detector import BaselineDetector
from src.data.aligner import phones_for_text
from src.data.librispeech import corpus_stats
from src.features.extractor import SAMPLE_RATE, extract_features
from src.utils.audio_io import decode_audio
from src.utils.logger import get_logger

_LOG = get_logger(__name__, level=logging.INFO)

VERSION = "0.1.0"

_CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "librispeech")
_BASELINE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "baseline_rf.pkl")

app = FastAPI(
    title="CAPT",
    version=VERSION,
)


@app.get("/api/health")
async def health() -> dict:
    """Liveness check."""
    return {"status": "ok", "version": VERSION}


@app.get("/api/phones")
async def phones(text: str = Query(..., description="Text to convert to ARPABET phones")) -> dict:
    """Return the ARPABET phone sequence for the given text (CMU dictionary)."""
    return {"text": text, "phones": phones_for_text(text)}


_FEATURE_LABELS = [f"mfcc{i}" for i in range(13)] + ["f0", "rms", "speech_rate", "pause_ratio"]


@app.post("/api/features")
async def features(audio: UploadFile = File(...)) -> dict:
    """Extracting the 17 dimensional acoustic feature vector from an uploaded clip."""
    raw = await audio.read()
    waveform = decode_audio(raw)
    vector = extract_features(waveform, sr=SAMPLE_RATE)
    return {
        "duration_s": round(len(waveform) / SAMPLE_RATE, 3),
        "feature_dim": int(len(vector)),
        "features": {name: round(float(v), 4) for name, v in zip(_FEATURE_LABELS, vector)},
    }


@lru_cache(maxsize=1)
def _cached_corpus_stats() -> dict:
    return corpus_stats(_CORPUS_DIR)


@app.get("/api/data/stats")
async def data_stats() -> dict:
    return _cached_corpus_stats()


@lru_cache(maxsize=1)
def _cached_baseline_detector() -> BaselineDetector:
    if not os.path.exists(_BASELINE_MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail="Baseline model not trained yet  run scripts/train_baseline.py first.",
        )
    return BaselineDetector.load(_BASELINE_MODEL_PATH)


@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...), transcript: str = Form(...)) -> dict:
    """Classical (Random Forest) per-phone mispronunciation."""
    detector = _cached_baseline_detector()
    raw = await audio.read()
    waveform = decode_audio(raw)
    try:
        results = detector.detect(waveform, transcript, sr=SAMPLE_RATE)
    except Exception:
        _LOG.exception(
            "analyze failed: transcript=%r audio_bytes=%d waveform_samples=%d",
            transcript, len(raw), len(waveform),
        )
        raise HTTPException(status_code=500, detail="Analysis failed — see server log.")
    return {
        "transcript": transcript,
        "phones": [
            {
                "phone": r.phone,
                "word": r.word,
                "prob_mispronounced": r.prob_mispronounced,
                "is_mispronounced": r.is_mispronounced,
            }
            for r in results
        ],
    }

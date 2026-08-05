from __future__ import annotations

import logging
import os
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile

from src.classification.phoneme_recognizer import PhonemeRecognizer
from src.data.aligner import phones_for_text
from src.data.librispeech import corpus_stats
from src.feedback.tts import cache_key as tts_cache_key
from src.feedback.tts import synthesize_wav_bytes
from src.features.extractor import SAMPLE_RATE, extract_features
from src.pipeline.capt_system import CAPTSystem
from src.utils.audio_io import decode_audio
from src.utils.logger import get_logger

_LOG = get_logger(__name__, level=logging.INFO)

VERSION = "0.1.0"

_CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "librispeech")

_capt = CAPTSystem()

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


@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...), transcript: str = Form(...),
                   learner_id: str = Form("learner-001"),
                   detector: str = Query("baseline", description="'baseline' or 'neural'")) -> dict:
    """Per-phone mispronunciation detection. detector='baseline' (Random Forest,
    default) or 'neural' (wav2vec2 recognise + align) -- both return the same shape.
    """
    raw = await audio.read()
    waveform = decode_audio(raw)
    try:
        result = _capt.analyze(waveform, transcript, learner_id, detector=detector)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        _LOG.exception(
            "analyze failed: detector=%r transcript=%r audio_bytes=%d waveform_samples=%d",
            detector, transcript, len(raw), len(waveform),
        )
        raise HTTPException(status_code=500, detail="Analysis failed — see server log.")

    return {
        "transcript": transcript,
        "learner_id": learner_id,
        "detector": detector,
        "phones": result["phones"],
        "match_confidence": result["match_confidence"],
        "match_warning": result["match_warning"],
        "articulation_rate": result["articulation_rate"],
    }


@app.get("/api/reference")
async def reference(text: str = Query(..., description="Text to synthesise as native reference audio")):
    """Native-quality TTS reference pronunciation of ``text``."""
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Query parameter 'text' is required.")
    if len(text) > 300:
        raise HTTPException(status_code=400, detail="Text too long (max 300 chars).")
    wav = synthesize_wav_bytes(text)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "Cache-Control": "public, max-age=86400",
            "ETag": f'"{tts_cache_key(text)}"',
        },
    )


@app.get("/api/profile/{learner_id}")
async def get_profile(learner_id: str) -> dict:
    """Return the learner's accumulated per-phone DTW error profile."""
    return _capt.profile_dict(learner_id)


@app.post("/api/calibrate")
async def calibrate(audio: list[UploadFile] = File(...),
                     transcripts: list[str] = Form(...),
                     learner_id: str = Form("learner-001")) -> dict:
    """Discover the learner's systematic error patterns from a few read-aloud
    sentences (Stage 4). Stores the discovered rules in the learner's profile,
    .
    """
    if len(audio) != len(transcripts):
        raise HTTPException(status_code=400,
                             detail="audio and transcripts must have the same length")

    sessions = []
    for f, t in zip(audio, transcripts):
        raw = await f.read()
        sessions.append((decode_audio(raw), t))

    try:
        rules = _capt.calibrate(learner_id, sessions)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        _LOG.exception("calibrate failed: learner_id=%r sessions=%d", learner_id, len(sessions))
        raise HTTPException(status_code=500, detail="Calibration failed  see server log.")

    return {"learner_id": learner_id, "rules": rules}


@app.post("/api/recognize")
async def recognize(audio: UploadFile = File(...)) -> dict:
    """Neural (wav2vec2) phoneme recognition -- what was actually said, no transcript needed."""
    raw = await audio.read()
    waveform = decode_audio(raw)
    try:
        recognizer = PhonemeRecognizer.instance()
        recognized = recognizer.recognize(waveform, sr=SAMPLE_RATE)
    except Exception:
        _LOG.exception("recognize failed: audio_bytes=%d waveform_samples=%d", len(raw), len(waveform))
        raise HTTPException(status_code=500, detail="Recognition failed — see server log.")

    return {
        "duration_s": round(len(waveform) / SAMPLE_RATE, 3),
        "phones": [
            {
                "phone": p.phone,
                "start_s": round(p.start_sample / SAMPLE_RATE, 3),
                "end_s": round(p.end_sample / SAMPLE_RATE, 3),
                "confidence": round(p.confidence, 4),
            }
            for p in recognized
        ],
    }

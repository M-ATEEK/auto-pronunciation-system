from __future__ import annotations

import logging

from fastapi import FastAPI, Query

from src.data.aligner import phones_for_text
from src.utils.logger import get_logger

_LOG = get_logger(__name__, level=logging.INFO)

VERSION = "0.1.0"

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

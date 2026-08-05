"""Articulatory text hints for all 44 ARPABET phonemes (Stage 5)"""

from src.utils.phoneme_constants import ARTICULATORY_HINTS, ENGLISH_PHONEMES

_DEFAULT_HINT = (
    "Focus on the placement of your tongue, lips, and airflow for this sound. "
    "Listen to the native reference carefully."
)


def get_hint(phone: str) -> str:
    """Return the articulatory hint for a phoneme.

    Falls back to a generic instruction if the phoneme is not in the table.
    """
    return ARTICULATORY_HINTS.get(phone, _DEFAULT_HINT)


def all_phones_with_hints() -> list[str]:
    """Return all phonemes that have explicit articulatory hints."""
    return [p for p in ENGLISH_PHONEMES if p in ARTICULATORY_HINTS]

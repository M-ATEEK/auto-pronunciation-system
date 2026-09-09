"""Articulatory description of each phone, and the difference between two (Stage 5).

Gives the visual feedback modality a data source
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Manner categories, used for manner-specific advice.
STOP = "stop"
FRICATIVE = "fricative"
AFFRICATE = "affricate"
NASAL = "nasal"
APPROXIMANT = "approximant"
VOWEL = "vowel"

BILABIAL = "bilabial"
LABIODENTAL = "labiodental"
DENTAL = "dental"
ALVEOLAR = "alveolar"
LATERAL = "lateral"
POSTALVEOLAR = "postalveolar"
PALATAL = "palatal"
VELAR = "velar"
LABIOVELAR = "labiovelar"
RHOTIC = "rhotic"
GLOTTAL = "glottal"


@dataclass(frozen=True)
class Articulation:
    """Canonical articulation of one phone. """

    phone: str
    lip_rounding: float     # 0 spread .. 0.5 neutral .. 1 fully rounded
    jaw_openness: float     # 0 closed .. 1 wide open
    tongue_front: float     # 0 back .. 1 front  (body of the tongue)
    tongue_height: float    # 0 low .. 1 high (close to palate)
    manner: str
    voiced: bool
    # The tongue TIP is a separate articulator from the body, and it is what
    # separates several high-value pairs the body alone cannot: /th/ (tip
    # between the teeth) from /s/ (tip behind the ridge).
    tongue_tip: float = 0.30   # 0 low/retracted .. 0.7 at the ridge .. 1 between teeth
    # Tense vs lax is what distinguishes the classic sheep/ship pair; it is not
    # a position at all, so no positional parameter can express it.
    tense: bool = False
    # Place of articulation; empty for vowels, which are described by height,
    # backness and rounding instead.
    place: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _v(phone, rounding, jaw, front, height, tense=False):
    return Articulation(phone, rounding, jaw, front, height, VOWEL, True,
                        tongue_tip=0.30, tense=tense)


def _c(phone, manner, place, rounding, jaw, front, height, voiced, tip=0.30):
    return Articulation(phone, rounding, jaw, front, height, manner, voiced,
                        tongue_tip=tip, tense=False, place=place)


# --- Vowels: height/backness/rounding from the standard vowel chart ---------
_VOWELS: dict[str, Articulation] = {
    "IY": _v("IY", 0.10, 0.15, 1.00, 1.00, tense=True),   # beet  -- high front unrounded
    "IH": _v("IH", 0.10, 0.30, 0.85, 0.80),   # bit
    "EY": _v("EY", 0.10, 0.30, 0.90, 0.75, tense=True),   # bait
    "EH": _v("EH", 0.10, 0.50, 0.80, 0.50),   # bet
    "AE": _v("AE", 0.10, 0.80, 0.75, 0.20),   # bat   -- low front
    "AA": _v("AA", 0.15, 0.95, 0.15, 0.05, tense=True),   # father-- low back
    "AO": _v("AO", 0.70, 0.75, 0.15, 0.30, tense=True),   # bought-- low-mid back rounded
    "OW": _v("OW", 0.80, 0.45, 0.20, 0.60, tense=True),   # boat
    "UH": _v("UH", 0.70, 0.30, 0.25, 0.75),   # book
    "UW": _v("UW", 0.95, 0.20, 0.15, 1.00, tense=True),   # boot  -- high back rounded
    "AH": _v("AH", 0.15, 0.55, 0.45, 0.45),   # but   -- mid central
    "ER": _v("ER", 0.35, 0.45, 0.45, 0.50),   # bird  -- rhotic mid central
    # Diphthongs are described by their STARTING position; the glide is shown
    # by the diagram's animation rather than a second static value.
    "AY": _v("AY", 0.10, 0.80, 0.50, 0.25),   # bite  (a -> i)
    "AW": _v("AW", 0.30, 0.85, 0.40, 0.25),   # bout  (a -> u)
    "OY": _v("OY", 0.60, 0.65, 0.30, 0.40),   # boy   (o -> i)
    # Reduced/allophonic vowels sometimes present in transcription sets.
    "AX": _v("AX", 0.15, 0.50, 0.45, 0.45),   # schwa
    "IX": _v("IX", 0.10, 0.35, 0.70, 0.70),   # reduced high central
    "UX": _v("UX", 0.70, 0.30, 0.45, 0.85),   # fronted /u/
    "AXR": _v("AXR", 0.35, 0.45, 0.45, 0.50),  # r-coloured schwa
}

# --- Consonants: place determines tongue/lips, manner determines the rest ---
_CONSONANTS: dict[str, Articulation] = {
    # Bilabial -- lips together, tongue not the active articulator.
    "P": _c("P", STOP, BILABIAL, 0.30, 0.02, 0.45, 0.40, False),
    "B": _c("B", STOP, BILABIAL, 0.30, 0.02, 0.45, 0.40, True),
    "M": _c("M", NASAL, BILABIAL, 0.30, 0.02, 0.45, 0.40, True),
    # Labiodental -- lower lip against upper teeth.
    "F": _c("F", FRICATIVE, LABIODENTAL, 0.20, 0.10, 0.45, 0.40, False),
    "V": _c("V", FRICATIVE, LABIODENTAL, 0.20, 0.10, 0.45, 0.40, True),
    # Dental -- tongue tip at/between the teeth.
    "TH": _c("TH", FRICATIVE, DENTAL, 0.15, 0.18, 1.00, 0.70, False, tip=1.00),
    "DH": _c("DH", FRICATIVE, DENTAL, 0.15, 0.18, 1.00, 0.70, True, tip=1.00),
    # Alveolar -- tongue tip at the ridge behind the teeth.
    "T": _c("T", STOP, ALVEOLAR, 0.15, 0.15, 0.90, 0.85, False, tip=0.70),
    "D": _c("D", STOP, ALVEOLAR, 0.15, 0.15, 0.90, 0.85, True, tip=0.70),
    "N": _c("N", NASAL, ALVEOLAR, 0.15, 0.15, 0.90, 0.85, True, tip=0.70),
    "S": _c("S", FRICATIVE, ALVEOLAR, 0.15, 0.12, 0.90, 0.88, False, tip=0.65),
    "Z": _c("Z", FRICATIVE, ALVEOLAR, 0.15, 0.12, 0.90, 0.88, True, tip=0.65),
    "L": _c("L", APPROXIMANT, LATERAL, 0.15, 0.25, 0.85, 0.75, True, tip=0.80),
    # Post-alveolar -- slightly further back, with a little lip rounding.
    "SH": _c("SH", FRICATIVE, POSTALVEOLAR, 0.45, 0.20, 0.70, 0.80, False),
    "ZH": _c("ZH", FRICATIVE, POSTALVEOLAR, 0.45, 0.20, 0.70, 0.80, True),
    "CH": _c("CH", AFFRICATE, POSTALVEOLAR, 0.45, 0.18, 0.72, 0.82, False),
    "JH": _c("JH", AFFRICATE, POSTALVEOLAR, 0.45, 0.18, 0.72, 0.82, True),
    "R": _c("R", APPROXIMANT, RHOTIC, 0.40, 0.28, 0.50, 0.60, True, tip=0.45),
    # Palatal.
    "Y": _c("Y", APPROXIMANT, PALATAL, 0.10, 0.18, 1.00, 0.95, True),
    # Velar -- tongue BACK raised to the soft palate.
    "K": _c("K", STOP, VELAR, 0.20, 0.18, 0.10, 0.90, False),
    "G": _c("G", STOP, VELAR, 0.20, 0.18, 0.10, 0.90, True),
    "NG": _c("NG", NASAL, VELAR, 0.20, 0.18, 0.10, 0.90, True),
    "ENG": _c("ENG", NASAL, VELAR, 0.20, 0.18, 0.10, 0.90, True),
    # Labio-velar -- rounded lips AND raised back tongue.
    "W": _c("W", APPROXIMANT, LABIOVELAR, 0.90, 0.18, 0.10, 0.90, True),
    # Glottal -- no oral constriction; mouth takes the shape of what follows.
    "HH": _c("HH", FRICATIVE, GLOTTAL, 0.30, 0.40, 0.45, 0.45, False),
}

_TABLE: dict[str, Articulation] = {**_VOWELS, **_CONSONANTS}

# A parameter must differ by more than this before it is worth mentioning.
# Below it, the two articulations are close enough that an instruction would
# be noise rather than guidance.
_MIN_DELTA = 0.22


def articulation_for(phone: str) -> Articulation | None:
    """Canonical articulation of ``phone``, or None if it is not described."""
    return _TABLE.get(phone)


def describe_difference(expected: str, produced: str) -> list[str]:
    """Concrete instructions to move from ``produced`` towards ``expected``.

    Ordered most- to least-important, and capped so the learner gets a short
    actionable list rather than an exhaustive phonetic analysis.
    """
    target = articulation_for(expected)
    heard = articulation_for(produced)
    if target is None or heard is None or expected == produced:
        return []

    scored: list[tuple[float, str]] = []

    d = target.lip_rounding - heard.lip_rounding
    if abs(d) > _MIN_DELTA:
        scored.append((abs(d), "Round your lips more" if d > 0
                       else "Spread your lips instead of rounding them"))

    d = target.jaw_openness - heard.jaw_openness
    if abs(d) > _MIN_DELTA:
        scored.append((abs(d), "Open your mouth wider" if d > 0
                       else "Close your mouth more"))

    d = target.tongue_front - heard.tongue_front
    if abs(d) > _MIN_DELTA:
        scored.append((abs(d), "Move your tongue forward" if d > 0
                       else "Move your tongue further back"))

    d = target.tongue_height - heard.tongue_height
    if abs(d) > _MIN_DELTA:
        scored.append((abs(d), "Raise your tongue towards the roof of your mouth"
                       if d > 0 else "Lower your tongue"))

    # Tip advice must name the TARGET's place of articulation, not merely the
    d = target.tongue_tip - heard.tongue_tip
    if abs(d) > _MIN_DELTA:
        if target.tongue_tip >= 0.90:
            tip_advice = "Put your tongue tip between your teeth"
        elif target.tongue_tip >= 0.60:
            tip_advice = "Touch your tongue tip to the ridge behind your upper teeth"
        else:
            tip_advice = ("Keep your tongue tip down, it is not used for this sound")
        scored.append((abs(d) + 0.5, tip_advice))  # tip errors are highly audible

    scored.sort(key=lambda x: -x[0])
    tips = [text for _, text in scored]

    # Voicing and manner are categorical, so they are not scored by magnitude;
    # voicing in particular changes the word ("sip" vs "zip") and is listed first.
    if target.voiced != heard.voiced:
        tips.insert(0, "Use your voice so your vocal cords vibrate"
                    if target.voiced else
                    "Whisper this sound without using your voice")

    if (target.manner == VOWEL and heard.manner == VOWEL
            and target.tense != heard.tense):
        tips.append("Hold the vowel longer and tenser" if target.tense
                    else "Make the vowel shorter and more relaxed")

    if target.manner != heard.manner:
        tips.append(_MANNER_ADVICE.get(target.manner, ""))

    return [t for t in tips if t][:4]


_PLACE_ADVICE: dict[str, str] = {
    # Deliberately says nothing about releasing: /m/ holds the closure while the
    # air leaves through the nose. The manner line supplies the release.
    BILABIAL: "Press both lips together",
    LABIODENTAL: "Rest your top teeth lightly on your bottom lip",
    DENTAL: "Put your tongue tip between your teeth",
    # "at", not "on": /t/ needs full contact but /s/ needs a narrow gap. The
    # manner line is what distinguishes them.
    ALVEOLAR: "Put your tongue tip at the ridge just behind your upper teeth",
    LATERAL: "Touch your tongue tip to the ridge behind your upper teeth and let "
             "the air flow around the sides of your tongue",
    POSTALVEOLAR: "Pull your tongue tip back a little, just behind that ridge",
    PALATAL: "Raise the middle of your tongue towards the roof of your mouth",
    VELAR: "Raise the BACK of your tongue against the soft palate, keeping the tip down",
    LABIOVELAR: "Round your lips tightly and raise the back of your tongue",
    RHOTIC: "Curl your tongue tip up and back without letting it touch the roof",
    GLOTTAL: "Just push air out from your throat, keeping the mouth relaxed",
}

# /h/ is classed as a fricative, but the friction is at the glottis and there is
_NO_MANNER_ADVICE = frozenset({GLOTTAL})


def describe_target(phone: str) -> list[str]:
    """How to produce ``phone`` correctly, with no reference to what was said."""
    a = articulation_for(phone)
    if a is None:
        return []

    tips: list[str] = []
    if a.manner == VOWEL:
        # Vowels have no constriction to place; they are defined by the shape of
        # the whole cavity, so describe jaw, tongue body and lips instead.
        tips.append("Open your mouth wide" if a.jaw_openness >= 0.70
                    else "Keep your mouth only slightly open" if a.jaw_openness <= 0.25
                    else "Open your mouth about halfway")
        if a.tongue_front >= 0.70:
            body = "Arch your tongue towards the front of your mouth"
        elif a.tongue_front <= 0.30:
            body = "Pull your tongue back"
        else:
            body = "Keep your tongue in a neutral, central position"
        if a.tongue_height >= 0.75:
            body += ", high and close to the roof"
        elif a.tongue_height <= 0.30:
            body += ", held low"
        tips.append(body)
        if a.lip_rounding >= 0.60:
            tips.append("Round your lips")
        elif a.lip_rounding <= 0.20:
            tips.append("Spread your lips instead of rounding them")
        tips.append("Hold it long and tense" if a.tense
                    else "Keep it short and relaxed")
    else:
        tips.append(_PLACE_ADVICE.get(a.place, ""))
        if a.place not in _NO_MANNER_ADVICE:
            tips.append(_MANNER_ADVICE.get(a.manner, ""))
        tips.append("Use your voice so your vocal cords vibrate" if a.voiced
                    else "Do not use your voice, this sound is just air")

    return [t for t in tips if t][:4]


_MANNER_ADVICE: dict[str, str] = {
    STOP: "Stop the air completely, then release it in one burst",
    FRICATIVE: "Let the air hiss through a narrow gap without blocking it fully",
    AFFRICATE: "Start by stopping the air, then release it into a hiss",
    NASAL: "Let the air out through your nose, not your mouth",
    APPROXIMANT: "Let the air flow freely without creating friction",
    VOWEL: "Keep the air flowing smoothly with an open mouth",
}

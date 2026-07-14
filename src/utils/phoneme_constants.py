"""English phoneme inventory and articulatory hint lookup table."""

ENGLISH_PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
    'AX', 'IX', 'UX', 'AXR', 'ENG',
]

VOICED_PHONEMES = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'AX', 'AXR',
    'B', 'D', 'DH',
    'EH', 'ER', 'ENG', 'EY',
    'G',
    'IH', 'IX', 'IY',
    'JH',
    'L',
    'M',
    'N', 'NG',
    'OW', 'OY',
    'R',
    'UH', 'UW', 'UX',
    'V',
    'W',
    'Y',
    'Z', 'ZH',
}

ARTICULATORY_HINTS: dict[str, str] = {
    'AA': 'Low back vowel /ɑː/: mouth wide open, tongue low and back. Like "father".',
    'AE': (
        'Open front vowel /æ/: open your mouth wide, tongue flat and low. '
        'Often confused with /ɛ/ — lower your jaw further. Like "cat".'
    ),
    'AH': 'Mid central vowel /ʌ/: tongue in neutral position, mouth half open. Like "cup".',
    'AO': 'Low back rounded vowel /ɔː/: rounded lips, tongue low and back. Like "thought".',
    'AW': 'Diphthong /aʊ/: start with open mouth (AA) and glide to rounded lips (UH). Like "cow".',
    'AY': 'Diphthong /aɪ/: start with open mouth (AA) and glide to high front (IY). Like "my".',
    'B': (
        'Voiced bilabial stop /b/: press both lips together, add voice — '
        'feel vibration in throat. Like "bat".'
    ),
    'CH': 'Affricate /tʃ/: combine /t/ then /ʃ/ — tongue tip on alveolar ridge, release with friction. Like "cheese".',
    'D': 'Voiced alveolar stop /d/: tongue tip touches ridge behind upper teeth, add voice. Like "dog".',
    'DH': 'Voiced dental fricative /ð/: tongue tip lightly between teeth, blow air and add voice. Like "the".',
    'EH': 'Mid front vowel /ɛ/: tongue mid-high, lips slightly spread. Like "bed".',
    'ER': 'Rhotacised mid vowel /ɜːr/: curl tongue tip back in neutral position. Like "bird".',
    'EY': 'Diphthong /eɪ/: start with mid front (EH) and glide to high front (IY). Like "face".',
    'F': 'Labiodental fricative /f/: upper teeth on lower lip, blow air. Like "fan".',
    'G': 'Voiced velar stop /ɡ/: back of tongue contacts velum (soft palate), add voice. Like "go".',
    'HH': 'Glottal fricative /h/: open glottis, breathe out with no constriction. Like "hat".',
    'IH': 'Near-close front vowel /ɪ/: tongue high and front but slightly lower than IY. Like "bit".',
    'IY': 'Close front vowel /iː/: tongue high and front, lips spread. Like "beat".',
    'JH': 'Voiced affricate /dʒ/: combine /d/ then /ʒ/ — tongue tip to ridge, release with voiced friction. Like "jump".',
    'K': 'Voiceless velar stop /k/: back of tongue to soft palate, release burst. Like "cat".',
    'L': 'Lateral approximant /l/: tongue tip to alveolar ridge, air flows around sides. Like "let".',
    'M': 'Bilabial nasal /m/: lips together, release air through nose with voice. Like "man".',
    'N': 'Alveolar nasal /n/: tongue tip to alveolar ridge, air through nose with voice. Like "not".',
    'NG': 'Velar nasal /ŋ/: back tongue to soft palate, air through nose — no release. Like "sing".',
    'OW': 'Diphthong /oʊ/: start with mid-back (AO) and glide to rounded (UH). Like "go".',
    'OY': 'Diphthong /ɔɪ/: start with rounded (AO) and glide to high front (IY). Like "boy".',
    'P': (
        'Bilabial stop /p/: press both lips together, release with a burst of air '
        '(aspirated at word start). Like "pat".'
    ),
    'R': 'American /r/: curl tongue tip back (retroflex) or bunch it — do not trill. Like "red".',
    'S': 'Alveolar fricative /s/: tongue near alveolar ridge, blow air through narrow channel. Like "sit".',
    'SH': 'Palato-alveolar fricative /ʃ/: tongue slightly back from /s/, lips slightly rounded. Like "ship".',
    'T': 'Voiceless alveolar stop /t/: tongue tip to alveolar ridge, release burst (aspirated). Like "top".',
    'TH': 'Dental fricative /θ/: place tongue tip lightly between teeth, blow air (no voice). Like "think".',
    'UH': 'Near-close back rounded vowel /ʊ/: lips rounded, tongue high and back but lower than UW. Like "foot".',
    'UW': 'Close back rounded vowel /uː/: lips tightly rounded, tongue high and back. Like "boot".',
    'V': 'Voiced labiodental fricative /v/: upper teeth on lower lip, blow air with voice. Like "van".',
    'W': 'Labial-velar approximant /w/: round lips, back tongue raised — glide into vowel. Like "wet".',
    'Y': 'Palatal approximant /j/: tongue toward hard palate, glide into vowel. Like "yes".',
    'Z': 'Voiced alveolar fricative /z/: same as /s/ but add voice. Like "zoo".',
    'ZH': 'Voiced palato-alveolar fricative /ʒ/: same as /ʃ/ but add voice. Like "measure".',
    'AX': 'Reduced central vowel /ə/ (schwa): unstressed, tongue in neutral position. Like "about".',
    'IX': 'Reduced high front vowel /ɨ/: unstressed, between IH and UH. Like "roses" final syllable.',
    'UX': 'Reduced high back vowel: unstressed back vowel. Very brief and centralised.',
    'AXR': 'Rhotacised schwa /ɚ/: unstressed rhotacised vowel, curl tongue slightly. Like "butter".',
    'ENG': 'Syllabic velar nasal /ŋ̍/: same as NG but syllabic — forms its own syllable.',
}

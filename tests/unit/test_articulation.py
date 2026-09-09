"""Articulatory descriptions and the instructions derived from them."""

from __future__ import annotations

from src.feedback.articulation import (
    VOWEL,
    articulation_for,
    describe_difference,
    describe_target,
)
from src.utils.phoneme_constants import ENGLISH_PHONEMES


class TestCoverage:
    def test_every_phone_is_described(self):
        missing = [p for p in ENGLISH_PHONEMES if articulation_for(p) is None]
        assert missing == []

    def test_unknown_phone_returns_none(self):
        assert articulation_for("QQ") is None

    def test_parameters_are_in_range(self):
        for p in ENGLISH_PHONEMES:
            a = articulation_for(p)
            for name in ("lip_rounding", "jaw_openness", "tongue_front",
                         "tongue_height", "tongue_tip"):
                v = getattr(a, name)
                assert 0.0 <= v <= 1.0, f"{p}.{name} out of range: {v}"


class TestKnownPhonetics:
    def test_bilabial_stops_close_the_lips(self):
        for p in ("P", "B", "M"):
            assert articulation_for(p).jaw_openness < 0.1

    def test_rounded_vowels_are_rounded(self):
        assert articulation_for("UW").lip_rounding > 0.8
        assert articulation_for("IY").lip_rounding < 0.3

    def test_velars_have_back_tongue(self):
        for p in ("K", "G", "NG"):
            assert articulation_for(p).tongue_front < 0.2

    def test_dentals_have_the_most_forward_tip(self):
        assert articulation_for("TH").tongue_tip > articulation_for("S").tongue_tip
        assert articulation_for("TH").tongue_tip > articulation_for("T").tongue_tip

    def test_voicing_pairs_differ_only_in_voicing(self):
        for voiceless, voiced in (("P", "B"), ("T", "D"), ("S", "Z"), ("F", "V")):
            a, b = articulation_for(voiceless), articulation_for(voiced)
            assert a.voiced is False and b.voiced is True
            assert a.manner == b.manner
            assert a.tongue_front == b.tongue_front


class TestDifferences:
    def test_identical_phones_need_no_advice(self):
        assert describe_difference("S", "S") == []

    def test_unknown_phone_yields_no_advice(self):
        assert describe_difference("S", "QQ") == []
        assert describe_difference("QQ", "S") == []

    def test_voicing_error_is_reported_first(self):
        tips = describe_difference("B", "P")
        assert tips and "voice" in tips[0].lower()

    def test_rounding_error_is_reported(self):
        tips = describe_difference("UW", "IY")
        assert any("round" in t.lower() for t in tips)

    def test_sheep_ship_contrast_is_not_silent(self):
        """IY vs IH differ by tenseness, not position -- the canonical L2 pair."""
        assert describe_difference("IY", "IH")

    def test_think_sink_contrast_is_not_silent(self):
        assert describe_difference("TH", "S")

    def test_dental_target_says_between_the_teeth(self):
        tips = describe_difference("TH", "S")
        assert any("between your teeth" in t for t in tips)

    def test_alveolar_target_does_not_say_between_the_teeth(self):
        """/n/ is at the ridge, not between the teeth -- naming the direction
        of change rather than the target's place gave wrong anatomy."""
        tips = describe_difference("N", "M")
        assert tips
        assert not any("between your teeth" in t for t in tips)
        assert any("ridge" in t for t in tips)

    def test_velar_target_does_not_give_tongue_tip_placement(self):
        """The tip is not the articulator for /k/."""
        tips = describe_difference("K", "T")
        assert not any("ridge" in t or "between your teeth" in t for t in tips)

    def test_advice_is_capped(self):
        for e in ENGLISH_PHONEMES:
            for p in ENGLISH_PHONEMES:
                assert len(describe_difference(e, p)) <= 4

    def test_vowels_are_marked_as_vowels(self):
        assert articulation_for("AA").manner == VOWEL
        assert articulation_for("K").manner != VOWEL


class TestTargetInstructions:
    """Instructions for producing a phone, needing no knowledge of what was said.

    This is the classical detector's only route to articulatory feedback: it
    reports a match/mismatch verdict and never names the produced sound, and
    both alternatives measured (DTW template matching at 6.5% top-1, the RF at
    10.6%) identify the wrong phone far too often to display.
    """

    def test_every_phone_can_be_described(self):
        missing = [p for p in ENGLISH_PHONEMES if not describe_target(p)]
        assert missing == []

    def test_unknown_phone_yields_nothing(self):
        assert describe_target("QQ") == []

    def test_instructions_are_capped(self):
        for p in ENGLISH_PHONEMES:
            assert len(describe_target(p)) <= 4

    def test_place_is_named_for_consonants(self):
        assert any("between your teeth" in t for t in describe_target("TH"))
        assert any("lips together" in t for t in describe_target("B"))
        assert any("bottom lip" in t for t in describe_target("F"))
        assert any("BACK of your tongue" in t for t in describe_target("K"))

    def test_velar_instructions_do_not_place_the_tongue_tip_at_the_ridge(self):
        """The tip is not the articulator for /k/, /g/, /ng/."""
        for p in ("K", "G", "NG"):
            assert not any("tip at the ridge" in t for t in describe_target(p))

    def test_voicing_is_stated_for_consonants(self):
        assert any("Do not use your voice" in t for t in describe_target("S"))
        assert any("Use your voice" in t for t in describe_target("Z"))

    def test_nasals_are_not_told_to_release_the_closure(self):
        """/m/ holds the lips shut while air leaves through the nose."""
        tips = describe_target("M")
        assert any("nose" in t for t in tips)
        assert not any("release" in t for t in tips)

    def test_glottal_h_is_not_given_an_oral_constriction(self):
        """/h/ has friction at the glottis, not a narrow gap in the mouth."""
        assert not any("narrow gap" in t for t in describe_target("HH"))

    def test_vowels_are_described_by_shape_not_place(self):
        tips = describe_target("UW")
        assert any("Round your lips" in t for t in tips)
        assert any("Pull your tongue back" in t for t in tips)
        assert not any("ridge" in t for t in tips)

    def test_open_and_close_vowels_differ(self):
        assert any("wide" in t for t in describe_target("AE"))
        assert any("slightly open" in t for t in describe_target("IY"))

    def test_tenseness_is_stated_for_vowels(self):
        assert any("tense" in t for t in describe_target("IY"))
        assert any("relaxed" in t for t in describe_target("IH"))

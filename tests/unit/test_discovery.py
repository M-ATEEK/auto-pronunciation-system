from __future__ import annotations

from src.personalization.discovery import discover_from_recognition


def correct(phone: str, n: int) -> list[tuple[str, str]]:
    return [(phone, phone)] * n


class TestNoErrorNoRules:
    def test_a_fluent_learner_gets_no_rules(self):
        obs = correct("S", 10) + correct("R", 10) + correct("IY", 10)
        assert discover_from_recognition(obs) == []

    def test_empty_input_is_safe(self):
        assert discover_from_recognition([]) == []

    def test_occasional_error_is_not_systematic(self):
        """One slip in ten is a slip, not a pattern."""
        obs = correct("R", 9) + [("R", "L")]
        assert discover_from_recognition(obs) == []

    def test_a_phone_seen_twice_cannot_yield_a_rule(self):
        """Too little evidence, even when both occurrences are disputed."""
        assert discover_from_recognition([("R", "L"), ("R", "L")]) == []


class TestSystematicErrorIsFound:
    def test_consistent_substitution_yields_a_substitution_rule(self):
        obs = correct("S", 10) + [("R", "L")] * 6
        rules = discover_from_recognition(obs)
        assert len(rules) == 1
        assert rules[0].phone == "R"
        assert rules[0].type == "substitution"
        assert rules[0].substituted_with == "L"

    def test_inconsistent_errors_yield_a_distortion_rule(self):
        """Disputed every time, but with no single dominant substitute."""
        obs = [("R", "L"), ("R", "W"), ("R", "ER"), ("R", "AH"),
               ("R", "Y"), ("R", "M")]
        rules = discover_from_recognition(obs)
        assert len(rules) == 1
        assert rules[0].type == "distortion"
        assert rules[0].substituted_with is None

    def test_omissions_count_as_evidence(self):
        """A phone the learner never produces is a systematic error too."""
        rules = discover_from_recognition([("TH", None)] * 5)
        assert len(rules) == 1
        assert rules[0].phone == "TH"

    def test_confusable_substitutes_are_not_errors(self):
        """Stage 4 must not disagree with Stage 3 about what counts as wrong."""
        assert discover_from_recognition([("AH", "IH")] * 8) == []

    def test_only_the_affected_phone_is_reported(self):
        obs = correct("S", 12) + correct("T", 12) + [("V", "W")] * 8
        rules = discover_from_recognition(obs)
        assert [r.phone for r in rules] == ["V"]

    def test_confidence_reports_the_dispute_rate(self):
        obs = [("R", "L")] * 8 + correct("R", 2)
        rules = discover_from_recognition(obs)
        assert rules[0].confidence == 0.8
        assert rules[0].support == 8

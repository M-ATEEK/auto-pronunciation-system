"""Word -> phone lookup, and the refusal to invent phones.

"""

from __future__ import annotations

from src.data.aligner import _phones_for_word, phones_for_text, unknown_words


class TestKnownWords:
    def test_dictionary_word(self):
        assert _phones_for_word("hello") == ["HH", "AH", "L", "OW"]

    def test_direct_arpabet_symbol_passes_through(self):
        assert _phones_for_word("AE") == ["AE"]

    def test_punctuation_is_stripped(self):
        assert _phones_for_word("hello,") == _phones_for_word("hello")


class TestBritishSpelling:
    """The CMU dictionary is American; British spellings must still resolve."""

    def test_analysed_resolves_to_analyzed(self):
        assert _phones_for_word("analysed") == ["AE", "N", "AH", "L", "AY", "Z", "D"]

    def test_ise_family(self):
        for word in ("organised", "realise", "apologise", "recognised"):
            assert _phones_for_word(word), f"{word} should resolve"

    def test_resolved_spelling_matches_american_form(self):
        assert _phones_for_word("analyse") == _phones_for_word("analyze")
        assert _phones_for_word("organised") == _phones_for_word("organized")


class TestUnknownWords:
    def test_unknown_word_returns_no_phones(self):
        assert _phones_for_word("qwertyuiop") == []

    def test_unknown_word_is_not_invented(self):
        """The old fallback returned hash-derived phones; it must not return any."""
        assert _phones_for_word("zzzzqqqqxxxx") == []

    def test_lookup_is_deterministic_across_calls(self):
        """hash()-based phones differed run to run, breaking reproducibility."""
        first = [_phones_for_word("analysed") for _ in range(5)]
        assert all(p == first[0] for p in first)

    def test_unknown_words_are_reported(self):
        assert unknown_words("analysed the qwertyuiop data") == ["qwertyuiop"]

    def test_no_unknown_words_for_ordinary_text(self):
        assert unknown_words("the quick brown fox jumps over the lazy dog") == []

    def test_unknown_words_do_not_contribute_phones(self):
        known = phones_for_text("the cat")
        with_unknown = phones_for_text("the cat qwertyuiop")
        assert known == with_unknown

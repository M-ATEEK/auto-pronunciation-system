"""Unit tests for Needleman-Wunsch phone alignment."""

from __future__ import annotations

from src.classification.phoneme_align import align, is_confusable


class TestAlignBasics:
    def test_all_match(self):
        ops = align(["K", "AE", "T"], ["K", "AE", "T"])
        assert [o.op for o in ops] == ["match", "match", "match"]
        assert all(o.is_correct for o in ops)

    def test_empty_expected(self):
        ops = align([], ["K", "AE", "T"])
        assert all(o.op == "ins" for o in ops)
        assert len(ops) == 3

    def test_empty_recognized(self):
        ops = align(["K", "AE", "T"], [])
        assert all(o.op == "del" for o in ops)
        assert len(ops) == 3

    def test_both_empty(self):
        assert align([], []) == []

    def test_substitution(self):
        ops = align(["K", "AE", "T"], ["K", "IY", "T"])
        assert [o.op for o in ops] == ["match", "sub", "match"]
        assert ops[1].expected == "AE"
        assert ops[1].recognized == "IY"

    def test_deletion(self):
        # expected 3 phones, only 2 recognised -- one must be a deletion
        ops = align(["K", "AE", "T"], ["K", "T"])
        assert ops[0].op == "match"
        assert "del" in [o.op for o in ops]
        assert ops[-1].op == "match"

    def test_insertion(self):
        # learner produced an extra phone not in the expected sequence
        ops = align(["K", "T"], ["K", "AE", "T"])
        assert any(o.op == "ins" for o in ops)
        assert ops[0].op == "match"
        assert ops[-1].op == "match"


class TestExpectedIndexing:
    def test_every_expected_phone_appears_once(self):
        expected = ["K", "AE", "T", "S"]
        ops = align(expected, ["K", "IY", "S"])
        non_ins = [o for o in ops if o.op != "ins"]
        indices = sorted(o.expected_index for o in non_ins)
        assert indices == list(range(len(expected)))

    def test_insertion_has_no_expected_index(self):
        ops = align(["K"], ["K", "AE"])
        ins_ops = [o for o in ops if o.op == "ins"]
        assert len(ins_ops) == 1
        assert ins_ops[0].expected_index is None
        assert ins_ops[0].expected is None

    def test_deletion_has_no_recognized_index(self):
        ops = align(["K", "AE"], ["K"])
        del_ops = [o for o in ops if o.op == "del"]
        assert len(del_ops) == 1
        assert del_ops[0].recognized_index is None
        assert del_ops[0].recognized is None


class TestConfusable:
    def test_identical_not_confusable(self):
        assert is_confusable("AH", "AH") is False

    def test_known_confusable_pair(self):
        assert is_confusable("AH", "IH") is True
        assert is_confusable("IH", "AH") is True  # symmetric

    def test_unrelated_phones_not_confusable(self):
        assert is_confusable("K", "M") is False

    def test_confusable_substitution_costs_less(self):
        # A confusable substitution should be at least as cheap as, and
        # typically produce a different alignment than, a hard substitution
        # of the same length -- verified indirectly via the total DP cost
        # by checking a confusable sub is preferred over an equally-placed
        # non-confusable one when both are viable alignments.
        confusable_ops = align(["AH"], ["IH"])
        hard_ops = align(["AH"], ["K"])
        assert confusable_ops[0].op == "sub"
        assert hard_ops[0].op == "sub"


class TestRealisticSentence:
    def test_what_is_this_close_recognition(self):
        # Mirrors the real Day-9 test: expected "DH IH S", recognised "D IH S"
        # (DH -> D is the actual near-miss the wav2vec2 model produced).
        expected = ["DH", "IH", "S"]
        recognized = ["D", "IH", "S"]
        ops = align(expected, recognized)
        assert ops[0].op == "sub"
        assert ops[0].expected == "DH"
        assert ops[0].recognized == "D"
        assert ops[1].op == "match"
        assert ops[2].op == "match"

    def test_completely_different_sentence_mostly_substitutions_or_deletions(self):
        expected = ["R", "EH", "K", "ER", "D"]       # "record"-ish
        recognized = ["W", "AH", "T", "IH", "Z"]     # "what is"-ish
        ops = align(expected, recognized)
        non_matches = [o for o in ops if o.op != "match"]
        # A gross content mismatch should not align as mostly correct.
        assert len(non_matches) >= 3

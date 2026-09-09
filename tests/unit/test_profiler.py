from __future__ import annotations

from pathlib import Path

from src.personalization.profiler import ProfileManager, safe_learner_id


class TestSafeLearnerId:
    def test_ordinary_ids_are_left_alone(self):
        for ok in ("P07", "learner-001", "learner_abc_123"):
            assert safe_learner_id(ok) == ok

    def test_path_separators_cannot_survive(self):
        """The id becomes a filename, so a traversal attempt must not stay one."""
        for attempt in ("../../etc/passwd", "a/b/c", "..", "./x"):
            cleaned = safe_learner_id(attempt)
            assert "/" not in cleaned
            assert ".." not in cleaned

    def test_empty_or_blank_falls_back(self):
        assert safe_learner_id("") == "anonymous"
        assert safe_learner_id("   ") == "anonymous"

    def test_length_is_capped(self):
        assert len(safe_learner_id("x" * 500)) == 64


class TestProfileManagerStaysInsideItsDirectory:
    def test_a_traversing_id_writes_inside_the_profile_dir(self, tmp_path: Path):
        manager = ProfileManager(tmp_path)
        profile = manager.get_or_create("../../escaped")
        profile.update("R", 42.0)
        manager.save("../../escaped")

        written = list(tmp_path.rglob("*.json"))
        assert len(written) == 1
        # resolve() so a symlinked tmp dir does not fail the containment check
        assert written[0].resolve().parent == tmp_path.resolve()

    def test_the_same_id_round_trips(self, tmp_path: Path):
        manager = ProfileManager(tmp_path)
        manager.get_or_create("P07").update("TH", 12.5)
        manager.save("P07")

        reloaded = ProfileManager(tmp_path).get_or_create("P07")
        assert reloaded.get_stats("TH")["count"] == 1

    def test_distinct_learners_do_not_share_a_profile(self, tmp_path: Path):
        manager = ProfileManager(tmp_path)
        manager.get_or_create("P01").update("R", 30.0)
        assert manager.get_or_create("P02").get_stats("R") is None

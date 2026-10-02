"""Tests for the live solve feed (answers streamed to the player's watch view)."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from crosswise.solver import llm_solver
from crosswise.solver.live_feed import LiveFeed
from crosswise.solver.puzzles import get_tiny_3x3

LENGTHS = {"1-across": 3, "3-across": 3, "1-down": 3, "2-down": 3}


def _feed():
    events = []
    return LiveFeed(events.append), events


class TestTentative:
    def test_answers_emitted_as_soon_as_complete(self):
        feed, events = _feed()
        on_text = feed.text_listener("Pass 1", LENGTHS)
        for chunk in ['{"1-acr', 'oss": "CA', 'T", "3-across"', ': "BAD"', "}"]:
            on_text(chunk)
        assert [(e["clue"], e["word"]) for e in events] == [("1-across", "CAT"), ("3-across", "BAD")]
        assert all(e["type"] == "tentative" and e["phase"] == "Pass 1" for e in events)

    def test_partial_word_not_emitted(self):
        feed, events = _feed()
        feed.text_listener("Pass 1", LENGTHS)('{"1-across": "CA')
        assert events == []

    def test_wrong_length_and_unknown_clues_skipped(self):
        feed, events = _feed()
        feed.text_listener("Pass 1", LENGTHS)('{"1-across": "CATS", "9-down": "DOG", "2-down": "tad"}')
        assert [(e["clue"], e["word"]) for e in events] == [("2-down", "TAD")]

    def test_each_answer_once(self):
        feed, events = _feed()
        on_text = feed.text_listener("Pass 1", LENGTHS)
        on_text('{"1-across": "CAT"')
        on_text(', "3-across": "BAD"}')
        assert len(events) == 2


class TestCommit:
    def test_reports_changes_only(self):
        feed, events = _feed()
        feed.commit({"1-across": "CAT"}, "Pass 1")
        feed.commit({"1-across": "CAT"}, "Crossing logic")  # no change, nothing pending
        feed.commit({"1-across": "COT", "3-across": "BAD"}, "Pass 2")
        feed.commit({"3-across": "BAD"}, "Verification")
        assert [(e["answers"], e["removed"]) for e in events] == [
            ({"1-across": "CAT"}, []),
            ({"1-across": "COT", "3-across": "BAD"}, []),
            ({}, ["1-across"]),
        ]
        assert feed.snapshot() == {"3-across": "BAD"}

    def test_commit_sent_when_all_tentatives_rejected(self):
        """Nothing changed, but the viewer must learn its tentative answers were rejected."""
        feed, events = _feed()
        feed.text_listener("Pass 1", LENGTHS)('{"1-across": "COW"}')
        feed.commit({}, "Pass 1")
        assert events[-1] == {"type": "commit", "phase": "Pass 1", "answers": {}, "removed": [], "total": 0}


class TestSolveWithLlmFeed:
    def test_streamed_answers_then_commit(self):
        """solve_with_llm streams the model's text into the feed and commits each pass."""
        si, cands = get_tiny_3x3()
        clue_text = {"1-across": "Feline pet", "3-across": "Not good", "1-down": "Taxi", "2-down": "Small amount"}
        reply = json.dumps({"1-across": "CAT", "3-across": "BAD", "1-down": "CAB", "2-down": "TOD"})

        def fake_create(client, on_text=None, **kwargs):
            # The solve pass streams; feed the reply in small chunks like the API would
            if on_text:
                for i in range(0, len(reply), 7):
                    on_text(reply[i:i + 7])
            return SimpleNamespace(stop_reason="end_turn", stop_details=None, usage=None,
                                   content=[SimpleNamespace(type="text", text=reply)])

        feed, events = _feed()
        with patch.object(llm_solver, "create_message", side_effect=fake_create), \
             patch.object(llm_solver, "prefill_from_db", return_value={}), \
             patch.object(llm_solver, "propagate_constraints", return_value={}), \
             patch.object(llm_solver, "find_conflict_clusters", return_value=[]), \
             patch.object(llm_solver, "verify_and_repair", side_effect=lambda si, t, c, a, **k: (a, {})):
            result = llm_solver.solve_with_llm(si, clue_text, cands, pass_efforts=["medium"], live=feed)

        tentative = [(e["clue"], e["word"]) for e in events if e["type"] == "tentative"]
        assert tentative == [("1-across", "CAT"), ("3-across", "BAD"), ("1-down", "CAB"), ("2-down", "TOD")]
        commits = [e for e in events if e["type"] == "commit"]
        assert commits[0]["phase"] == "Pass 1"
        assert commits[0]["answers"] == result
        assert events.index(commits[0]) > max(i for i, e in enumerate(events) if e["type"] == "tentative")

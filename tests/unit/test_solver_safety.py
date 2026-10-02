"""Tests for the solver's safety nets: CSP cleanup, answer parsing, fill
verification/repair, the forced-word check, and the grid-detection fallback.

All Claude calls are mocked; nothing here hits the network.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from crosswise.solver import llm_solver
from crosswise.solver.csp import solve_csp
from crosswise.solver.puzzles import get_tiny_3x3

#   C A T
#   A # A
#   B A D
SOLUTION = {"1-across": "CAT", "3-across": "BAD", "1-down": "CAB", "2-down": "TAD"}
CLUE_TEXT = {"1-across": "Feline pet", "3-across": "Not good", "1-down": "Taxi", "2-down": "Small amount"}


def _response(text: str, stop_reason: str = "end_turn"):
    """A minimal stand-in for an Anthropic Message."""
    return SimpleNamespace(
        stop_reason=stop_reason, stop_details=None, usage=None, model="claude-opus-5-5",
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
    )


class _StubWordIndex:
    """Stands in for the 600K-word index (whose data files aren't in CI)."""

    def __init__(self, words):
        self.words = set(words)

    def contains(self, word):
        return word.upper() in self.words

    def match_pattern(self, pattern, max_results=50):
        regex = __import__("re").compile("^" + pattern.replace("_", "[A-Z]") + "$")
        return [w for w in self.words if regex.match(w)][:max_results]


@pytest.fixture
def word_index():
    stub = _StubWordIndex(list(SOLUTION.values()) + ["COT", "COB", "BED", "TOD", "TED"])
    with patch("crosswise.solver.word_index.get_word_index", return_value=stub):
        yield stub


# --- CSP cleanup -----------------------------------------------------------

class TestCspCleanupKeepsLlmAnswers:
    def _setup(self):
        si, cands = get_tiny_3x3()
        cands = dict(cands)
        cands["1-across"] = ["COT", "CAT"]
        scores = {cid: {w: 0.9 for w in ws} for cid, ws in cands.items()}
        scores["1-across"] = {"COT": 0.95, "CAT": 0.9}  # the wrong word scores higher
        llm = {"1-across": "CAT", "1-down": "CAB", "3-across": "BAD"}
        return si, cands, scores, llm

    def test_unlocked_seed_can_replace_correct_answers(self):
        """The old behavior, kept as a regression marker: a seed is only a warm start."""
        si, cands, scores, llm = self._setup()
        r = solve_csp(si, cands, return_partial=True, candidate_scores=scores,
                      mac_mode="search-only", seed_assignment=llm)
        assert r.assignment["1-across"] == "COT"

    def test_locked_answers_are_kept_and_blanks_filled(self):
        si, cands, scores, llm = self._setup()
        r = solve_csp(si, cands, return_partial=True, candidate_scores=scores,
                      mac_mode="search-only", locked=llm)
        assert {k: r.assignment.get(k) for k in llm} == llm
        # 2-down's middle cell crosses nothing, so any T?D candidate completes the grid
        assert r.assignment["2-down"] in {"TAD", "TED", "TOD"}

    def test_locked_answer_outside_candidates_is_kept(self):
        si, cands, scores, llm = self._setup()
        cands["1-across"] = ["COW", "CUP"]
        r = solve_csp(si, cands, return_partial=True, mac_mode="search-only", locked=llm)
        assert {k: r.assignment.get(k) for k in llm} == llm
        assert r.assignment["2-down"] in {"TAD", "TED", "TOD"}

    def test_locked_answer_never_dropped_when_crosser_has_no_fit(self):
        """A locked answer that leaves a crossing clue unfillable stays; the crosser stays blank."""
        si, cands, _, llm = self._setup()
        cands["2-down"] = ["SOD", "POD"]  # nothing starts with T (from CAT)
        r = solve_csp(si, cands, return_partial=True, mac_mode="search-only", locked=llm)
        assert {k: r.assignment.get(k) for k in llm} == llm
        assert "2-down" not in r.assignment

    def test_should_stop_ends_search(self):
        si, cands, _, llm = self._setup()
        cands["2-down"] = ["SOD"]
        r = solve_csp(si, cands, return_partial=True, mac_mode="search-only",
                      locked=llm, should_stop=lambda: True)
        assert {k: r.assignment.get(k) for k in llm} == llm

    def test_pipeline_rejects_result_that_drops_an_answer(self):
        """Even if the search returned more answers, any lost LLM answer means keep the LLM's."""
        from crosswise.api import pipeline
        si, cands, scores, llm = self._setup()
        worse = SimpleNamespace(assignment={"1-across": "COT", "1-down": "COB",
                                            "3-across": "BED", "2-down": "TOD"})
        with patch("crosswise.solver.llm_solver.solve_with_llm", return_value=dict(llm)), \
             patch("crosswise.solver.csp.solve_csp", return_value=worse):
            assignment, solved = pipeline._run_solver(
                si, CLUE_TEXT, cands, scores, scores, 4, lambda p: None, lambda: "",
            )
        assert assignment == llm and solved == 3


# --- Answer parsing -----------------------------------------------------------

class TestParseAnswerJson:
    @pytest.mark.parametrize("text,expected", [
        ('{"1-across": "CAT"}', {"1-across": "CAT"}),
        ('```json\n{"1-across": "CAT"}\n```', {"1-across": "CAT"}),
        ('Let me check {"1-across": "COT"}... Final: {"1-across": "CAT"}', {"1-across": "CAT"}),
        ('{"1-across": "CAT", "2-down": 7, "3-across": null}', {"1-across": "CAT"}),
        ('["CAT"]', None),
        ('42', None),
        ('', None),
        ('no json here', None),
    ])
    def test_cases(self, text, expected):
        assert llm_solver._parse_answer_json(text) == expected

    def test_solve_pass_survives_non_object_reply(self):
        si, cands = get_tiny_3x3()
        with patch.object(llm_solver, "create_message", return_value=_response('["CAT", "BAD"]')):
            assert llm_solver.solve_pass(si, CLUE_TEXT, cands, {}, pass_num=1) is None

    def test_solve_pass_truncated_is_failure(self):
        si, cands = get_tiny_3x3()
        with patch.object(llm_solver, "create_message",
                          return_value=_response("", stop_reason="max_tokens")):
            assert llm_solver.solve_pass(si, CLUE_TEXT, cands, {}, pass_num=4) is None


# --- Fill verification and repair ----------------------------------------------

def _wrong_fill():
    # 3-across BED and 2-down TED are wrong but consistent with each other
    return {"1-across": "CAT", "1-down": "CAB", "3-across": "BED", "2-down": "TED"}


class TestFindSuspectAnswers:
    def test_unknown_words_are_hints_not_flags(self, word_index):
        si, cands = get_tiny_3x3()
        fill = {**SOLUTION, "2-down": "TXD"}  # not a word, not a candidate
        mock = patch.object(llm_solver, "create_message",
                            return_value=_response(json.dumps({"suspects": []})))
        with mock as create:
            suspects = llm_solver.find_suspect_answers(si, CLUE_TEXT, cands, fill)
        assert suspects == {}  # the reviewer decides; it flagged nothing
        prompt = create.call_args.kwargs["messages"][0]["content"]
        assert "2-down=TXD" in prompt and "aren't in our word list" in prompt
        assert "flag that crossing answer too" in prompt

    def test_reviewer_flags_are_returned(self, word_index):
        si, cands = get_tiny_3x3()
        reply = {"suspects": [{"clue_id": "3-across", "reason": "BED isn't 'not good'"},
                              {"clue_id": "9-across", "reason": "not in grid"}]}
        with patch.object(llm_solver, "create_message", return_value=_response(json.dumps(reply))):
            suspects = llm_solver.find_suspect_answers(si, CLUE_TEXT, cands, _wrong_fill())
        assert suspects == {"3-across": "BED isn't 'not good'"}

    def test_review_error_flags_nothing(self, word_index):
        si, cands = get_tiny_3x3()
        with patch.object(llm_solver, "create_message", side_effect=RuntimeError("down")):
            assert llm_solver.find_suspect_answers(si, CLUE_TEXT, cands, _wrong_fill()) == {}


class TestResolveSuspectAnswers:
    def test_clears_suspect_and_crossers_then_refills(self, word_index):
        si, cands = get_tiny_3x3()
        reply = json.dumps({"3-across": "BAD", "2-down": "TAD", "1-down": "CAB"})
        with patch.object(llm_solver, "create_message", return_value=_response(reply)) as create:
            new = llm_solver.resolve_suspect_answers(
                si, CLUE_TEXT, cands, _wrong_fill(), {"3-across": "wrong"})
        assert new == SOLUTION
        prompt = create.call_args.kwargs["messages"][0]["content"]
        # The suspect and both clues crossing it were sent for re-solving
        for cid in ("3-across", "1-down", "2-down"):
            assert f"{cid}:" in prompt

    def test_answer_that_breaks_kept_letters_is_dropped(self, word_index):
        si, cands = get_tiny_3x3()
        # 1-across (CAT) is kept; "DAB" for 1-down would need a D where CAT has C
        reply = json.dumps({"3-across": "BAD", "2-down": "TAD", "1-down": "DAB"})
        with patch.object(llm_solver, "create_message", return_value=_response(reply)):
            new = llm_solver.resolve_suspect_answers(
                si, CLUE_TEXT, cands, _wrong_fill(), {"3-across": "wrong"})
        assert "1-down" not in new and new["1-across"] == "CAT"

    def test_api_error_keeps_assignment(self, word_index):
        si, cands = get_tiny_3x3()
        with patch.object(llm_solver, "create_message", side_effect=RuntimeError("down")):
            fill = _wrong_fill()
            assert llm_solver.resolve_suspect_answers(si, CLUE_TEXT, cands, fill, {"3-across": "x"}) == fill


class TestVerifyAndRepair:
    def test_no_suspects_no_change(self, word_index):
        si, cands = get_tiny_3x3()
        with patch.object(llm_solver, "find_suspect_answers", return_value={}):
            new, suspects = llm_solver.verify_and_repair(si, CLUE_TEXT, cands, dict(SOLUTION))
        assert new == SOLUTION and suspects == {}

    def test_backstop_repairs_blank_left_by_resolve(self, word_index):
        """If the re-solve leaves a dead end, conflict resolution runs on it."""
        si, cands = get_tiny_3x3()
        left_blank = {"1-across": "CAT", "1-down": "CAB", "2-down": "TED"}  # 3-across blank
        cluster = {"unsolved": [{"clue_id": "3-across"}], "blamed": [{"clue_id": "2-down"}]}
        with patch.object(llm_solver, "find_suspect_answers", return_value={"3-across": "x"}), \
             patch.object(llm_solver, "resolve_suspect_answers", return_value=dict(left_blank)), \
             patch.object(llm_solver, "find_conflict_clusters", return_value=[cluster]) as fcc, \
             patch.object(llm_solver, "resolve_conflict_cluster",
                          return_value={"2-down": "TAD", "3-across": "BAD"}), \
             patch.object(llm_solver, "solve_pass") as sp:
            new, _ = llm_solver.verify_and_repair(si, CLUE_TEXT, cands, _wrong_fill())
        fcc.assert_called_once()
        sp.assert_not_called()  # nothing left blank for the last-chance pass
        assert new == SOLUTION


# --- Forced-word check ------------------------------------------------------------

class TestForcedWordCheck:
    def test_no_clue_text_fails_closed(self):
        assert llm_solver._dictionary_and_llm_confirm("CAT", "") is False

    def test_api_error_fails_closed(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        with patch.object(llm_solver, "_get_dictionary_definitions", return_value=None), \
             patch.object(llm_solver, "create_message", side_effect=RuntimeError("down")):
            assert llm_solver._dictionary_and_llm_confirm("CAT", "Feline pet") is False

    def test_substring_is_not_a_dictionary_match(self, monkeypatch):
        """'art' inside 'start' must not count; falls through to the model, which says no."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        with patch.object(llm_solver, "_get_dictionary_definitions", return_value="to start a race"), \
             patch.object(llm_solver, "create_message",
                          return_value=_response(json.dumps({"correct": False}))) as create:
            assert llm_solver._dictionary_and_llm_confirm("BEGIN", "Art class") is False
        create.assert_called_once()

    def test_whole_word_dictionary_match_accepts(self):
        with patch.object(llm_solver, "_get_dictionary_definitions", return_value="to start a race"), \
             patch.object(llm_solver, "create_message") as create:
            assert llm_solver._dictionary_and_llm_confirm("BEGIN", "Start up") is True
        create.assert_not_called()


# --- Grid detection fallback ------------------------------------------------------

def _grid_image(n: int, size: int = 600) -> np.ndarray:
    img = np.full((size, size), 255, np.uint8)
    for i in range(n + 1):
        p = round(i * (size - 1) / n)
        cv2.line(img, (p, 0), (p, size - 1), 0, 3)
        cv2.line(img, (0, p), (size - 1, p), 0, 3)
    return img


class TestGridIntersectionFallback:
    def test_one_line_per_row_and_column(self):
        from crosswise.vision.grid_detection import _detect_grid_intersections
        xs, ys = _detect_grid_intersections(_grid_image(15))
        assert len(xs) == 16 and len(ys) == 16
        assert xs == sorted(xs) and ys == sorted(ys)

    def test_implausible_grid_size_is_rejected(self):
        from crosswise.vision import grid_detection
        lines = list(range(0, 600, 10))  # 59 x 59 cells
        with patch.object(grid_detection, "_detect_grid_lines_by_projection", return_value=(lines, lines)):
            with pytest.raises(ValueError, match="implausible"):
                grid_detection.detect_grid(_grid_image(15), None)

"""Tests for LLM-based candidate generation."""

import pytest
from unittest.mock import patch, MagicMock

from crosswise.solver.candidate_generator import (
    ClueInput,
    generate_candidates,
    generate_candidates_batch,
    _build_prompt,
    _parse_response,
)


class TestBuildPrompt:
    """Test prompt building."""

    def test_single_clue(self):
        """Prompt should include clue details."""
        clues = [ClueInput(clue_id="1-across", text="Capital of France", length=5)]
        prompt = _build_prompt(clues, candidates_per_clue=4)

        assert "1-across" in prompt
        assert "Capital of France" in prompt
        assert "5 letters" in prompt
        assert "4 candidate" in prompt

    def test_multiple_clues(self):
        """Prompt should include all clues."""
        clues = [
            ClueInput(clue_id="1-across", text="Capital of France", length=5),
            ClueInput(clue_id="2-down", text="Sound reflection", length=4),
        ]
        prompt = _build_prompt(clues, candidates_per_clue=4)

        assert "1-across" in prompt
        assert "2-down" in prompt
        assert "Capital of France" in prompt
        assert "Sound reflection" in prompt


class TestParseResponse:
    """Test response parsing."""

    def test_valid_json(self):
        """Should parse valid JSON response."""
        response = '{"1-across": ["PARIS", "LYONS", "TOURS", "BREST"]}'
        result = _parse_response(response, ["1-across"])

        assert result["1-across"] == ["PARIS", "LYONS", "TOURS", "BREST"]

    def test_json_with_markdown(self):
        """Should handle JSON in markdown code blocks."""
        response = '```json\n{"1-across": ["PARIS", "LYONS"]}\n```'
        result = _parse_response(response, ["1-across"])

        assert result["1-across"] == ["PARIS", "LYONS"]

    def test_normalizes_to_uppercase(self):
        """Should normalize answers to uppercase."""
        response = '{"1-across": ["paris", "Lyons"]}'
        result = _parse_response(response, ["1-across"])

        assert result["1-across"] == ["PARIS", "LYONS"]

    def test_removes_spaces(self):
        """Should remove spaces from multi-word answers."""
        response = '{"1-across": ["NEW YORK", "Los Angeles"]}'
        result = _parse_response(response, ["1-across"])

        assert result["1-across"] == ["NEWYORK", "LOSANGELES"]

    def test_missing_clue_returns_empty(self):
        """Should return empty list for missing clues."""
        response = '{"1-across": ["PARIS"]}'
        result = _parse_response(response, ["1-across", "2-down"])

        assert result["1-across"] == ["PARIS"]
        assert result["2-down"] == []


class TestGenerateCandidatesBatch:
    """Test batch candidate generation."""

    @patch("crosswise.solver.candidate_generator.OpenAI")
    def test_calls_api_with_prompt(self, mock_openai_class):
        """Should call OpenAI API with correct prompt."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"1-across": ["PARIS"]}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        clues = [ClueInput(clue_id="1-across", text="Capital of France", length=5)]
        result = generate_candidates_batch(clues, api_key="test-key")

        # Verify API was called
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert "Capital of France" in call_args.kwargs["messages"][0]["content"]

        assert result["1-across"] == ["PARIS"]

    def test_raises_without_api_key(self):
        """Should raise if no API key provided."""
        clues = [ClueInput(clue_id="1-across", text="Test", length=4)]

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                generate_candidates_batch(clues)


class TestGenerateCandidates:
    """Test batched candidate generation."""

    @patch("crosswise.solver.candidate_generator.generate_candidates_batch")
    def test_batches_clues(self, mock_batch):
        """Should batch clues according to batch_size."""
        mock_batch.return_value = {}

        clues = [
            ClueInput(clue_id=f"{i}-across", text=f"Clue {i}", length=4)
            for i in range(75)
        ]

        generate_candidates(clues, batch_size=30, api_key="test-key")

        # Should have 3 batches: 30 + 30 + 15
        assert mock_batch.call_count == 3

    @patch("crosswise.solver.candidate_generator.generate_candidates_batch")
    def test_combines_batch_results(self, mock_batch):
        """Should combine results from all batches."""
        mock_batch.side_effect = [
            {"1-across": ["WORD"]},
            {"2-across": ["TEST"]},
        ]

        clues = [
            ClueInput(clue_id="1-across", text="Clue 1", length=4),
            ClueInput(clue_id="2-across", text="Clue 2", length=4),
        ]

        result = generate_candidates(clues, batch_size=1, api_key="test-key")

        assert result["1-across"] == ["WORD"]
        assert result["2-across"] == ["TEST"]

    @patch("crosswise.solver.candidate_generator.generate_candidates_batch")
    def test_callback_called_per_batch(self, mock_batch):
        """Should call progress callback for each batch."""
        mock_batch.return_value = {}
        callback = MagicMock()

        clues = [
            ClueInput(clue_id=f"{i}-across", text=f"Clue {i}", length=4)
            for i in range(45)
        ]

        generate_candidates(
            clues, batch_size=30, api_key="test-key", on_batch_complete=callback
        )

        # 2 batches: 30 + 15
        assert callback.call_count == 2
        callback.assert_any_call(1, 2, {})
        callback.assert_any_call(2, 2, {})

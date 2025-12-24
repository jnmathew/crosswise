"""Test data models."""
import pytest
from src.core.models import Cell, Clue, Puzzle, Direction, PuzzleMetadata


class TestCell:
    """Test Cell model."""

    def test_cell_creation(self):
        """Test basic cell creation."""
        cell = Cell(row=0, col=0, is_block=False, clue_number=1)
        assert cell.row == 0
        assert cell.col == 0
        assert cell.clue_number == 1
        assert cell.is_block is False

    def test_cell_with_value(self):
        """Test cell with solved letter."""
        cell = Cell(row=0, col=0, value="A")
        assert cell.value == "A"

    def test_cell_value_uppercase(self):
        """Test cell value is auto-uppercased."""
        cell = Cell(row=0, col=0, value="a")
        assert cell.value == "A"

    def test_cell_value_must_be_letter(self):
        """Test cell value validation."""
        with pytest.raises(ValueError):
            Cell(row=0, col=0, value="1")

    def test_cell_confidence(self):
        """Test confidence score validation."""
        cell = Cell(row=0, col=0, confidence=0.95)
        assert cell.confidence == 0.95

        with pytest.raises(ValueError):
            Cell(row=0, col=0, confidence=1.5)  # Out of range


class TestClue:
    """Test Clue model."""

    def test_clue_creation(self):
        """Test basic clue creation."""
        clue = Clue(
            number=1,
            direction=Direction.ACROSS,
            text="Capital of France",
            answer_length=[5],
        )
        assert clue.number == 1
        assert clue.direction == Direction.ACROSS
        assert clue.text == "Capital of France"
        assert clue.answer_length == [5]

    def test_clue_with_answer(self):
        """Test clue with solved answer."""
        clue = Clue(
            number=1,
            direction=Direction.ACROSS,
            text="Capital of France",
            answer_length=[5],
            answer="PARIS",
        )
        assert clue.answer == "PARIS"

    def test_clue_answer_uppercase(self):
        """Test answer is auto-uppercased."""
        clue = Clue(
            number=1,
            direction=Direction.ACROSS,
            text="Test",
            answer_length=[4],
            answer="test",
        )
        assert clue.answer == "TEST"

    def test_clue_total_length(self):
        """Test total_length property."""
        clue = Clue(
            number=1,
            direction=Direction.ACROSS,
            text="Multi-word answer",
            answer_length=[6, 5],
        )
        assert clue.total_length == 11

    def test_invalid_answer_length(self):
        """Test answer_length validation."""
        with pytest.raises(ValueError):
            Clue(
                number=1,
                direction=Direction.ACROSS,
                text="Test",
                answer_length=[0],  # Invalid
            )

        with pytest.raises(ValueError):
            Clue(
                number=1,
                direction=Direction.ACROSS,
                text="Test",
                answer_length=[],  # Empty
            )


class TestPuzzle:
    """Test Puzzle model."""

    def test_puzzle_creation(self, sample_puzzle_data):
        """Test puzzle creation from dict."""
        puzzle = Puzzle.from_json(sample_puzzle_data)
        assert puzzle.rows == 1
        assert puzzle.cols == 5
        assert len(puzzle.clues[Direction.ACROSS]) == 1

    def test_puzzle_get_cell(self, sample_puzzle_data):
        """Test get_cell method."""
        puzzle = Puzzle.from_json(sample_puzzle_data)
        cell = puzzle.get_cell(0, 0)
        assert cell is not None
        assert cell.row == 0
        assert cell.col == 0
        assert cell.clue_number == 1

    def test_puzzle_get_cell_out_of_bounds(self, sample_puzzle_data):
        """Test get_cell with invalid coordinates."""
        puzzle = Puzzle.from_json(sample_puzzle_data)
        cell = puzzle.get_cell(10, 10)
        assert cell is None

    def test_puzzle_to_json(self, sample_puzzle_data):
        """Test JSON serialization."""
        puzzle = Puzzle.from_json(sample_puzzle_data)
        json_data = puzzle.to_json()
        assert "grid" in json_data
        assert "clues" in json_data
        assert "metadata" in json_data

    def test_puzzle_roundtrip(self, sample_puzzle_data):
        """Test from_json -> to_json roundtrip."""
        puzzle1 = Puzzle.from_json(sample_puzzle_data)
        json_data = puzzle1.to_json()
        puzzle2 = Puzzle.from_json(json_data)

        assert puzzle1.rows == puzzle2.rows
        assert puzzle1.cols == puzzle2.cols
        assert len(puzzle1.clues[Direction.ACROSS]) == len(
            puzzle2.clues[Direction.ACROSS]
        )

"""Test data models."""
import pytest
from src.core.models import Cell, Clue, Direction


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

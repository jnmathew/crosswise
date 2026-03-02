"""
Core data models for crossword puzzle representation.
Uses Pydantic for validation and serialization.
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class Direction(str, Enum):
    """Clue direction."""

    ACROSS = "across"
    DOWN = "down"


class Cell(BaseModel):
    """Represents a single grid cell."""

    row: int = Field(ge=0, description="Row index (0-based)")
    col: int = Field(ge=0, description="Column index (0-based)")
    is_block: bool = Field(default=False, description="True if black/blocked cell")
    clue_number: Optional[int] = Field(
        default=None, ge=1, description="Clue number if present"
    )
    value: Optional[str] = Field(
        default=None, max_length=1, description="Solved letter"
    )
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: Optional[str]) -> Optional[str]:
        """Ensure value is uppercase letter if provided."""
        if v is not None:
            v = v.upper()
            if not v.isalpha():
                raise ValueError("Cell value must be a letter")
        return v


class Clue(BaseModel):
    """Represents a crossword clue."""

    number: int = Field(ge=1, description="Clue number")
    direction: Direction
    text: str = Field(min_length=1, description="Clue text")
    answer_length: List[int] = Field(
        description="Letter counts, e.g., [6,5] for (6,5)"
    )
    answer: Optional[str] = Field(default=None, description="Solved answer")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ocr_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("answer_length")
    @classmethod
    def validate_answer_length(cls, v: List[int]) -> List[int]:
        """Ensure answer_length contains positive integers."""
        if not v or any(x <= 0 for x in v):
            raise ValueError("answer_length must contain positive integers")
        return v

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v: Optional[str]) -> Optional[str]:
        """Ensure answer is uppercase if provided."""
        if v is not None:
            return v.upper()
        return v

    @property
    def total_length(self) -> int:
        """Total answer length across all words."""
        return sum(self.answer_length)


class PuzzleMetadata(BaseModel):
    """Metadata about puzzle processing."""

    source_image: str
    processed_at: str
    grid_size: tuple[int, int] = Field(description="(rows, cols)")
    total_clues: int = Field(ge=0)
    ocr_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    processing_time_seconds: Optional[float] = Field(default=None, ge=0.0)
    debug_artifacts: List[str] = Field(default_factory=list)


class Puzzle(BaseModel):
    """Complete crossword puzzle representation."""

    grid: List[List[Cell]] = Field(description="2D grid of cells")
    clues: Dict[Direction, List[Clue]] = Field(
        default_factory=lambda: {Direction.ACROSS: [], Direction.DOWN: []}
    )
    metadata: PuzzleMetadata

    @property
    def rows(self) -> int:
        """Number of rows in the grid."""
        return len(self.grid)

    @property
    def cols(self) -> int:
        """Number of columns in the grid."""
        return len(self.grid[0]) if self.grid else 0

    def get_cell(self, row: int, col: int) -> Optional[Cell]:
        """
        Safely get cell at position.

        Args:
            row: Row index (0-based)
            col: Column index (0-based)

        Returns:
            Cell at position, or None if out of bounds
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        return None


from pydantic import BaseModel
from typing import Optional
from enum import Enum


class SessionStatus(str, Enum):
    UPLOADED = "uploaded"
    GRID_DETECTED = "grid_detected"
    AWAITING_MASK = "awaiting_mask"
    OCR_RUNNING = "ocr_running"
    VERIFIED = "verified"
    VERIFICATION_FAILED = "verification_failed"
    SOLVING = "solving"
    GENERATING_HINTS = "generating_hints"
    COMPLETE = "complete"
    FAILED = "failed"


class UploadResponse(BaseModel):
    session_id: str
    status: SessionStatus
    grid_size: list[int]
    clue_slot_count: int
    warped_grid_url: str
    original_image_url: str


class MaskRect(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class MaskRequest(BaseModel):
    rectangles: list[MaskRect]
    separators: list[MaskRect]


class MaskResponse(BaseModel):
    status: SessionStatus
    verification_passed: bool
    ocr_clue_count: int
    grid_slot_count: int
    matched_count: int
    errors: list[str]
    puzzle_id: Optional[str] = None


class SolveProgress(BaseModel):
    stage: str
    message: str
    progress: float  # 0.0 to 1.0


class SolveStatusResponse(BaseModel):
    status: SessionStatus
    solved_count: Optional[int] = None
    total_clues: Optional[int] = None


class GridEditRequest(BaseModel):
    black_cells: list[list[bool]]  # rows x cols, True = black


class GridEditResponse(BaseModel):
    clue_slot_count: int
    clue_number_count: int
    grid_size: list[int]


class GridResizeRequest(BaseModel):
    rows: int
    cols: int


class GridResizeResponse(BaseModel):
    grid_size: list[int]
    clue_slot_count: int
    black_cells: list[list[bool]]


class StartPipelineResponse(BaseModel):
    session_id: str
    puzzle_id: str
    status: SessionStatus


class ManualCropRequest(BaseModel):
    corners: list[list[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

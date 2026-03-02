"""FastAPI server for the Crosswise crossword puzzle app."""

import asyncio
import json
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

from src.core.config import settings
from src.api.models import (
    SessionStatus,
    UploadResponse,
    MaskRequest,
    MaskResponse,
    StartPipelineResponse,
    SolveStatusResponse,
    GridEditRequest,
    GridEditResponse,
    GridResizeRequest,
    GridResizeResponse,
    ManualCropRequest,
)
from src.api.session_manager import SessionManager
from src.api import pipeline

SESSIONS_DIR = settings.DATA_DIR / "sessions"
PUZZLES_DIR = Path(__file__).parent.parent.parent / "web" / "public" / "puzzles"

session_mgr = SessionManager(SESSIONS_DIR)

# Track background solve progress per session
progress_queues: dict[str, asyncio.Queue] = {}
cancel_events: dict[str, threading.Event] = {}

app = FastAPI(title="Crosswise API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve session files (images)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/api/files", StaticFiles(directory=str(SESSIONS_DIR)), name="session_files")


@app.get("/api/config")
async def get_config():
    return {"ocr_provider": settings.OCR_PROVIDER}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_photo(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG or PNG)")

    session_id = session_mgr.create_session()
    session_dir = session_mgr.get_session_dir(session_id)

    # Save uploaded file
    original_path = session_dir / "original.jpg"
    content = await file.read()
    with open(original_path, "wb") as f:
        f.write(content)

    try:
        result = pipeline.run_grid_detection(session_dir, settings)
    except Exception as e:
        session_mgr.update_status(session_id, SessionStatus.FAILED, error=str(e))
        raise HTTPException(422, f"Grid detection failed: {e}")

    session_mgr.update_status(
        session_id,
        SessionStatus.GRID_DETECTED,
        grid_size=result["grid_size"],
        clue_slot_count=result["clue_slot_count"],
    )

    return UploadResponse(
        session_id=session_id,
        status=SessionStatus.GRID_DETECTED,
        grid_size=list(result["grid_size"]),
        clue_slot_count=result["clue_slot_count"],
        warped_grid_url=f"/api/files/{session_id}/warped.jpg",
        original_image_url=f"/api/files/{session_id}/original.jpg",
    )


@app.post("/api/{session_id}/grid-edit", response_model=GridEditResponse)
async def edit_grid(session_id: str, edit: GridEditRequest):
    session_dir = session_mgr.get_session_dir(session_id)
    try:
        result = pipeline.apply_grid_edit(session_dir, edit.black_cells)
    except Exception as e:
        raise HTTPException(422, f"Grid edit failed: {e}")

    return GridEditResponse(
        clue_slot_count=result["clue_slot_count"],
        clue_number_count=result["clue_number_count"],
        grid_size=list(result["grid_size"]),
    )


@app.post("/api/{session_id}/resize-grid", response_model=GridResizeResponse)
async def resize_grid(session_id: str, req: GridResizeRequest):
    session_dir = session_mgr.get_session_dir(session_id)
    if req.rows < 3 or req.rows > 30 or req.cols < 3 or req.cols > 30:
        raise HTTPException(400, "Rows and cols must be between 3 and 30")
    try:
        result = pipeline.resize_grid(session_dir, req.rows, req.cols)
    except Exception as e:
        raise HTTPException(422, f"Grid resize failed: {e}")

    return GridResizeResponse(
        grid_size=result["grid_size"],
        clue_slot_count=result["clue_slot_count"],
        black_cells=result["black_cells"],
    )


@app.post("/api/{session_id}/manual-crop", response_model=UploadResponse)
async def manual_crop(session_id: str, req: ManualCropRequest):
    """Re-run grid detection with user-specified quad corners."""
    session_dir = session_mgr.get_session_dir(session_id)
    if not (session_dir / "original.jpg").exists():
        raise HTTPException(404, "No original image found for this session")

    if len(req.corners) != 4 or any(len(c) != 2 for c in req.corners):
        raise HTTPException(400, "corners must be exactly 4 points, each [x, y]")

    try:
        result = pipeline.run_grid_detection(session_dir, settings, manual_quad=req.corners)
    except Exception as e:
        raise HTTPException(422, f"Manual crop failed: {e}")

    session_mgr.update_status(
        session_id,
        SessionStatus.GRID_DETECTED,
        grid_size=result["grid_size"],
        clue_slot_count=result["clue_slot_count"],
    )

    return UploadResponse(
        session_id=session_id,
        status=SessionStatus.GRID_DETECTED,
        grid_size=list(result["grid_size"]),
        clue_slot_count=result["clue_slot_count"],
        warped_grid_url=f"/api/files/{session_id}/warped.jpg",
        original_image_url=f"/api/files/{session_id}/original.jpg",
    )


@app.post("/api/{session_id}/mask", response_model=MaskResponse)
async def submit_mask(session_id: str, mask: MaskRequest, background_tasks: BackgroundTasks):
    session_dir = session_mgr.get_session_dir(session_id)
    session_mgr.update_status(session_id, SessionStatus.OCR_RUNNING)

    try:
        result = pipeline.run_ocr_and_verify(session_dir, mask, settings)
    except Exception as e:
        session_mgr.update_status(session_id, SessionStatus.FAILED, error=str(e))
        raise HTTPException(422, f"OCR/verification failed: {e}")

    if not result["verification_passed"]:
        session_mgr.update_status(session_id, SessionStatus.VERIFICATION_FAILED)
        return MaskResponse(
            status=SessionStatus.VERIFICATION_FAILED,
            verification_passed=False,
            ocr_clue_count=result["ocr_clue_count"],
            grid_slot_count=result["grid_slot_count"],
            matched_count=result["matched_count"],
            errors=result["errors"],
        )

    # Build preliminary puzzle JSON (no answers yet)
    puzzle_id = session_id
    pipeline.build_preliminary_puzzle(session_dir, PUZZLES_DIR, puzzle_id)

    session_mgr.update_status(session_id, SessionStatus.VERIFIED, puzzle_id=puzzle_id)

    # Fire background solve
    queue: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()
    progress_queues[session_id] = queue
    cancel_events[session_id] = cancel_event
    background_tasks.add_task(
        pipeline.run_solve_background, session_dir, PUZZLES_DIR, puzzle_id, queue, session_mgr, session_id,
        cancel_event,
    )

    return MaskResponse(
        status=SessionStatus.VERIFIED,
        verification_passed=True,
        ocr_clue_count=result["ocr_clue_count"],
        grid_slot_count=result["grid_slot_count"],
        matched_count=result["matched_count"],
        errors=[],
        puzzle_id=puzzle_id,
    )


@app.post("/api/{session_id}/start-pipeline", response_model=StartPipelineResponse)
async def start_pipeline(session_id: str, mask: MaskRequest, background_tasks: BackgroundTasks):
    """Start the full OCR + solve pipeline in the background.

    Creates a skeleton puzzle immediately so the player can load right away,
    then runs OCR, verification, solving, and hint generation as a background task.
    """
    session_dir = session_mgr.get_session_dir(session_id)
    puzzle_id = session_id

    # Build skeleton puzzle so the player has something to load immediately
    pipeline.build_skeleton_puzzle(session_dir, PUZZLES_DIR, puzzle_id)

    # Create progress queue and start background pipeline
    queue: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()
    progress_queues[session_id] = queue
    cancel_events[session_id] = cancel_event
    background_tasks.add_task(
        pipeline.run_full_pipeline_background,
        session_dir, PUZZLES_DIR, puzzle_id, mask, settings,
        queue, session_mgr, session_id, cancel_event,
    )

    session_mgr.update_status(session_id, SessionStatus.OCR_RUNNING, puzzle_id=puzzle_id)

    return StartPipelineResponse(
        session_id=session_id,
        puzzle_id=puzzle_id,
        status=SessionStatus.OCR_RUNNING,
    )


@app.post("/api/{session_id}/solve")
async def retrigger_solve(session_id: str, background_tasks: BackgroundTasks):
    """Re-trigger the solve for an existing session (e.g., after pipeline fix)."""
    session_dir = session_mgr.get_session_dir(session_id)
    puzzle_id = session_mgr.get_session_data(session_id).get("puzzle_id", session_id)
    puzzle_path = PUZZLES_DIR / f"{puzzle_id}.json"
    if not puzzle_path.exists():
        raise HTTPException(404, "No puzzle found for this session")

    queue: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()
    progress_queues[session_id] = queue
    cancel_events[session_id] = cancel_event
    background_tasks.add_task(
        pipeline.run_solve_background, session_dir, PUZZLES_DIR, puzzle_id, queue, session_mgr, session_id,
        cancel_event,
    )

    return {"status": "solve_started", "session_id": session_id, "puzzle_id": puzzle_id}


@app.post("/api/{session_id}/cancel")
async def cancel_solve(session_id: str):
    """Cancel a running background solve."""
    event = cancel_events.get(session_id)
    if not event:
        raise HTTPException(404, "No active solve to cancel")
    event.set()
    return {"status": "cancelling", "session_id": session_id}


@app.get("/api/{session_id}/progress")
async def stream_progress(session_id: str):
    queue = progress_queues.get(session_id)
    if not queue:
        status = session_mgr.get_status(session_id)
        if status == SessionStatus.COMPLETE:
            async def done():
                yield f'data: {{"stage":"complete","message":"Puzzle ready!","progress":1.0}}\n\n'
            return StreamingResponse(done(), media_type="text/event-stream")
        if status in (SessionStatus.OCR_RUNNING, SessionStatus.SOLVING, SessionStatus.GENERATING_HINTS):
            async def starting():
                yield f'data: {{"stage":"heartbeat","message":"Pipeline starting...","progress":0}}\n\n'
            return StreamingResponse(starting(), media_type="text/event-stream")
        raise HTTPException(404, "No active solve for this session")

    async def event_generator():
        while True:
            try:
                progress = await asyncio.wait_for(queue.get(), timeout=120)
                yield f"data: {progress.model_dump_json()}\n\n"
                if progress.stage in ("complete", "failed", "verification_failed", "cancelled"):
                    progress_queues.pop(session_id, None)
                    cancel_events.pop(session_id, None)
                    break
            except asyncio.TimeoutError:
                yield f'data: {{"stage":"heartbeat","message":"Still working...","progress":-1}}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/{session_id}/diagnostics")
async def get_diagnostics(session_id: str):
    """Return per-clue solve diagnostics (candidates, sources, scores)."""
    session_dir = session_mgr.get_session_dir(session_id)
    diag_path = session_dir / "solve_diagnostics.json"
    if not diag_path.exists():
        raise HTTPException(404, "No diagnostics available — solve has not run yet")
    with open(diag_path) as f:
        return json.load(f)


@app.get("/api/{session_id}/status", response_model=SolveStatusResponse)
async def get_session_status(session_id: str):
    data = session_mgr.get_session_data(session_id)
    return SolveStatusResponse(
        status=SessionStatus(data["status"]),
        solved_count=data.get("solved_count"),
        total_clues=data.get("total_clues"),
    )


@app.get("/api/puzzles")
async def list_puzzles():
    """List all available puzzle JSONs."""
    PUZZLES_DIR.mkdir(parents=True, exist_ok=True)
    puzzles = []
    for p in sorted(PUZZLES_DIR.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        clues = data.get("clues", {})
        across = clues.get("across", [])
        down = clues.get("down", [])
        total = len(across) + len(down)
        solved = sum(1 for c in across + down if c.get("answer") and "?" not in c["answer"])
        puzzles.append({
            "id": p.stem,
            "title": meta.get("name") or meta.get("source_image", p.stem).replace(".JPG", "").replace("IMG_", "Puzzle #"),
            "gridSize": meta.get("grid_size", [0, 0]),
            "totalClues": total,
            "solved": solved,
        })
    return puzzles


@app.patch("/api/puzzles/{puzzle_id}")
async def update_puzzle(puzzle_id: str, body: dict):
    """Update puzzle metadata (e.g. name)."""
    puzzle_path = PUZZLES_DIR / f"{puzzle_id}.json"
    if not puzzle_path.exists():
        raise HTTPException(404, "Puzzle not found")
    with open(puzzle_path) as f:
        data = json.load(f)
    if "name" in body:
        data.setdefault("metadata", {})["name"] = body["name"]
    with open(puzzle_path, "w") as f:
        json.dump(data, f, indent=2)
    return {"ok": True}


def main():
    uvicorn.run(
        "src.api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )


if __name__ == "__main__":
    main()

"""
FastAPI server for Crosswise photo upload pipeline.

Usage:
    .venv/bin/python3 -m src.api.server
"""

import asyncio
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
    SolveStatusResponse,
    GridEditRequest,
    GridEditResponse,
)
from src.api.session_manager import SessionManager
from src.api import pipeline

SESSIONS_DIR = settings.DATA_DIR / "sessions"
PUZZLES_DIR = Path(__file__).parent.parent.parent / "web" / "public" / "puzzles"

session_mgr = SessionManager(SESSIONS_DIR)

# Track background solve progress per session
progress_queues: dict[str, asyncio.Queue] = {}

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
    progress_queues[session_id] = queue
    background_tasks.add_task(
        pipeline.run_solve_background, session_dir, PUZZLES_DIR, puzzle_id, queue, session_mgr, session_id
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


@app.post("/api/{session_id}/solve")
async def retrigger_solve(session_id: str, background_tasks: BackgroundTasks):
    """Re-trigger the solve for an existing session (e.g., after pipeline fix)."""
    session_dir = session_mgr.get_session_dir(session_id)
    puzzle_json = session_dir / "puzzle.json"
    if not puzzle_json.exists():
        raise HTTPException(404, "No puzzle.json found for this session")

    puzzle_id = session_mgr.get_session_data(session_id).get("puzzle_id", session_id)

    queue: asyncio.Queue = asyncio.Queue()
    progress_queues[session_id] = queue
    background_tasks.add_task(
        pipeline.run_solve_background, session_dir, PUZZLES_DIR, puzzle_id, queue, session_mgr, session_id
    )

    return {"status": "solve_started", "session_id": session_id, "puzzle_id": puzzle_id}


@app.get("/api/{session_id}/progress")
async def stream_progress(session_id: str):
    queue = progress_queues.get(session_id)
    if not queue:
        # Check if already complete
        status = session_mgr.get_status(session_id)
        if status == SessionStatus.COMPLETE:
            async def done():
                yield f'data: {{"stage":"complete","message":"Puzzle ready!","progress":1.0}}\n\n'
            return StreamingResponse(done(), media_type="text/event-stream")
        raise HTTPException(404, "No active solve for this session")

    async def event_generator():
        while True:
            try:
                progress = await asyncio.wait_for(queue.get(), timeout=120)
                yield f"data: {progress.model_dump_json()}\n\n"
                if progress.stage in ("complete", "failed"):
                    progress_queues.pop(session_id, None)
                    break
            except asyncio.TimeoutError:
                yield f'data: {{"stage":"heartbeat","message":"Still working...","progress":-1}}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
        import json
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
    import json as _json
    puzzle_path = PUZZLES_DIR / f"{puzzle_id}.json"
    if not puzzle_path.exists():
        raise HTTPException(404, "Puzzle not found")
    with open(puzzle_path) as f:
        data = _json.load(f)
    if "name" in body:
        data.setdefault("metadata", {})["name"] = body["name"]
    with open(puzzle_path, "w") as f:
        _json.dump(data, f, indent=2)
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

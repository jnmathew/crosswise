"""FastAPI server for the Crosswise crossword puzzle app."""

import asyncio
from contextlib import asynccontextmanager
import json
import shutil
import threading
from functools import partial
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from crosswise.config import settings
from crosswise.api.models import (
    SessionStatus,
    SolveProgress,
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
from crosswise.api.session_manager import SessionManager, SessionNotFound
from crosswise.api.rate_limit import rate_limited
from crosswise.api.storage import write_json_atomic
from crosswise.api import pipeline
from crosswise.vision.grid_detection import MAX_GRID_DIM, MIN_GRID_DIM

SESSIONS_DIR = settings.DATA_DIR / "sessions"
PUZZLES_DIR = settings.PROJECT_ROOT / "web" / "public" / "puzzles"

session_mgr = SessionManager(SESSIONS_DIR)

# Track background solve progress per session
progress_queues: dict[str, asyncio.Queue] = {}
cancel_events: dict[str, threading.Event] = {}


def _run_tracked(task, session_id: str, queue: asyncio.Queue,
                 cancel_event: threading.Event):
    """Run a background pipeline task, then drop its progress-tracking entries.

    Cleanup must happen on the producer side: if no SSE consumer ever drains
    the terminal message (browser closed mid-solve), the queue/cancel-event
    entries would otherwise leak forever. A consumer that is already streaming
    holds its own reference to the queue, so it can still drain the terminal
    message after this cleanup runs. The identity checks avoid removing the
    entries of a newer solve that re-registered under the same session_id.
    """
    try:
        task()
    finally:
        if progress_queues.get(session_id) is queue:
            progress_queues.pop(session_id, None)
        if cancel_events.get(session_id) is cancel_event:
            cancel_events.pop(session_id, None)



def _start_tracked(session_id: str, background_tasks: BackgroundTasks, make_task):
    """Register progress tracking for a session and schedule its background task.

    ``make_task(queue, cancel_event, loop)`` returns the zero-arg callable to run.
    Refuses (409) while a solve is already active for the session: two solves
    would race to write the same puzzle JSON. The check and the registration
    both run on the event loop thread, so they can't interleave.
    """
    if session_id in cancel_events:
        raise HTTPException(409, "A solve is already running for this puzzle")
    queue: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()
    loop = asyncio.get_running_loop()
    progress_queues[session_id] = queue
    cancel_events[session_id] = cancel_event
    background_tasks.add_task(
        _run_tracked, make_task(queue, cancel_event, loop), session_id, queue, cancel_event,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Solve tracking is in memory, so anything still "running" at startup was
    # killed by the restart (including uvicorn --reload on a code change).
    for session_id in session_mgr.mark_interrupted():
        logger.warning(f"Session {session_id} was interrupted by a server restart; marked failed")
    yield


app = FastAPI(title="Crosswise API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(SessionNotFound)
async def session_not_found(request: Request, exc: SessionNotFound):
    return JSONResponse(status_code=404, content={"detail": "Session not found"})

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


# Leading bytes of the image formats OpenCV decodes here
_IMAGE_SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")


def _check_upload_size(request: Request) -> None:
    """Reject oversized uploads from Content-Length before the body is parsed."""
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image must be under {settings.MAX_UPLOAD_MB} MB")


@app.post(
    "/api/upload", response_model=UploadResponse,
    dependencies=[Depends(_check_upload_size), Depends(rate_limited("upload"))],
)
async def upload_photo(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG or PNG)")

    # Content-Length can be absent (chunked uploads), so enforce the cap on
    # the bytes actually read too.
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(413, f"Image must be under {settings.MAX_UPLOAD_MB} MB")
    if not content.startswith(_IMAGE_SIGNATURES):
        raise HTTPException(400, "File must be a JPEG or PNG image")

    session_id = session_mgr.create_session()
    session_dir = session_mgr.get_session_dir(session_id)

    # Save uploaded file
    original_path = session_dir / "original.jpg"
    with open(original_path, "wb") as f:
        f.write(content)

    # Grid detection is seconds of CPU work; run it in a worker thread so it
    # doesn't stall the event loop (and every live progress stream).
    try:
        result = await run_in_threadpool(pipeline.run_grid_detection, session_dir, settings)
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
    if not (MIN_GRID_DIM <= req.rows <= MAX_GRID_DIM and MIN_GRID_DIM <= req.cols <= MAX_GRID_DIM):
        raise HTTPException(400, f"Rows and cols must be between {MIN_GRID_DIM} and {MAX_GRID_DIM}")
    try:
        result = await run_in_threadpool(pipeline.resize_grid, session_dir, req.rows, req.cols)
    except Exception as e:
        raise HTTPException(422, f"Grid resize failed: {e}")

    return GridResizeResponse(
        grid_size=result["grid_size"],
        clue_slot_count=result["clue_slot_count"],
        black_cells=result["black_cells"],
    )


@app.post("/api/{session_id}/manual-crop", response_model=UploadResponse,
          dependencies=[Depends(rate_limited("upload"))])
async def manual_crop(session_id: str, req: ManualCropRequest):
    """Re-run grid detection with user-specified quad corners."""
    session_dir = session_mgr.get_session_dir(session_id)
    if not (session_dir / "original.jpg").exists():
        raise HTTPException(404, "No original image found for this session")

    if len(req.corners) != 4 or any(len(c) != 2 for c in req.corners):
        raise HTTPException(400, "corners must be exactly 4 points, each [x, y]")

    try:
        result = await run_in_threadpool(
            pipeline.run_grid_detection, session_dir, settings, manual_quad=req.corners,
        )
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


@app.post("/api/{session_id}/mask", response_model=MaskResponse,
          dependencies=[Depends(rate_limited("solve"))])
async def submit_mask(session_id: str, mask: MaskRequest, background_tasks: BackgroundTasks):
    session_dir = session_mgr.get_session_dir(session_id)
    if session_id in cancel_events:
        # Checked before OCR so a rejected request doesn't spend a Gemini call
        raise HTTPException(409, "A solve is already running for this puzzle")
    session_mgr.update_status(session_id, SessionStatus.OCR_RUNNING)

    try:
        # OCR is a multi-second network call; keep it off the event loop
        result = await run_in_threadpool(pipeline.run_ocr_and_verify, session_dir, mask, settings)
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
    _start_tracked(session_id, background_tasks, lambda queue, cancel_event, loop: partial(
        pipeline.run_solve_background,
        session_dir, PUZZLES_DIR, puzzle_id, queue, session_mgr, session_id, cancel_event, loop,
    ))

    return MaskResponse(
        status=SessionStatus.VERIFIED,
        verification_passed=True,
        ocr_clue_count=result["ocr_clue_count"],
        grid_slot_count=result["grid_slot_count"],
        matched_count=result["matched_count"],
        errors=[],
        puzzle_id=puzzle_id,
    )


@app.post("/api/{session_id}/start-pipeline", response_model=StartPipelineResponse,
          dependencies=[Depends(rate_limited("solve"))])
async def start_pipeline(session_id: str, mask: MaskRequest, background_tasks: BackgroundTasks):
    """Start the full OCR + solve pipeline in the background.

    Creates a skeleton puzzle immediately so the player can load right away,
    then runs OCR, verification, solving, and hint generation as a background task.
    """
    session_dir = session_mgr.get_session_dir(session_id)
    puzzle_id = session_id
    if session_id in cancel_events:
        raise HTTPException(409, "A solve is already running for this puzzle")

    # Build skeleton puzzle so the player has something to load immediately
    pipeline.build_skeleton_puzzle(session_dir, PUZZLES_DIR, puzzle_id)

    # Create progress queue and start background pipeline
    _start_tracked(session_id, background_tasks, lambda queue, cancel_event, loop: partial(
        pipeline.run_full_pipeline_background,
        session_dir, PUZZLES_DIR, puzzle_id, mask, settings,
        queue, session_mgr, session_id, cancel_event, loop,
    ))

    session_mgr.update_status(session_id, SessionStatus.OCR_RUNNING, puzzle_id=puzzle_id)

    return StartPipelineResponse(
        session_id=session_id,
        puzzle_id=puzzle_id,
        status=SessionStatus.OCR_RUNNING,
    )


@app.post("/api/{session_id}/solve", dependencies=[Depends(rate_limited("solve"))])
async def retrigger_solve(session_id: str, background_tasks: BackgroundTasks):
    """Re-trigger the solve for an existing session (e.g., after pipeline fix)."""
    session_dir = session_mgr.get_session_dir(session_id)
    puzzle_id = session_mgr.get_session_data(session_id).get("puzzle_id", session_id)
    puzzle_path = PUZZLES_DIR / f"{puzzle_id}.json"
    if not puzzle_path.exists():
        raise HTTPException(404, "No puzzle found for this session")

    _start_tracked(session_id, background_tasks, lambda queue, cancel_event, loop: partial(
        pipeline.run_solve_background,
        session_dir, PUZZLES_DIR, puzzle_id, queue, session_mgr, session_id, cancel_event, loop,
    ))

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
                yield 'data: {"stage":"complete","message":"Puzzle ready!","progress":1.0}\n\n'
            return StreamingResponse(done(), media_type="text/event-stream")
        if status in (SessionStatus.OCR_RUNNING, SessionStatus.SOLVING, SessionStatus.GENERATING_HINTS):
            async def starting():
                yield 'data: {"stage":"heartbeat","message":"Pipeline starting...","progress":0}\n\n'
            return StreamingResponse(starting(), media_type="text/event-stream")
        raise HTTPException(404, "No active solve for this session")

    async def event_generator():
        # A viewer joining mid-solve (e.g. after a page reload) first gets the
        # answers so far, then the live events that follow.
        feed = pipeline.LIVE_FEEDS.get(session_id)
        if feed is not None:
            snapshot = feed.snapshot()
            if snapshot:
                event = SolveProgress(stage="live", message="", progress=-1,
                                      live={"type": "snapshot", "answers": snapshot})
                yield f"data: {event.model_dump_json()}\n\n"
        while True:
            try:
                progress = await asyncio.wait_for(queue.get(), timeout=120)
                yield f"data: {progress.model_dump_json()}\n\n"
                if progress.stage in ("complete", "failed", "verification_failed", "cancelled"):
                    # Guard by identity: a retriggered solve may have
                    # re-registered fresh entries under this session_id.
                    if progress_queues.get(session_id) is queue:
                        progress_queues.pop(session_id, None)
                        cancel_events.pop(session_id, None)
                    break
            except asyncio.TimeoutError:
                yield 'data: {"stage":"heartbeat","message":"Still working...","progress":-1}\n\n'

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
        try:
            with open(p) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            # One damaged file shouldn't take down the whole list
            logger.warning(f"Skipping unreadable puzzle {p.name}: {e}")
            continue
        meta = data.get("metadata", {})
        clues = data.get("clues", {})
        across = clues.get("across", [])
        down = clues.get("down", [])
        total = len(across) + len(down)
        solved = sum(1 for c in across + down if c.get("answer") and "?" not in c["answer"])
        # Build compact grid mask (list of lists of booleans) for thumbnail
        grid_data = data.get("grid", {})
        cells = grid_data.get("cells", [])
        grid_mask = [[c.get("is_block", False) for c in row] for row in cells] if cells else []

        puzzles.append({
            "id": p.stem,
            "title": meta.get("name") or meta.get("source_image", p.stem).replace(".JPG", "").replace("IMG_", "Puzzle #"),
            "gridSize": meta.get("grid_size", [0, 0]),
            "totalClues": total,
            "solved": solved,
            "gridMask": grid_mask,
        })
    return puzzles


def _puzzle_path(puzzle_id: str) -> Path:
    """Resolve a puzzle_id to its JSON path, rejecting IDs that escape PUZZLES_DIR."""
    path = (PUZZLES_DIR / f"{puzzle_id}.json").resolve()
    if not path.is_relative_to(PUZZLES_DIR.resolve()):
        raise HTTPException(400, "Invalid puzzle ID")
    if not path.exists():
        raise HTTPException(404, "Puzzle not found")
    return path


@app.patch("/api/puzzles/{puzzle_id}")
async def update_puzzle(puzzle_id: str, body: dict):
    """Update puzzle metadata (e.g. name)."""
    puzzle_path = _puzzle_path(puzzle_id)
    with open(puzzle_path) as f:
        data = json.load(f)
    if "name" in body:
        data.setdefault("metadata", {})["name"] = body["name"]
    write_json_atomic(puzzle_path, data, indent=2)
    return {"ok": True}


@app.delete("/api/puzzles/{puzzle_id}")
async def delete_puzzle(puzzle_id: str):
    """Delete a puzzle and its session data."""
    puzzle_path = _puzzle_path(puzzle_id)
    if puzzle_id in cancel_events:
        # The solve's final write would recreate the file
        raise HTTPException(409, "This puzzle is still solving; cancel the solve first")
    puzzle_path.unlink()
    try:
        # Puzzle IDs are session IDs; without this, session dirs (photos,
        # OCR output, diagnostics) accumulate forever.
        shutil.rmtree(session_mgr.get_session_dir(puzzle_id))
    except SessionNotFound:
        pass  # e.g. the demo puzzle, which has no session
    return {"ok": True}


def main():
    uvicorn.run(
        "crosswise.api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        reload_excludes=[".venv"],
    )


if __name__ == "__main__":
    main()

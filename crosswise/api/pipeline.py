"""
Pipeline orchestration for the Crosswise API.
Wraps existing core functions for use in FastAPI endpoints.
"""

import asyncio
from datetime import datetime
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from loguru import logger
import numpy as np
from crosswise.config import Settings
from crosswise.api.models import MaskRequest, SolveProgress, SessionStatus


class SolveCancelled(Exception):
    """Raised when a solve is cancelled by the user."""


def _check_cancel(cancel_event: Optional[threading.Event]):
    """Raise SolveCancelled if the cancel event is set."""
    if cancel_event and cancel_event.is_set():
        raise SolveCancelled("Solve cancelled by user")


def _serialize_and_save_grid(cells, clue_slots, session_dir: Path) -> Dict[str, Any]:
    """Serialize grid cells to JSON and save to session directory.

    Returns dict with rows, cols, cells_data for callers that need the raw data.
    """
    rows = len(cells)
    cols = len(cells[0]) if rows > 0 else 0

    cells_data = []
    for r in range(rows):
        row_data = []
        for c in range(cols):
            cell = cells[r][c]
            row_data.append({
                "row": cell.row,
                "col": cell.col,
                "is_block": cell.is_block,
                "clue_number": cell.clue_number,
            })
        cells_data.append(row_data)

    grid_data = {
        "rows": rows,
        "cols": cols,
        "cells": cells_data,
        "clue_slots": clue_slots,
    }

    with open(session_dir / "grid_data.json", "w") as f:
        json.dump(grid_data, f, indent=2)

    return {"rows": rows, "cols": cols, "cells_data": cells_data}


def run_grid_detection(
    session_dir: Path,
    config: Settings,
    manual_quad: Any = None,
) -> Dict[str, Any]:
    """Phase 1: Preprocess image and detect grid.

    Args:
        manual_quad: Optional user-specified quad corners (list of [x, y] pairs).
                     If provided, uses these instead of auto-detection.
    """
    from crosswise.vision.image_preprocessing import preprocess
    from crosswise.vision.grid_detection import detect_grid, assign_clue_numbers, compute_clue_slots

    original_path = session_dir / "original.jpg"

    if manual_quad is not None and not isinstance(manual_quad, np.ndarray):
        manual_quad = np.array(manual_quad, dtype="float32")

    result = preprocess(original_path, config, manual_quad=manual_quad)

    cv2.imwrite(str(session_dir / "warped.jpg"), result["warped"])
    cv2.imwrite(str(session_dir / "warped_gray.jpg"), result["warped_gray"])

    grid_result = detect_grid(result["warped_gray"], config)
    cells = grid_result["cells"]
    assign_clue_numbers(cells)
    clue_slots = compute_clue_slots(cells)

    info = _serialize_and_save_grid(cells, clue_slots, session_dir)

    return {
        "grid_size": (info["rows"], info["cols"]),
        "clue_slot_count": len(clue_slots),
    }


def apply_grid_edit(session_dir: Path, black_cells: list[list[bool]]) -> Dict[str, Any]:
    """Recompute grid clue numbers and slots from user-edited black cell map."""
    from crosswise.vision.grid_detection import assign_clue_numbers, compute_clue_slots
    from crosswise.models import Cell

    rows = len(black_cells)
    cols = len(black_cells[0]) if rows > 0 else 0

    cells = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(Cell(row=r, col=c, is_block=black_cells[r][c]))
        cells.append(row)

    assign_clue_numbers(cells)
    clue_slots = compute_clue_slots(cells)

    info = _serialize_and_save_grid(cells, clue_slots, session_dir)

    clue_number_count = max(
        (cell.clue_number for row in cells for cell in row if cell.clue_number is not None),
        default=0,
    )

    return {
        "grid_size": (info["rows"], info["cols"]),
        "clue_slot_count": len(clue_slots),
        "clue_number_count": clue_number_count,
    }


def resize_grid(session_dir: Path, rows: int, cols: int) -> Dict[str, Any]:
    """Re-detect black cells at new grid dimensions using evenly-spaced lines."""
    from crosswise.vision.grid_detection import classify_black_cells, assign_clue_numbers, compute_clue_slots

    warped_path = session_dir / "warped_gray.jpg"
    warped_gray = cv2.imread(str(warped_path), cv2.IMREAD_GRAYSCALE)
    if warped_gray is None:
        raise FileNotFoundError("warped_gray.jpg not found in session directory")

    h, w = warped_gray.shape[:2]

    xs = [round(i * w / cols) for i in range(cols + 1)]
    ys = [round(i * h / rows) for i in range(rows + 1)]

    cells = classify_black_cells(warped_gray, xs, ys)
    assign_clue_numbers(cells)
    clue_slots = compute_clue_slots(cells)

    _serialize_and_save_grid(cells, clue_slots, session_dir)

    # Also return black_cells map for the GridEditor UI
    black_cells = [
        [cells[r][c].is_block for c in range(cols)]
        for r in range(rows)
    ]

    return {
        "grid_size": [rows, cols],
        "clue_slot_count": len(clue_slots),
        "black_cells": black_cells,
    }


def apply_masks(image: np.ndarray, mask: MaskRequest) -> np.ndarray:
    """Apply white rectangles and aqua separator lines to image."""
    img = image.copy()
    for rect in mask.rectangles:
        cv2.rectangle(img, (rect.x1, rect.y1), (rect.x2, rect.y2), (255, 255, 255), -1)
    for sep in mask.separators:
        cv2.line(img, (sep.x1, sep.y1), (sep.x2, sep.y2), (255, 255, 0), 8)
    return img


def run_ocr_and_verify(session_dir: Path, mask: MaskRequest, config: Settings) -> Dict[str, Any]:
    """Phase 2: Apply masks, run OCR, verify against grid."""
    from crosswise.vision.clue_extraction import parse_ocr_markdown, verify_puzzle

    # Load original image and apply masks
    original = cv2.imread(str(session_dir / "original.jpg"))
    masked = apply_masks(original, mask)
    cv2.imwrite(str(session_dir / "masked.jpg"), masked)

    # Run OCR
    from crosswise.ocr.base import create_ocr_provider

    ocr_provider = create_ocr_provider(config)
    ocr_markdown = ocr_provider.extract_clues(session_dir / "masked.jpg")
    with open(session_dir / "ocr_result.md", "w") as f:
        f.write(ocr_markdown)

    # Parse OCR output
    ocr_clues, warnings = parse_ocr_markdown(ocr_markdown)

    # Load grid slots
    with open(session_dir / "grid_data.json") as f:
        grid_data = json.load(f)
    grid_slots = grid_data["clue_slots"]

    # Verify
    success, matched, report = verify_puzzle(ocr_clues, grid_slots)

    # Save verification data
    with open(session_dir / "verification.json", "w") as f:
        json.dump({"success": success, "report": report, "matched_count": len(matched)}, f, indent=2)

    if success:
        # Save matched clues for later puzzle building
        with open(session_dir / "matched_clues.json", "w") as f:
            json.dump(matched, f, indent=2)

    return {
        "verification_passed": success,
        "ocr_clue_count": report["ocr_clue_count"],
        "grid_slot_count": report["grid_slot_count"],
        "matched_count": report.get("matched_count", 0),
        "errors": report.get("errors", []) + report.get("duplicate_errors", []),
    }


def build_preliminary_puzzle(session_dir: Path, puzzles_dir: Path, puzzle_id: str):
    """Build preliminary puzzle JSON (no answers) and save to puzzles dir."""
    with open(session_dir / "grid_data.json") as f:
        grid_data = json.load(f)
    with open(session_dir / "matched_clues.json") as f:
        matched_clues = json.load(f)

    # Build puzzle JSON in the same format as existing puzzles
    clues_across = []
    clues_down = []
    for clue in matched_clues:
        entry = {
            "number": clue["number"],
            "text": clue["text"],
            "start": list(clue["start"]),
            "length": clue["length"],
            "answer": None,
            "hint": None,
            "explanation": None,
        }
        if clue["direction"] == "across":
            clues_across.append(entry)
        else:
            clues_down.append(entry)

    clues_across.sort(key=lambda c: c["number"])
    clues_down.sort(key=lambda c: c["number"])

    puzzle = {
        "metadata": {
            "source_image": "uploaded",
            "grid_size": [grid_data["rows"], grid_data["cols"]],
            "total_clues": len(matched_clues),
            "verification": "PASSED",
            "name": f"Puzzle {datetime.now().strftime('%-m/%-d %I:%M %p')}",
        },
        "grid": {
            "rows": grid_data["rows"],
            "cols": grid_data["cols"],
            "cells": grid_data["cells"],
        },
        "clues": {
            "across": clues_across,
            "down": clues_down,
        },
    }

    puzzles_dir.mkdir(parents=True, exist_ok=True)
    puzzle_path = puzzles_dir / f"{puzzle_id}.json"
    with open(puzzle_path, "w") as f:
        json.dump(puzzle, f, indent=2)

    # Also save to session dir for solve step
    with open(session_dir / "puzzle.json", "w") as f:
        json.dump(puzzle, f, indent=2)


def build_skeleton_puzzle(session_dir: Path, puzzles_dir: Path, puzzle_id: str):
    """Build grid-only skeleton puzzle JSON with placeholder clues.

    Creates a puzzle the player can load immediately while OCR runs in the background.
    Clue slots are derived from grid_data.json with text="..." and no answers.
    """
    with open(session_dir / "grid_data.json") as f:
        grid_data = json.load(f)

    clues_across = []
    clues_down = []
    for slot in grid_data["clue_slots"]:
        entry = {
            "number": slot["number"],
            "text": "...",
            "start": slot["start"],
            "length": slot["length"],
            "answer": None,
            "hint": None,
            "explanation": None,
        }
        if slot["direction"] == "across":
            clues_across.append(entry)
        else:
            clues_down.append(entry)

    clues_across.sort(key=lambda c: c["number"])
    clues_down.sort(key=lambda c: c["number"])

    puzzle = {
        "metadata": {
            "source_image": "uploaded",
            "grid_size": [grid_data["rows"], grid_data["cols"]],
            "total_clues": len(grid_data["clue_slots"]),
            "verification": "PENDING",
            "name": f"Puzzle {datetime.now().strftime('%-m/%-d %I:%M %p')}",
        },
        "grid": {
            "rows": grid_data["rows"],
            "cols": grid_data["cols"],
            "cells": grid_data["cells"],
        },
        "clues": {
            "across": clues_across,
            "down": clues_down,
        },
    }

    puzzles_dir.mkdir(parents=True, exist_ok=True)
    with open(puzzles_dir / f"{puzzle_id}.json", "w") as f:
        json.dump(puzzle, f, indent=2)


def run_full_pipeline_background(
    session_dir: Path,
    puzzles_dir: Path,
    puzzle_id: str,
    mask: "MaskRequest",
    config: "Settings",
    queue: asyncio.Queue,
    session_mgr: Any,
    session_id: str,
    cancel_event: Optional[threading.Event] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
):
    """Background task: OCR + verify + solve + hints, with SSE progress throughout.

    Runs in a worker thread. `loop` must be the event loop that owns `queue`
    (the server's main loop): asyncio.Queue is not thread-safe, so all puts are
    marshalled onto that loop via call_soon_threadsafe. The queue is unbounded,
    so put_nowait cannot raise QueueFull.
    """

    def put(progress: SolveProgress):
        loop.call_soon_threadsafe(queue.put_nowait, progress)

    try:
        # Phase 1: OCR
        session_mgr.update_status(session_id, SessionStatus.OCR_RUNNING)
        put(SolveProgress(stage="ocr", message="Extracting clues from image...", progress=0.02))

        _check_cancel(cancel_event)
        result = run_ocr_and_verify(session_dir, mask, config)

        if not result["verification_passed"]:
            errors = result.get("errors", [])
            session_mgr.update_status(session_id, SessionStatus.VERIFICATION_FAILED)
            put(SolveProgress(
                stage="verification_failed",
                message=f"Verification failed: {result['matched_count']}/{result['grid_slot_count']} slots matched. "
                        f"OCR found {result['ocr_clue_count']} clues. "
                        + ("; ".join(errors) if errors else ""),
                progress=0,
            ))
            return

        _check_cancel(cancel_event)

        # Phase 2: Build full puzzle with clues
        put(SolveProgress(stage="verification", message="Clues verified! Building puzzle...", progress=0.05))
        build_preliminary_puzzle(session_dir, puzzles_dir, puzzle_id)
        session_mgr.update_status(session_id, SessionStatus.VERIFIED, puzzle_id=puzzle_id)

        # Signal frontend to refetch for real clues
        put(SolveProgress(stage="clues_ready", message="Clues loaded", progress=0.06))

        # Phase 3: Solve (reuses existing _run_solve)
        _run_solve(session_dir, puzzles_dir, puzzle_id, put, session_mgr, session_id, cancel_event)

    except SolveCancelled:
        put(SolveProgress(stage="cancelled", message="Solve cancelled", progress=0))
        session_mgr.update_status(session_id, SessionStatus.FAILED, error="Cancelled by user")
    except Exception as e:
        put(SolveProgress(stage="failed", message=str(e), progress=0))
        session_mgr.update_status(session_id, SessionStatus.FAILED, error=str(e))


def run_solve_background(
    session_dir: Path,
    puzzles_dir: Path,
    puzzle_id: str,
    queue: asyncio.Queue,
    session_mgr: Any,
    session_id: str,
    cancel_event: Optional[threading.Event] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
):
    """Background task: solve puzzle + generate hints, updating puzzle JSON in-place.

    Runs in a worker thread. `loop` must be the event loop that owns `queue`
    (the server's main loop) — see run_full_pipeline_background.
    """

    def put(progress: SolveProgress):
        loop.call_soon_threadsafe(queue.put_nowait, progress)

    try:
        _run_solve(session_dir, puzzles_dir, puzzle_id, put, session_mgr, session_id, cancel_event)
    except SolveCancelled:
        put(SolveProgress(stage="cancelled", message="Solve cancelled", progress=0))
        session_mgr.update_status(session_id, SessionStatus.FAILED, error="Cancelled by user")
    except Exception as e:
        put(SolveProgress(stage="failed", message=str(e), progress=0))
        session_mgr.update_status(session_id, SessionStatus.FAILED, error=str(e))


def _generate_candidates(clue_inputs, put_progress, _elapsed, cancel_event=None, solve_trace=None, trace_global=None):
    """Generate, pad, and score candidate answers for all clues.

    Returns (candidates, scored, score_map, candidate_scores, candidate_sources, web_candidates, db).
    """
    from crosswise.solver.candidates import (
        generate_candidates_with_database,
        generate_with_claude,
        bouncer_filter,
        to_plain_candidates,
        to_score_map,
        web_search_prepass,
    )
    from crosswise.solver.clue_database import ClueDatabase

    # Build clue text lookup for bouncer filter
    clue_text_lookup = {}
    for ci in clue_inputs:
        clue_text_lookup[ci.clue_id] = ci.text

    # Database lookup
    _check_cancel(cancel_event)
    logger.info(f"{_elapsed()} Starting candidate generation")
    put_progress(SolveProgress(stage="candidates", message="Database lookup...", progress=0.1))
    db = ClueDatabase()
    candidates = generate_candidates_with_database(
        clue_inputs, db=db, candidates_per_clue=12
    )

    # Track which candidates came from DB vs LLM
    candidate_sources: Dict[str, Dict[str, str]] = {}
    for cid, words in candidates.items():
        candidate_sources[cid] = {w.upper(): "db" for w in words}

    # Trace: DB lookup results
    if solve_trace:
        db_hits = sum(1 for ci in clue_inputs if candidates.get(ci.clue_id))
        for ci in clue_inputs:
            count = len(candidates.get(ci.clue_id, []))
            solve_trace[ci.clue_id]["events"].append({"phase": "db_lookup", "candidates": count})
        if trace_global:
            trace_global("db_lookup", f"{db_hits}/{len(clue_inputs)} clues had DB hits")

    # Web search pre-pass
    _check_cancel(cancel_event)
    logger.info(f"{_elapsed()} Running web search pre-pass")
    put_progress(SolveProgress(stage="candidates", message="Web search pre-pass...", progress=0.12))
    web_candidates = web_search_prepass(clue_inputs)

    for cid, word in web_candidates.items():
        existing = candidates.get(cid, [])
        if word not in [w.upper() for w in existing]:
            existing.append(word)
            candidates[cid] = existing
        candidate_sources.setdefault(cid, {})
        candidate_sources[cid][word.upper()] = "web"

    # Trace: web search results
    if solve_trace:
        for ci in clue_inputs:
            cid = ci.clue_id
            if cid in web_candidates:
                solve_trace[cid]["events"].append({"phase": "web_search", "result": web_candidates[cid]})
        if trace_global:
            trace_global("web_search", f"{len(web_candidates)} clues got web results")

    # Parallel Opus (zero-hit) + Sonnet padding (<5 candidates)
    _check_cancel(cancel_event)
    clues_needing_llm = [c for c in clue_inputs if not candidates.get(c.clue_id)]
    clues_needing_pad = [c for c in clue_inputs if 0 < len(candidates.get(c.clue_id, [])) < 5]

    put_progress(SolveProgress(
        stage="candidates",
        message=f"Generating candidates ({len(clues_needing_llm)} Opus + {len(clues_needing_pad)} Sonnet padding)...",
        progress=0.15,
    ))

    def _run_opus():
        if clues_needing_llm:
            return generate_with_claude(clues_needing_llm, candidates_per_clue=12)
        return {}

    def _run_sonnet_pad():
        if clues_needing_pad:
            return generate_with_claude(clues_needing_pad, candidates_per_clue=15, model="claude-sonnet-4-6")
        return {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        opus_future = executor.submit(_run_opus)
        sonnet_future = executor.submit(_run_sonnet_pad)
        opus_cands = opus_future.result()
        sonnet_cands = sonnet_future.result()

    # Merge Opus candidates
    candidates.update(opus_cands)
    for cid, words in opus_cands.items():
        candidate_sources.setdefault(cid, {})
        for w in words:
            candidate_sources[cid].setdefault(w.upper(), "llm")
    if clues_needing_llm:
        logger.info(f"  {_elapsed()} Opus generated: {sum(1 for v in opus_cands.values() if v)}/{len(clues_needing_llm)} clues")

    # Merge Sonnet padding candidates
    for cid, words in sonnet_cands.items():
        existing = set(candidates.get(cid, []))
        for w in words:
            if w not in existing:
                existing.add(w)
        candidates[cid] = list(existing)
        candidate_sources.setdefault(cid, {})
        for w in words:
            candidate_sources[cid].setdefault(w.upper(), "sonnet")
    if clues_needing_pad:
        logger.info(f"  {_elapsed()} Sonnet padded: {len(clues_needing_pad)} clues")

    # Trace: LLM generation results
    if solve_trace:
        opus_set = set(opus_cands.keys())
        sonnet_set = set(sonnet_cands.keys())
        for ci in clue_inputs:
            cid = ci.clue_id
            if cid in opus_set:
                solve_trace[cid]["events"].append({"phase": "opus", "candidates": len(opus_cands[cid])})
            if cid in sonnet_set:
                solve_trace[cid]["events"].append({"phase": "sonnet_pad", "candidates": len(sonnet_cands[cid])})
        if trace_global:
            trace_global("llm_generation", f"{len(opus_set)} Opus + {len(sonnet_set)} Sonnet padding")

    # Second-pass: catch any clues still under 5
    _check_cancel(cancel_event)
    still_under = [c for c in clue_inputs if len(candidates.get(c.clue_id, [])) < 5]
    if still_under:
        logger.info(f"  {_elapsed()} Second-pass padding for {len(still_under)} clues still under 5")
        extra = generate_with_claude(still_under, candidates_per_clue=15, model="claude-sonnet-4-6")
        for cid, words in extra.items():
            existing = set(candidates.get(cid, []))
            for w in words:
                if w not in existing:
                    existing.add(w)
            candidates[cid] = list(existing)
            candidate_sources.setdefault(cid, {})
            for w in words:
                candidate_sources[cid].setdefault(w.upper(), "sonnet")

    logger.info(f"{_elapsed()} Padding complete")

    # Bouncer filter: score and sort candidates
    put_progress(SolveProgress(stage="candidates", message="Scoring candidates...", progress=0.25))
    scored = bouncer_filter(candidates, db=db, clue_text_lookup=clue_text_lookup, candidate_sources=candidate_sources, web_candidates=web_candidates)
    score_map = to_score_map(scored)
    candidates = to_plain_candidates(scored)

    candidate_scores: Dict[str, Dict[str, float]] = {}
    for cid, sc_list in scored.items():
        candidate_scores[cid] = {sc.word: sc.confidence for sc in sc_list}

    # Trace: bouncer scoring summary
    if solve_trace and trace_global:
        total_cands = sum(len(v) for v in candidates.values())
        trace_global("bouncer", f"Scored {total_cands} candidates across {len(candidates)} clues")

    return candidates, scored, score_map, candidate_scores, candidate_sources, web_candidates, db, clue_text_lookup


def _run_solver(solver_input, clue_text_lookup, candidates, candidate_scores, score_map, total, put_progress, _elapsed, cancel_event=None, solve_trace=None, trace_global=None):
    """Run LLM iterative solver with CSP cleanup fallback.

    Returns (assignment, solved_count).
    """
    from crosswise.solver.llm_solver import solve_with_llm
    from crosswise.solver.csp import solve_csp

    _check_cancel(cancel_event)
    logger.info(f"{_elapsed()} Starting LLM iterative solver")
    if trace_global:
        trace_global("solver_start", f"Starting LLM solver with {total} clues")

    def _llm_progress(pass_num, solved_count, total_count, warning=None):
        _check_cancel(cancel_event)
        put_progress(SolveProgress(
            stage="solving",
            message=f"LLM pass {pass_num}: {solved_count}/{total_count} solved...",
            progress=0.3 + 0.05 * pass_num,
            warning=warning,
        ))

    assignment = solve_with_llm(
        solver_input, clue_text_lookup, candidates,
        max_passes=6,
        progress_callback=_llm_progress,
        candidate_scores=candidate_scores,
        solve_trace=solve_trace,
        trace_global=trace_global,
    )
    logger.info(f"{_elapsed()} LLM solver: {len(assignment)}/{total}")

    # CSP cleanup for any remaining unsolved clues
    _check_cancel(cancel_event)
    if len(assignment) < total and len(assignment) > 0:
        csp_remaining = total - len(assignment)
        logger.info(f"{_elapsed()} Running CSP cleanup on {csp_remaining} remaining clues")
        if trace_global:
            trace_global("csp_cleanup", f"CSP cleanup for {csp_remaining} remaining clues")
        put_progress(SolveProgress(
            stage="solving",
            message=f"CSP cleanup: {csp_remaining} clues remaining...",
            progress=0.55,
        ))
        r = solve_csp(
            solver_input, candidates, return_partial=True,
            candidate_scores=score_map, mac_mode="search-only",
            seed_assignment=assignment,
        )
        if len(r.assignment) > len(assignment):
            csp_new = set(r.assignment.keys()) - set(assignment.keys())
            assignment = r.assignment
            logger.info(f"{_elapsed()} CSP cleanup: {len(assignment)}/{total}")
            if solve_trace:
                for cid in csp_new:
                    solve_trace[cid]["solved_by"] = "csp_cleanup"
            if trace_global:
                trace_global("csp_done", f"+{len(csp_new)} from CSP, total {len(assignment)}/{total}")

    return assignment, len(assignment)


def _save_diagnostics(clue_inputs, clue_text_lookup, candidates, candidate_sources, web_candidates, assignment, db, session_dir, solve_trace=None, global_trace=None):
    """Build and save per-clue solve diagnostics JSON."""
    from crosswise.solver.candidates import bouncer_filter

    final_scored = bouncer_filter(candidates, db=db, clue_text_lookup=clue_text_lookup, candidate_sources=candidate_sources, web_candidates=web_candidates)

    diagnostics = []
    for ci in clue_inputs:
        cid = ci.clue_id
        cands = final_scored.get(cid, [])
        answer = assignment.get(cid)
        if answer:
            status = "solved"
        elif not cands:
            status = "no_candidates"
        else:
            status = "unsolved"
        entry = {
            "clue_id": cid,
            "text": clue_text_lookup.get(cid, ""),
            "length": ci.length,
            "category": ci.category,
            "candidates": [
                {
                    "word": sc.word,
                    "source": sc.source,
                    "confidence": round(sc.confidence, 3),
                    "verified": sc.verified,
                }
                for sc in cands
            ],
            "candidate_count": len(cands),
            "assigned_answer": answer,
            "status": status,
        }
        if solve_trace and cid in solve_trace:
            entry["trace"] = solve_trace[cid]["events"]
            entry["solved_by"] = solve_trace[cid]["solved_by"]
        diagnostics.append(entry)

    output = {"clues": diagnostics}
    if global_trace:
        output["timeline"] = global_trace

    with open(session_dir / "solve_diagnostics.json", "w") as f:
        json.dump(output, f, indent=2)


def _generate_and_apply_hints(puzzle_data, assignment, put_progress, _elapsed):
    """Generate hints in parallel batches, apply to puzzle_data in-place.

    Returns list of solved clue dicts.
    """
    from crosswise.solver.generate_hints import generate_hints_batch

    solved_clues = []
    for direction in ("across", "down"):
        for clue in puzzle_data["clues"][direction]:
            key = f"{clue['number']}-{direction}"
            answer = assignment.get(key)
            if answer:
                clue["answer"] = answer
                solved_clues.append({
                    "number": clue["number"],
                    "direction": direction,
                    "text": clue["text"],
                    "answer": answer,
                })

    if not solved_clues:
        return solved_clues

    batch_size = 20
    batches = [solved_clues[i:i + batch_size] for i in range(0, len(solved_clues), batch_size)]
    total_batches = len(batches)
    put_progress(SolveProgress(
        stage="hints",
        message=f"Generating hints ({total_batches} batches in parallel)...",
        progress=0.6,
    ))

    all_hints: List[Dict[str, str]] = []
    failed_batches = 0
    with ThreadPoolExecutor(max_workers=total_batches) as executor:
        futures = {executor.submit(generate_hints_batch, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            batch_num = futures[future] + 1
            try:
                hints = future.result()
                all_hints.extend(hints)
            except Exception as e:
                failed_batches += 1
                logger.error(f"Hint batch {batch_num} failed: {e}")
            warning = f"{failed_batches} hint batch(es) failed due to API errors" if failed_batches else None
            put_progress(SolveProgress(
                stage="hints",
                message=f"Hints batch {batch_num}/{total_batches} done",
                progress=0.6 + 0.35 * (max(len(all_hints), 1) / len(solved_clues)),
                warning=warning,
            ))

    # Map hints back to puzzle_data
    hint_map = {h["id"]: h for h in all_hints}
    for direction in ("across", "down"):
        for clue in puzzle_data["clues"][direction]:
            key = f"{clue['number']}-{direction}"
            if key in hint_map:
                clue["hint"] = hint_map[key]["hint"]
                clue["explanation"] = hint_map[key]["explanation"]

    return solved_clues


def _run_solve(
    session_dir: Path,
    puzzles_dir: Path,
    puzzle_id: str,
    put_progress,
    session_mgr,
    session_id: str,
    cancel_event: Optional[threading.Event] = None,
):
    """Orchestrate the full solve pipeline: candidates -> solve -> diagnostics -> hints."""
    from crosswise.solver.models import build_solver_input_from_json, build_clue_inputs_from_json
    from crosswise.solver.cost_tracker import reset_tracker

    _t0 = time.time()
    def _elapsed():
        return f"[{time.time() - _t0:.1f}s]"

    tracker = reset_tracker()
    session_mgr.update_status(session_id, SessionStatus.SOLVING)

    with open(puzzles_dir / f"{puzzle_id}.json") as f:
        puzzle_data = json.load(f)

    put_progress(SolveProgress(stage="setup", message="Building solver input...", progress=0.05))
    solver_input = build_solver_input_from_json(puzzle_data)
    clue_inputs = build_clue_inputs_from_json(puzzle_data, solver_input)
    total = len(clue_inputs)

    # Solve trace: per-clue journey + global timeline
    solve_trace: Dict[str, dict] = {
        ci.clue_id: {"events": [], "solved_by": None}
        for ci in clue_inputs
    }
    global_trace: List[dict] = []

    def _trace_global(phase: str, summary: str):
        global_trace.append({"t": round(time.time() - _t0, 1), "phase": phase, "summary": summary})

    # 1. Generate and score candidates
    _check_cancel(cancel_event)
    candidates, scored, score_map, candidate_scores, candidate_sources, web_candidates, db, clue_text_lookup = \
        _generate_candidates(clue_inputs, put_progress, _elapsed, cancel_event, solve_trace, _trace_global)

    # 2. Solve
    assignment, solved = _run_solver(
        solver_input, clue_text_lookup, candidates,
        candidate_scores, score_map, total, put_progress, _elapsed, cancel_event,
        solve_trace, _trace_global,
    )

    # 3. Save diagnostics
    _trace_global("diagnostics", f"Saving diagnostics for {total} clues")
    _save_diagnostics(clue_inputs, clue_text_lookup, candidates, candidate_sources, web_candidates, assignment, db, session_dir, solve_trace, global_trace)

    put_progress(SolveProgress(
        stage="solved",
        message=f"Solved {solved}/{total} clues",
        progress=0.6,
    ))

    # 4. Generate hints
    _check_cancel(cancel_event)
    _trace_global("hints", f"Generating hints for {solved} solved clues")
    logger.info(f"{_elapsed()} Solve complete, starting hints")
    session_mgr.update_status(session_id, SessionStatus.GENERATING_HINTS)
    _generate_and_apply_hints(puzzle_data, assignment, put_progress, _elapsed)

    # Save enriched puzzle
    puzzle_path = puzzles_dir / f"{puzzle_id}.json"
    with open(puzzle_path, "w") as f:
        json.dump(puzzle_data, f, indent=2)

    cost_summary = tracker.summary()
    _trace_global("complete", f"Done: {solved}/{total} solved, {cost_summary.splitlines()[0] if cost_summary else ''}")

    session_mgr.update_status(
        session_id, SessionStatus.COMPLETE,
        solved_count=solved, total_clues=total,
    )

    # Update diagnostics with final global trace (now includes hints + cost)
    _save_diagnostics(clue_inputs, clue_text_lookup, candidates, candidate_sources, web_candidates, assignment, db, session_dir, solve_trace, global_trace)

    logger.info(f"\n{cost_summary}\n")
    logger.info(f"{_elapsed()} Done")
    put_progress(SolveProgress(stage="complete", message="Puzzle ready!", progress=1.0))

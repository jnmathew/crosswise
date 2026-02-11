"""
Pipeline orchestration for the Crosswise API.
Wraps existing core functions for use in FastAPI endpoints.
"""

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.core.config import Settings
from src.api.models import MaskRequest, SolveProgress, SessionStatus


def run_grid_detection(session_dir: Path, config: Settings) -> Dict[str, Any]:
    """Phase 1: Preprocess image and detect grid."""
    from src.core.image_preprocessing import preprocess
    from src.core.grid_detection import detect_grid, assign_clue_numbers, compute_clue_slots

    original_path = session_dir / "original.jpg"
    result = preprocess(original_path, config)

    # Save warped images
    cv2.imwrite(str(session_dir / "warped.jpg"), result["warped"])
    cv2.imwrite(str(session_dir / "warped_gray.jpg"), result["warped_gray"])

    # Detect grid
    grid_result = detect_grid(result["warped_gray"], config)
    cells = grid_result["cells"]
    assign_clue_numbers(cells)
    clue_slots = compute_clue_slots(cells)

    rows = len(cells)
    cols = len(cells[0]) if rows > 0 else 0

    # Serialize cells for JSON storage
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

    return {
        "grid_size": (rows, cols),
        "clue_slot_count": len(clue_slots),
    }


def apply_grid_edit(session_dir: Path, black_cells: list[list[bool]]) -> Dict[str, Any]:
    """Recompute grid clue numbers and slots from user-edited black cell map."""
    from src.core.grid_detection import assign_clue_numbers, compute_clue_slots
    from src.core.models import Cell

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

    # Serialize and save
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

    clue_number_count = max(
        (cell.clue_number for row in cells for cell in row if cell.clue_number is not None),
        default=0,
    )

    return {
        "grid_size": (rows, cols),
        "clue_slot_count": len(clue_slots),
        "clue_number_count": clue_number_count,
    }


def apply_masks(image: np.ndarray, mask: MaskRequest) -> np.ndarray:
    """Apply white rectangles and aqua separator lines to image."""
    img = image.copy()
    for rect in mask.rectangles:
        cv2.rectangle(img, (rect.x1, rect.y1), (rect.x2, rect.y2), (255, 255, 255), -1)
    for sep in mask.separators:
        cv2.line(img, (sep.x1, sep.y1), (sep.x2, sep.y2), (255, 255, 0), 8)
    return img


def run_mistral_ocr(image_path: Path) -> str:
    """Run Mistral OCR on an image, return markdown-formatted clue text."""
    from pydantic import BaseModel as PydanticBaseModel
    from typing import List as TypingList
    from mistralai import Mistral, ImageURLChunk
    from mistralai.extra import response_format_from_pydantic_model

    class Clue(PydanticBaseModel):
        num: int
        clue: str

    class CrosswordClues(PydanticBaseModel):
        ACROSS: TypingList[Clue]
        DOWN: TypingList[Clue]

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as mistral:
        res = mistral.ocr.process(
            model="mistral-ocr-latest",
            document=ImageURLChunk(image_url=f"data:image/jpeg;base64,{img_b64}"),
            document_annotation_format=response_format_from_pydantic_model(CrosswordClues),
            include_image_base64=False,
        )

    # Convert to markdown format expected by parse_ocr_markdown
    if not hasattr(res, "document_annotation") or not res.document_annotation:
        raise ValueError("Mistral OCR returned no structured data")

    clues = json.loads(res.document_annotation) if isinstance(res.document_annotation, str) else res.document_annotation

    lines = ["## ACROSS\n"]
    for clue in clues.get("ACROSS", []) if isinstance(clues, dict) else clues.ACROSS:
        c = clue if isinstance(clue, dict) else {"num": clue.num, "clue": clue.clue}
        lines.append(f"{c['num']}. {c['clue']}")
    lines.append("\n## DOWN\n")
    for clue in clues.get("DOWN", []) if isinstance(clues, dict) else clues.DOWN:
        c = clue if isinstance(clue, dict) else {"num": clue.num, "clue": clue.clue}
        lines.append(f"{c['num']}. {c['clue']}")

    return "\n".join(lines)


def run_ocr_and_verify(session_dir: Path, mask: MaskRequest, config: Settings) -> Dict[str, Any]:
    """Phase 2: Apply masks, run OCR, verify against grid."""
    from src.core.clue_extraction import parse_ocr_markdown, verify_puzzle

    # Load original image and apply masks
    original = cv2.imread(str(session_dir / "original.jpg"))
    masked = apply_masks(original, mask)
    cv2.imwrite(str(session_dir / "masked.jpg"), masked)

    # Run Mistral OCR
    ocr_markdown = run_mistral_ocr(session_dir / "masked.jpg")
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


def run_solve_background(
    session_dir: Path,
    puzzles_dir: Path,
    puzzle_id: str,
    queue: asyncio.Queue,
    session_mgr: Any,
    session_id: str,
):
    """Background task: solve puzzle + generate hints, updating puzzle JSON in-place."""
    import asyncio as _asyncio

    loop = _asyncio.new_event_loop()

    def put(progress: SolveProgress):
        loop.run_until_complete(queue.put(progress))

    try:
        _run_solve(session_dir, puzzles_dir, puzzle_id, put, session_mgr, session_id)
    except Exception as e:
        put(SolveProgress(stage="failed", message=str(e), progress=0))
        session_mgr.update_status(session_id, SessionStatus.FAILED, error=str(e))
    finally:
        loop.close()


def _run_solve(
    session_dir: Path,
    puzzles_dir: Path,
    puzzle_id: str,
    put_progress,
    session_mgr,
    session_id: str,
):
    from src.solver.solve_puzzle import build_solver_input_from_json, build_clue_inputs_from_json
    from src.solver.candidate_generator import (
        generate_candidates_with_database,
        generate_with_claude,
        regenerate_with_patterns,
        bouncer_filter,
        to_plain_candidates,
        to_score_map,
        ensure_minimum_candidates,
        categorize_clue,
        ClueInput,
    )
    from src.solver.csp import solve_csp, extract_letter_patterns
    from src.solver.clue_database import ClueDatabase
    from src.solver.generate_hints import generate_hints_batch

    session_mgr.update_status(session_id, SessionStatus.SOLVING)

    with open(session_dir / "puzzle.json") as f:
        puzzle_data = json.load(f)

    put_progress(SolveProgress(stage="setup", message="Building solver input...", progress=0.05))

    solver_input = build_solver_input_from_json(puzzle_data)
    clue_inputs = build_clue_inputs_from_json(puzzle_data, solver_input)

    # Build clue text lookup for bouncer filter
    clue_text_lookup = {}
    for direction in ("across", "down"):
        for clue in puzzle_data["clues"].get(direction, []):
            clue_text_lookup[f"{clue['number']}-{direction}"] = clue["text"]

    # Generate candidates (DB lookup + Claude fallback instead of OpenAI)
    put_progress(SolveProgress(stage="candidates", message="Database lookup...", progress=0.1))
    db = ClueDatabase()
    candidates = generate_candidates_with_database(
        clue_inputs, db=db, candidates_per_clue=12, use_llm_fallback=False
    )

    # Claude fallback for clues not found in database
    clues_needing_llm = [c for c in clue_inputs if not candidates.get(c.clue_id)]
    if clues_needing_llm:
        put_progress(SolveProgress(
            stage="candidates",
            message=f"Claude generating for {len(clues_needing_llm)} clues...",
            progress=0.15,
        ))
        claude_cands = generate_with_claude(clues_needing_llm, candidates_per_clue=12)
        candidates.update(claude_cands)
        print(f"  Claude generated: {sum(1 for v in claude_cands.values() if v)}/{len(clues_needing_llm)} clues")

    # Ensure minimum candidates per clue
    put_progress(SolveProgress(stage="candidates", message="Ensuring minimum candidates...", progress=0.2))
    candidates = ensure_minimum_candidates(
        clue_inputs, candidates, db=db, min_candidates=5,
    )

    # Bouncer filter: score and sort candidates by DB/word-index verification
    put_progress(SolveProgress(stage="candidates", message="Scoring candidates...", progress=0.25))
    scored = bouncer_filter(candidates, db=db, clue_text_lookup=clue_text_lookup)
    score_map = to_score_map(scored)
    candidates = to_plain_candidates(scored)

    total = len(clue_inputs)

    def _run_csp_best_of_n(candidates, score_map, n=3, label=""):
        """Run CSP solver n times and return the best result."""
        best = {}
        for attempt in range(n):
            r = solve_csp(
                solver_input, candidates, return_partial=True,
                candidate_scores=score_map, mac_mode="search-only",
            )
            if len(r.assignment) > len(best):
                best = r.assignment.copy()
            if r.solved:
                break
        return best

    # Pass 1: Solve with search-only MAC (skip AC-3 preprocessing that can wipe out domains)
    put_progress(SolveProgress(stage="solving", message="Running CSP solver (pass 1, best of 3)...", progress=0.3))
    best_assignment = _run_csp_best_of_n(candidates, score_map, n=3, label="pass 1")

    # Multi-pass pattern refinement (up to 4 additional passes)
    # Each pass: extract crossing-letter patterns → regenerate candidates → re-solve
    max_passes = 4
    for pass_num in range(2, 2 + max_passes):
        if len(best_assignment) >= total or len(best_assignment) == 0:
            break

        put_progress(SolveProgress(
            stage="solving",
            message=f"Pass {pass_num}: {len(best_assignment)}/{total} solved. Refining with patterns...",
            progress=0.3 + 0.08 * (pass_num - 1),
        ))

        # Use best assignment for patterns (more known letters = better patterns)
        patterns = extract_letter_patterns(solver_input, best_assignment)
        unsolved_with_patterns = {
            cid: pat for cid, pat in patterns.items()
            if "_" in pat and pat.count("_") < len(pat)
        }

        if not unsolved_with_patterns:
            break

        constrained_clues = []
        for clue_id, pattern in unsolved_with_patterns.items():
            text = clue_text_lookup.get(clue_id, "")
            constrained_clues.append(ClueInput(
                clue_id=clue_id,
                text=text,
                length=len(pattern),
                pattern=pattern,
                category=categorize_clue(text),
                num_crossings=solver_input.crossing_count(clue_id),
            ))

        # DB pattern matching first
        new_candidates = regenerate_with_patterns(
            constrained_clues, db=db, candidates_per_clue=16,
        )

        # Claude fallback for clues where DB pattern matching found nothing or few results
        clues_for_llm = [
            c for c in constrained_clues
            if len(new_candidates.get(c.clue_id, [])) < 3
        ]
        if clues_for_llm:
            put_progress(SolveProgress(
                stage="solving",
                message=f"Claude generating for {len(clues_for_llm)} clues with patterns...",
                progress=0.32 + 0.08 * (pass_num - 1),
            ))
            claude_candidates = generate_with_claude(
                clues_for_llm, candidates_per_clue=12,
            )
            for cid, cands in claude_candidates.items():
                if cands:
                    existing = new_candidates.get(cid, [])
                    new_candidates[cid] = existing + cands

        # Merge new candidates into main pool
        added = 0
        for clue_id, cands in new_candidates.items():
            if cands:
                existing = set(candidates.get(clue_id, []))
                new_unique = [c.upper() for c in cands if c.upper() not in existing]
                if new_unique:
                    candidates[clue_id] = list(existing) + new_unique
                    added += 1

        if added == 0:
            break

        # Re-score after adding new candidates
        scored = bouncer_filter(candidates, db=db, clue_text_lookup=clue_text_lookup)
        score_map = to_score_map(scored)
        candidates = to_plain_candidates(scored)

        put_progress(SolveProgress(
            stage="solving",
            message=f"Running CSP solver (pass {pass_num}, best of 3)...",
            progress=0.35 + 0.08 * (pass_num - 1),
        ))
        # Fresh solve with enriched candidate pool (no seeding — locks in wrong answers)
        assignment = _run_csp_best_of_n(candidates, score_map, n=3, label=f"pass {pass_num}")

        # Keep the best assignment across all passes
        if len(assignment) > len(best_assignment):
            best_assignment = assignment.copy()

    # Use the best assignment found across all passes
    assignment = best_assignment
    solved = len(assignment)

    put_progress(SolveProgress(
        stage="solved",
        message=f"Solved {solved}/{total} clues",
        progress=0.6,
    ))

    # Generate hints
    session_mgr.update_status(session_id, SessionStatus.GENERATING_HINTS)
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

    if solved_clues:
        batch_size = 20
        all_hints: List[Dict[str, str]] = []
        for i in range(0, len(solved_clues), batch_size):
            batch = solved_clues[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(solved_clues) + batch_size - 1) // batch_size
            put_progress(SolveProgress(
                stage="hints",
                message=f"Generating hints (batch {batch_num}/{total_batches})...",
                progress=0.6 + 0.35 * (i / len(solved_clues)),
            ))
            hints = generate_hints_batch(batch)
            all_hints.extend(hints)

        # Map hints back
        hint_map = {h["id"]: h for h in all_hints}
        for direction in ("across", "down"):
            for clue in puzzle_data["clues"][direction]:
                key = f"{clue['number']}-{direction}"
                if key in hint_map:
                    clue["hint"] = hint_map[key]["hint"]
                    clue["explanation"] = hint_map[key]["explanation"]

    # Save enriched puzzle
    puzzle_path = puzzles_dir / f"{puzzle_id}.json"
    with open(puzzle_path, "w") as f:
        json.dump(puzzle_data, f, indent=2)

    session_mgr.update_status(
        session_id, SessionStatus.COMPLETE,
        solved_count=solved, total_clues=total,
    )

    put_progress(SolveProgress(stage="complete", message="Puzzle ready!", progress=1.0))

"""
Grid detection utilities for crossword puzzles.

Pipeline:
1. Detect grid lines using projection method (primary)
2. Fallback to intersection method if insufficient lines
3. Build Cell grid from line positions
4. Classify black cells by mean intensity
"""
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from loguru import logger

from .config import Settings
from .models import Cell


def _smooth1d(arr: np.ndarray, k: int) -> np.ndarray:
    """Smooth 1D array with moving average filter."""
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(arr.astype(np.float32), kernel, mode="same")


def _find_peaks(signal: np.ndarray, min_distance: int, threshold_rel: float) -> List[int]:
    """
    Find peaks in 1D signal with non-maximum suppression.

    Args:
        signal: 1D array
        min_distance: Minimum distance between peaks
        threshold_rel: Threshold as fraction of max value (0.0-1.0)

    Returns:
        List of peak indices
    """
    s = signal.astype(np.float32)
    if s.size == 0:
        return []
    thr = float(s.max()) * float(threshold_rel)
    peaks = []
    last_keep = -10_000
    for i in range(1, len(s) - 1):
        if s[i] >= thr and s[i] >= s[i - 1] and s[i] >= s[i + 1]:
            if i - last_keep >= min_distance:
                peaks.append(i)
                last_keep = i
            else:
                # non-max suppression within window: keep the higher one
                if peaks and s[i] > s[peaks[-1]]:
                    peaks[-1] = i
                    last_keep = i
    return peaks


def _validate_grid_lines(lines: List[int], image_size: int) -> List[int]:
    """
    Filter grid lines to remove duplicates and extreme outliers.

    Uses simple ratio-based filtering to remove only egregious spacing outliers
    while preserving grids with slightly irregular spacing.

    Args:
        lines: List of grid line positions
        image_size: Total image dimension (for bounds checking)

    Returns:
        Filtered list of grid lines
    """
    if len(lines) < 4:
        return lines

    # Step 1: Remove obvious duplicates (lines closer than min spacing)
    min_spacing = max(10, image_size // 150)
    deduped = [lines[0]]
    for i in range(1, len(lines)):
        if lines[i] - deduped[-1] >= min_spacing:
            deduped.append(lines[i])
        else:
            logger.debug(f"Removing duplicate at {lines[i]} (spacing={lines[i] - deduped[-1]}px)")

    if len(deduped) < 4:
        return deduped

    # Step 2: Remove extreme spacing outliers (simple ratio-based approach)
    spacings = np.array([deduped[i+1] - deduped[i] for i in range(len(deduped) - 1)])
    median_spacing = np.median(spacings)

    # Remove lines with spacing < 40% or > 250% of median (very extreme)
    filtered = [deduped[0]]
    for i in range(len(deduped) - 1):
        spacing = deduped[i+1] - deduped[i]
        ratio = spacing / median_spacing

        if 0.4 <= ratio <= 2.5:
            filtered.append(deduped[i+1])
        else:
            logger.debug(f"Removing extreme outlier at {deduped[i+1]} (spacing={spacing:.1f}, median={median_spacing:.1f}, ratio={ratio:.2f})")

    return filtered


def _merge_close(indices: List[int], median_spacing: float) -> List[int]:
    """
    Merge peaks aggressively based on expected spacing.

    Args:
        indices: List of peak indices
        median_spacing: Median spacing between peaks (for adaptive merging)

    Returns:
        Merged list of peaks
    """
    if not indices:
        return []

    # Use median spacing to determine merge threshold
    # Merge if peaks are closer than 40% of expected spacing
    merge_threshold = max(12, int(median_spacing * 0.4)) if median_spacing > 0 else 12

    indices = sorted(indices)
    merged = [indices[0]]
    for idx in indices[1:]:
        if idx - merged[-1] < merge_threshold:
            merged[-1] = int((merged[-1] + idx) / 2)
        else:
            merged.append(idx)
    return merged


def _auto_peaks(
    signal_sm: np.ndarray,
    image_size: int,
    axis: str,
    min_distance: int,
    *,
    rel_thresholds=(0.55, 0.45, 0.35, 0.25),
    min_lines=6,
    max_lines=40,
    max_spacing_jitter=0.35,
) -> List[int]:
    """
    Automatically choose a good peak threshold for one axis (x or y).

    Tries multiple thresholds and picks the candidate with:
    - count in a reasonable range
    - relatively regular spacing (low jitter)

    Args:
        signal_sm: Smoothed 1D projection signal
        image_size: Total image dimension (for validation)
        axis: "x" or "y" (for debug logging)
        min_distance: Minimum distance between peaks
        rel_thresholds: Thresholds to try in descending order
        min_lines: Minimum acceptable number of grid lines
        max_lines: Maximum acceptable number of grid lines
        max_spacing_jitter: Maximum acceptable spacing irregularity (std/median)

    Returns:
        Best candidate grid line positions
    """
    best_score = -1.0
    best_lines: List[int] = []

    for thr in rel_thresholds:
        cand = _find_peaks(signal_sm, min_distance, threshold_rel=thr)
        if not cand:
            continue

        cand = _validate_grid_lines(sorted(cand), image_size)
        if len(cand) < 2:
            continue

        spacings = np.diff(cand)
        if spacings.size == 0:
            continue
        median = float(np.median(spacings))
        if median <= 0:
            continue
        jitter = float(np.std(spacings) / median)
        n = len(cand)

        # Scoring: prefer counts in [min_lines, max_lines] and low jitter
        count_score = 1.0 - min(
            1.0,
            abs(n - (min_lines + max_lines) / 2)
            / ((max_lines - min_lines) / 2 + 1e-6),
        )
        jitter_penalty = min(1.0, jitter / max_spacing_jitter)
        score = max(0.0, count_score) * (1.0 - jitter_penalty)

        # Small bonus for higher threshold (cleaner peaks)
        score += 0.05 * thr

        logger.debug(
            f"{axis}-axis thr={thr:.2f}: n={n}, jitter={jitter:.3f}, score={score:.3f}"
        )

        if score > best_score:
            best_score = score
            best_lines = cand

        # Early exit if this candidate is already quite good
        if (
            min_lines <= n <= max_lines
            and jitter < max_spacing_jitter * 0.7
        ):
            logger.debug(f"{axis}-axis: early exit with excellent candidate (n={n}, jitter={jitter:.3f})")
            break

    return best_lines


def _detect_grid_lines_by_projection(warped_gray: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Detect grid line positions using row/column projections.

    This is the primary/robust method for grid detection.
    Uses adaptive threshold selection to work across different images.

    Args:
        warped_gray: Grayscale warped grid image

    Returns:
        (xs, ys) where:
            - xs: x-coordinates of vertical grid lines
            - ys: y-coordinates of horizontal grid lines
    """
    # Adaptive threshold to emphasize lines
    th = cv2.adaptiveThreshold(
        warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    h, w = th.shape[:2]

    # Mild closing to fill broken line gaps predominantly in each direction
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, h // 120)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 120), 1))
    th_v = cv2.morphologyEx(th, cv2.MORPH_CLOSE, v_kernel, iterations=1)
    th_h = cv2.morphologyEx(th, cv2.MORPH_CLOSE, h_kernel, iterations=1)

    # Project to find grid lines
    col_proj = th_v.sum(axis=0)
    row_proj = th_h.sum(axis=1)
    col_sm = _smooth1d(col_proj, max(5, w // 200))
    row_sm = _smooth1d(row_proj, max(5, h // 200))

    # Find peaks using adaptive threshold selection
    col_min_dist = max(8, w // 40)
    row_min_dist = max(8, h // 40)

    xs = _auto_peaks(
        col_sm,
        image_size=w,
        axis="x",
        min_distance=col_min_dist,
    )

    ys = _auto_peaks(
        row_sm,
        image_size=h,
        axis="y",
        min_distance=row_min_dist,
    )

    logger.info(f"Adaptive grid detection: {len(xs)} vertical, {len(ys)} horizontal lines")

    return xs, ys


def _detect_grid_intersections(warped_gray: np.ndarray) -> List[Tuple[float, float]]:
    """
    Detect grid line intersections (fallback method).

    Detects vertical and horizontal lines separately, then finds their
    intersections and clusters them into a grid.

    Args:
        warped_gray: Grayscale warped grid image

    Returns:
        List of (x, y) intersection points in row-major order
    """
    # Adaptive threshold to emphasise lines
    th = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)

    h, w = warped_gray.shape[:2]

    # Detect vertical lines
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(2, w // 40)))
    vert = cv2.erode(th, vert_kernel, iterations=1)
    vert = cv2.dilate(vert, vert_kernel, iterations=2)

    # Detect horizontal lines
    horz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(2, h // 40), 1))
    horz = cv2.erode(th, horz_kernel, iterations=1)
    horz = cv2.dilate(horz, horz_kernel, iterations=2)

    # Intersections between vertical and horizontal lines
    inter = cv2.bitwise_and(vert, horz)

    # Find connected components / centroids
    cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in cnts:
        M = cv2.moments(c)
        if M.get("m00", 0) == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centers.append((cx, cy))

    if not centers:
        return []

    # Sort centers into a grid by clustering y and x
    pts = np.array(centers)
    ys = np.sort(pts[:, 1])
    xs = np.sort(pts[:, 0])

    # Estimate cell spacing as the median difference between adjacent coords
    def median_spacing(values):
        if len(values) < 2:
            return None
        diffs = np.diff(values)
        return np.median(diffs)

    y_spacing = median_spacing(ys)
    x_spacing = median_spacing(xs)
    if y_spacing is None or x_spacing is None:
        return []

    y_tol = max(4.0, float(y_spacing) * 0.4)

    # Group by rows
    rows = []
    for cx, cy in sorted(centers, key=lambda p: (p[1], p[0])):
        placed = False
        for r in rows:
            if abs(r["y_mean"] - cy) <= y_tol:
                r["pts"].append((cx, cy))
                r["y_mean"] = np.mean([p[1] for p in r["pts"]])
                placed = True
                break
        if not placed:
            rows.append({"y_mean": cy, "pts": [(cx, cy)]})

    # For each row, sort pts by x
    grid = []
    for r in rows:
        row_pts = sorted(r["pts"], key=lambda p: p[0])
        grid.append(row_pts)

    # Verify roughly rectangular grid: consistent number of cols
    counts = [len(r) for r in grid]
    if not counts:
        return []
    col_count = max(set(counts), key=counts.count)

    # Filter rows that don't match majority column count
    grid = [r for r in grid if len(r) >= max(2, col_count - 1)]

    # Normalize rows to the same number of cols by selecting equally-spaced points
    final_grid = []
    for r in grid:
        if len(r) == col_count:
            final_grid.append(r)
        else:
            # Pick approximate evenly spaced
            xs_row = np.array([p[0] for p in r])
            target_xs = np.linspace(xs_row.min(), xs_row.max(), col_count)
            chosen = []
            for tx in target_xs:
                # Pick nearest
                idx = int(np.argmin(np.abs(xs_row - tx)))
                chosen.append(r[idx])
            final_grid.append(chosen)

    # Flatten into intersection points in row-major order
    intersections = []
    for r in final_grid:
        for p in r:
            intersections.append((float(p[0]), float(p[1])))

    return intersections


def detect_grid(warped_gray: np.ndarray, config: Settings) -> Dict[str, Any]:
    """
    Detect grid structure from warped puzzle image.

    Tries projection-based method first (robust), falls back to intersection
    method if insufficient lines detected.

    Args:
        warped_gray: Grayscale warped grid image (typically 1000x1000)
        config: Settings object

    Returns:
        Dictionary with:
            - cells: List[List[Cell]] - 2D grid of Cell objects
            - grid_lines: {'xs': List[int], 'ys': List[int]} - Grid line positions
            - grid_size: (rows, cols) - Grid dimensions
            - meta: dict - Detection metadata

    Raises:
        ValueError: If grid detection fails
    """
    logger.info("Detecting grid structure")

    # Try projection method first (primary)
    xs, ys = _detect_grid_lines_by_projection(warped_gray)

    method = "projection"
    if len(xs) < 2 or len(ys) < 2:
        logger.warning(f"Projection method found insufficient lines ({len(xs)} cols, {len(ys)} rows), trying fallback")
        # Fallback to intersection method
        intersections = _detect_grid_intersections(warped_gray)
        if not intersections:
            raise ValueError("Grid detection failed: no intersections found")

        # Extract xs and ys from intersections
        pts = np.array(intersections)
        xs = sorted(list(set(np.round(pts[:, 0]).astype(int))))
        ys = sorted(list(set(np.round(pts[:, 1]).astype(int))))
        method = "intersection"

    if len(xs) < 2 or len(ys) < 2:
        raise ValueError(f"Grid detection failed: insufficient lines detected ({len(xs)} cols, {len(ys)} rows)")

    num_cols = len(xs) - 1
    num_rows = len(ys) - 1

    logger.info(f"Grid detected: {num_rows} rows × {num_cols} cols (method: {method})")

    # Build Cell grid
    cells: List[List[Cell]] = []
    h, w = warped_gray.shape[:2]

    # First pass: collect all cell intensities for adaptive thresholding
    cell_intensities = []
    for r in range(num_rows):
        y0 = ys[r]
        y1 = ys[r + 1] if r + 1 < len(ys) else h

        for c in range(num_cols):
            x0 = xs[c]
            x1 = xs[c + 1] if c + 1 < len(xs) else w

            cell_crop = warped_gray[y0:y1, x0:x1]
            if cell_crop.size > 0:
                cell_intensities.append(cell_crop.mean())

    # Compute adaptive threshold using multiple methods and select best
    if cell_intensities:
        intensities_array = np.array(cell_intensities)

        # Method 1: Otsu's method
        intensities_uint8 = intensities_array.astype(np.uint8)
        otsu_threshold, _ = cv2.threshold(
            intensities_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Method 2: Percentile-based (assume 15-25% of cells are black)
        # Use 23rd percentile as balanced estimate across different puzzles
        percentile_threshold = np.percentile(intensities_array, 23)

        # Method 3: Look for gap in intensity distribution (dark vs light cells)
        # Sort intensities and find largest gap in lower 40% of cells
        sorted_intensities = np.sort(intensities_array)
        search_range = int(len(sorted_intensities) * 0.40)  # Look in lower 40%
        if search_range > 1:
            diffs = np.diff(sorted_intensities[:search_range])
            if len(diffs) > 0:
                gap_idx = np.argmax(diffs)
                gap_size = diffs[gap_idx]
                gap_threshold = (sorted_intensities[gap_idx] + sorted_intensities[gap_idx + 1]) / 2
            else:
                gap_size = 0
                gap_threshold = percentile_threshold
        else:
            gap_size = 0
            gap_threshold = percentile_threshold

        # Intelligent selection based on gap size:
        # - Small gap (3-15 units): indicates subtle but clear threshold → use gap method
        # - Large gap (>15 units): indicates multiple black cell clusters → use percentile
        # - Very small gap (<3): noise, use percentile
        if 40 <= gap_threshold <= 150 and 3 <= gap_size <= 15:
            threshold = gap_threshold
            method_used = "gap"
            logger.debug(f"Using gap method: clean separation with gap of {gap_size:.1f} units")
        elif 40 <= percentile_threshold <= 150:
            threshold = percentile_threshold
            method_used = "percentile"
            if gap_size > 15:
                logger.debug(f"Using percentile: large gap ({gap_size:.1f}) suggests multiple clusters")
        elif 40 <= gap_threshold <= 150:
            threshold = gap_threshold
            method_used = "gap_fallback"
        else:
            threshold = otsu_threshold
            method_used = "otsu"

        logger.info(
            f"Adaptive threshold: {threshold:.1f} (method={method_used}, "
            f"otsu={otsu_threshold}, percentile={percentile_threshold:.1f}, gap={gap_threshold:.1f}, gap_size={gap_size:.1f})"
        )
    else:
        threshold = 55  # Fallback
        logger.warning(f"No cells found, using fallback threshold: {threshold}")

    # Second pass: classify cells using adaptive threshold
    idx = 0
    for r in range(num_rows):
        row_cells = []
        y0 = ys[r]
        y1 = ys[r + 1] if r + 1 < len(ys) else h

        for c in range(num_cols):
            x0 = xs[c]
            x1 = xs[c + 1] if c + 1 < len(xs) else w

            # Extract cell crop
            cell_crop = warped_gray[y0:y1, x0:x1]

            # Classify using adaptive threshold
            if cell_crop.size > 0 and idx < len(cell_intensities):
                mean_intensity = cell_intensities[idx]
                is_black = bool(mean_intensity < threshold)
                idx += 1
            else:
                is_black = False

            cell = Cell(row=r, col=c, is_block=is_black)
            row_cells.append(cell)

        cells.append(row_cells)

    # Count black cells for metadata
    black_count = sum(1 for row in cells for cell in row if cell.is_block)
    logger.info(f"Classified {black_count} black cells using threshold={threshold}")

    return {
        "cells": cells,
        "grid_lines": {"xs": xs, "ys": ys},
        "grid_size": (num_rows, num_cols),
        "meta": {
            "method": method,
            "confidence": 1.0 if method == "projection" else 0.8,
            "black_cells": black_count
        }
    }

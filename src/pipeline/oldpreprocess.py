"""
Minimal preprocessing pipeline:
- load images from src/examples
- detect largest quadrilateral (assumed puzzle)
- perspective-warp to a square
- produce gray, blurred, edged, thresh, and warped images in data/temp/
- optional quick OCR dump of the warped image (pytesseract --psm 6)

Run: python src/pipeline/preprocess.py
"""
from pathlib import Path
import cv2
import numpy as np
import pytesseract
import os
import argparse
import sys
import csv
import time

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
OUT_DIR = Path.cwd() / "data" / "temp"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image(path):
    return cv2.imread(str(path))


def resize_max(image, max_dim=1200):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def four_point_transform(image, pts, dst_size=1000):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    dst = np.array([
        [0, 0],
        [dst_size - 1, 0],
        [dst_size - 1, dst_size - 1],
        [0, dst_size - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (dst_size, dst_size))
    return warped


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    sums = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _contour_quad_from_mask(mask, approx_eps=0.02):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, approx_eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")
    return None


def _is_valid_quad(pts, image_shape, min_area_ratio=0.05):
    img_area = image_shape[0] * image_shape[1]
    area = abs(cv2.contourArea(pts.reshape(4, 1, 2)))
    if area <= 0:
        return False
    ratio = area / float(img_area)
    return ratio >= min_area_ratio

def manual_select_quad(image):
    # interactive fallback for difficult images (requires GUI)
    # returns 4-point float32 rect or None
    # We implement a small mouse-based ROI selector because cv2.selectROI can
    # behave inconsistently across platforms. Controls:
    #  - click+drag to draw rectangle
    #  - ENTER / RETURN: accept selection
    #  - ESC or 'c': cancel
    win = "Select ROI - drag, SPACE/'a' accept, c/ESC cancel, double-click accept"

    # Work on a scaled copy for very large images to keep the window manageable
    max_win = 1200
    ih, iw = image.shape[:2]
    scale = min(1.0, max_win / float(max(ih, iw)))
    disp = image if scale == 1.0 else cv2.resize(image, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA)

    selection = {"start": None, "end": None, "dragging": False, "finalized": False}

    def _on_mouse(evt, x, y, flags, param):
        if evt == cv2.EVENT_LBUTTONDOWN:
            selection["start"] = (x, y)
            selection["end"] = (x, y)
            selection["dragging"] = True
        elif evt == cv2.EVENT_MOUSEMOVE and selection["dragging"]:
            selection["end"] = (x, y)
        elif evt == cv2.EVENT_LBUTTONUP and selection["dragging"]:
            selection["end"] = (x, y)
            selection["dragging"] = False
            selection["finalized"] = True
        elif evt == cv2.EVENT_LBUTTONDBLCLK:
            # accept current rect on double-click if any
            selection["end"] = (x, y)
            selection["dragging"] = False
            selection["finalized"] = True

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _on_mouse)

    result_rect = None
    try:
        while True:
            frame = disp.copy()
            if selection["start"] and selection["end"]:
                x0, y0 = selection["start"]
                x1, y1 = selection["end"]
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            info = "Drag. SPACE/'a'=accept, c/ESC=cancel, double-click=accept"
            cv2.putText(frame, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, frame)
            key = cv2.waitKey(20) & 0xFF
            # accept on SPACE or 'a' to avoid ENTER-return glitches on some macOS builds
            if key in (32, ord('a')) or selection.get("finalized"):
                if selection["start"] and selection["end"]:
                    x0, y0 = selection["start"]
                    x1, y1 = selection["end"]
                    x_min, x_max = sorted((x0, x1))
                    y_min, y_max = sorted((y0, y1))
                    # map back to original image scale
                    if scale != 1.0:
                        x_min = int(x_min / scale)
                        x_max = int(x_max / scale)
                        y_min = int(y_min / scale)
                        y_max = int(y_max / scale)
                    w = x_max - x_min
                    h = y_max - y_min
                    if w > 5 and h > 5:
                        result_rect = (x_min, y_min, w, h)
                break
            if key == 27 or key == ord("c"):  # ESC or c
                result_rect = None
                break
    finally:
        try:
            cv2.destroyWindow(win)
            # On some platforms a small delay + extra waitKey helps flush window events
            cv2.waitKey(1)
            time.sleep(0.02)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass

    if not result_rect:
        return None
    x, y, w, h = result_rect
    pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32")
    return pts


def manual_select_corners(image):
    """Manual four-corner selector.
    Click 4 corners of the puzzle in any order (TL, TR, BR, BL recommended).
    Controls: left-click to add point, 'u' to undo, 'r' to reset, ESC/'c' to cancel.
    When 4 points are placed, the selection auto-accepts.
    Returns 4x2 float32 points ordered using order_points, or None if canceled.
    """
    win = "Select 4 corners: click 4 points, u=undo, r=reset, ESC/c=cancel"

    # scale display if image is large
    max_win = 1200
    ih, iw = image.shape[:2]
    scale = min(1.0, max_win / float(max(ih, iw)))
    disp = image if scale == 1.0 else cv2.resize(image, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA)

    points = []  # list of (x, y) in display coords

    def _on_mouse(evt, x, y, flags, param):
        if evt == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _on_mouse)

    result_pts = None
    try:
        while True:
            frame = disp.copy()
            # draw existing points and lines
            for i, (px, py) in enumerate(points):
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            if len(points) >= 2:
                for i in range(1, len(points)):
                    cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
            info = "Click 4 corners. u=undo, r=reset, ESC/c=cancel"
            cv2.putText(frame, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('u') and points:
                points.pop()
            elif key == ord('r'):
                points.clear()
            elif key in (27, ord('c')):
                result_pts = None
                break
            # auto-accept when 4 points set
            if len(points) == 4:
                # map back to original image coordinates
                pts = []
                for (px, py) in points:
                    if scale != 1.0:
                        ox = int(px / scale)
                        oy = int(py / scale)
                    else:
                        ox, oy = px, py
                    pts.append([ox, oy])
                pts = np.array(pts, dtype="float32")
                # order corners for perspective transform
                result_pts = order_points(pts)
                break
    finally:
        try:
            cv2.destroyWindow(win)
            cv2.waitKey(1)
        except Exception:
            pass

    return result_pts


def find_largest_quad(image, try_manual=False):
    """
    Try multiple strategies to find the puzzle quad:
      1) Canny edges + contour approx (fast)
      2) Adaptive threshold + morphological close + contour approx
      3) minAreaRect on largest contour (fallback)
      4) manual ROI (interactive) if try_manual=True
    Returns (pts, gray, edged) where pts is 4x2 float32 quad or None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Strategy A: blurred Canny + contour approx
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    quad = _contour_quad_from_mask(edged, approx_eps=0.02)
    if quad is not None and _is_valid_quad(quad, image.shape):
        return quad, gray, edged

    # Strategy B: adaptive threshold + morphological close
    thresh = adaptive_thresh(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    quad = _contour_quad_from_mask(closed, approx_eps=0.03)
    if quad is not None and _is_valid_quad(quad, image.shape):
        # produce a better edged image for debugging
        edged_from_thresh = cv2.Canny(cv2.GaussianBlur(closed, (3, 3), 0), 30, 100)
        return quad, gray, edged_from_thresh

    # Strategy C: largest contour -> minAreaRect fallback
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect).astype("float32")
        if _is_valid_quad(box, image.shape):
            return box, gray, edged

    # Strategy D: interactive manual selection (only if requested)
    if try_manual:
        pts = manual_select_quad(image)
        if pts is not None and _is_valid_quad(pts, image.shape, min_area_ratio=0.01):
            return pts, gray, edged

    return None, gray, edged


def adaptive_thresh(gray):
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
    )
    return thresh


def save_image(img, path: Path):
    cv2.imwrite(str(path), img)


def quick_ocr(image):
    # simple OCR example on the whole warped image
    config = "--psm 6"  # assume a uniform block of text (clues / printed words)
    text = pytesseract.image_to_string(image, config=config)
    return text


def _smooth1d(arr: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(arr.astype(np.float32), kernel, mode="same")


def _find_peaks(signal: np.ndarray, min_distance: int, threshold_rel: float) -> list[int]:
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


def _detect_grid_lines_by_projection(warped_gray: np.ndarray) -> tuple[list[int], list[int]]:
    """Detect grid line positions using row/column projections.
    Returns (xs, ys): x column indices of vertical lines, y row indices of horizontal lines.
    """
    th = cv2.adaptiveThreshold(
        warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    h, w = th.shape[:2]
    # mild closing to fill broken line gaps predominantly in each direction
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, h // 120)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 120), 1))
    th_v = cv2.morphologyEx(th, cv2.MORPH_CLOSE, v_kernel, iterations=1)
    th_h = cv2.morphologyEx(th, cv2.MORPH_CLOSE, h_kernel, iterations=1)

    col_proj = th_v.sum(axis=0)
    row_proj = th_h.sum(axis=1)
    col_sm = _smooth1d(col_proj, max(5, w // 200))
    row_sm = _smooth1d(row_proj, max(5, h // 200))

    col_min_dist = max(8, w // 40)
    row_min_dist = max(8, h // 40)
    xs = _find_peaks(col_sm, col_min_dist, threshold_rel=0.35)
    ys = _find_peaks(row_sm, row_min_dist, threshold_rel=0.35)

    # merge very close peaks
    def _merge_close(indices: list[int], min_dist: int) -> list[int]:
        if not indices:
            return []
        indices = sorted(indices)
        merged = [indices[0]]
        for idx in indices[1:]:
            if idx - merged[-1] < (min_dist // 2):
                merged[-1] = int((merged[-1] + idx) / 2)
            else:
                merged.append(idx)
        return merged

    xs = _merge_close(xs, col_min_dist)
    ys = _merge_close(ys, row_min_dist)
    return xs, ys


def _detect_grid_intersections(warped_gray):
    """Detect grid line intersections on a warped square puzzle image.
    Returns a list of intersection points (x,y) as floats.
    """
    # Adaptive threshold to emphasise lines
    th = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)

    h, w = warped_gray.shape[:2]
    # detect vertical lines
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(2, w // 40)))
    vert = cv2.erode(th, vert_kernel, iterations=1)
    vert = cv2.dilate(vert, vert_kernel, iterations=2)

    # detect horizontal lines
    horz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(2, h // 40), 1))
    horz = cv2.erode(th, horz_kernel, iterations=1)
    horz = cv2.dilate(horz, horz_kernel, iterations=2)

    # intersections between vertical and horizontal lines
    inter = cv2.bitwise_and(vert, horz)

    # find connected components / centroids
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

    # sort centers into a grid by clustering y and x
    pts = np.array(centers)
    ys = np.sort(pts[:, 1])
    xs = np.sort(pts[:, 0])

    # estimate cell spacing as the median difference between adjacent coords
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

    # group by rows
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

    # for each row, sort pts by x
    grid = []
    for r in rows:
        row_pts = sorted(r["pts"], key=lambda p: p[0])
        grid.append(row_pts)

    # verify roughly rectangular grid: consistent number of cols
    counts = [len(r) for r in grid]
    if not counts:
        return []
    col_count = max(set(counts), key=counts.count)
    # filter rows that don't match majority column count
    grid = [r for r in grid if len(r) >= max(2, col_count - 1)]

    # normalize rows to the same number of cols by selecting equally-spaced points
    final_grid = []
    for r in grid:
        if len(r) == col_count:
            final_grid.append(r)
        else:
            # pick approximate evenly spaced
            xs_row = np.array([p[0] for p in r])
            target_xs = np.linspace(xs_row.min(), xs_row.max(), col_count)
            chosen = []
            for tx in target_xs:
                # pick nearest
                idx = int(np.argmin(np.abs(xs_row - tx)))
                chosen.append(r[idx])
            final_grid.append(chosen)

    # flatten into intersection points in row-major order
    intersections = []
    for r in final_grid:
        for p in r:
            intersections.append((float(p[0]), float(p[1])))

    return intersections


def _extract_digits_from_warp(
    warped,
    warped_gray,
    out_base: Path,
    *,
    max_rois: int = 600,
    roi_timeout: float = 1.0,
    total_timeout: float = 90.0,
    conf_min: int = 45,
):
    """Detect grid intersections, derive cells, extract top-left digit ROIs, OCR each ROI with digit-only config, save CSV and overlay.
    Returns path to CSV or None if grid not found.
    """
    # Use projection peaks for robust line detection
    xs_proj, ys_proj = _detect_grid_lines_by_projection(warped_gray)
    unique_xs = np.array(xs_proj, dtype=int)
    unique_ys = np.array(ys_proj, dtype=int)
    rows = len(unique_ys)
    cols = len(unique_xs)
    if rows < 2 or cols < 2:
        # fallback to intersection-based method
        inter_pts = _detect_grid_intersections(warped_gray)
        if not inter_pts:
            print("No grid intersections found; skipping digit extraction.")
            return None
        pts = np.array(inter_pts)
        ys = np.round(pts[:, 1]).astype(int)
        xs = np.round(pts[:, 0]).astype(int)
        unique_ys = np.unique(ys)
        unique_xs = np.unique(xs)
        rows = len(unique_ys)
        cols = len(unique_xs)
    if rows < 2 or cols < 2:
        print("Detected insufficient grid lines (rows/cols)")
        return None

    # If the grid is unexpectedly large (noisy intersections), thin lines to cap workload
    max_lines = 26  # -> up to 25x25 cells
    if rows > max_lines:
        step = int(np.ceil(rows / max_lines))
        unique_ys_sorted = np.sort(unique_ys)[::step]
        rows = len(unique_ys_sorted)
    else:
        unique_ys_sorted = np.sort(unique_ys)
    if cols > max_lines:
        step = int(np.ceil(cols / max_lines))
        unique_xs_sorted = np.sort(unique_xs)[::step]
        cols = len(unique_xs_sorted)
    else:
        unique_xs_sorted = np.sort(unique_xs)

    # build mapping from (row,col) to point using the sorted (and possibly thinned) line coordinates

    total_cells = (rows - 1) * (cols - 1)
    print(f"[digits] planned cells: {total_cells} (rows={rows-1}, cols={cols-1}), cap={max_rois}, time budget={total_timeout}s")
    csv_path = out_base / "grid_digits_candidates.csv"
    overlay = warped.copy()
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["row", "col", "x", "y", "w", "h", "pred", "conf"])

        processed = 0  # count only OCR attempts, not every cell visited
        start_ts = time.time()
        # Track black cells to decide which white cells start a clue (OCR candidates)
        is_black = np.zeros((rows - 1, cols - 1), dtype=bool)
        # Diagnostics counters
        total_cells = (rows - 1) * (cols - 1)
        cnt_white = 0
        cnt_black = 0
        cnt_start_candidates = 0
        cnt_nonstart_white = 0
        cnt_ocr_attempts = 0
        cnt_blank_solid_skips = 0
        cnt_lowconf_rejects = 0
        cnt_accepts = 0
        for r in range(rows - 1):
            # progress line per row
            print(f"[digits] row {r+1}/{rows-1}")
            for c in range(cols - 1):
                # global caps
                if processed >= max_rois:
                    print("[digits] hit ROI cap; stopping early")
                    break
                if (time.time() - start_ts) > total_timeout:
                    print("[digits] hit total time budget; stopping early")
                    break
                x0 = int(unique_xs_sorted[c])
                y0 = int(unique_ys_sorted[r])
                x1 = int(unique_xs_sorted[c + 1])
                y1 = int(unique_ys_sorted[r + 1])
                wcell = max(3, x1 - x0)
                hcell = max(3, y1 - y0)

                # Skip black cells (solid blocks) by mean intensity
                cell_gray = warped_gray[y0:y1, x0:x1]
                if cell_gray.size == 0:
                    pred = ""
                    conf = -1
                    writer.writerow([r, c, x0, y0, wcell, hcell, pred, conf])
                    txt = pred if pred else "-"
                    cv2.putText(overlay, txt, (x0 + 4, y0 + int(hcell * 0.22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    continue
                cell_is_black = bool(cell_gray.mean() < 55)
                is_black[r, c] = cell_is_black
                if cell_is_black:  # likely a black square
                    cnt_black += 1
                    pred = ""
                    conf = -1
                    writer.writerow([r, c, x0, y0, wcell, hcell, pred, conf])
                    txt = "-"
                    cv2.putText(overlay, txt, (x0 + 4, y0 + int(hcell * 0.22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    continue
                else:
                    cnt_white += 1

                # Only attempt OCR if this white cell likely starts a clue (left or above is black, or at the grid border)
                starts_across = (c == 0) or (is_black[r, c - 1] if c - 1 >= 0 else False)
                starts_down = (r == 0) or (is_black[r - 1, c] if r - 1 >= 0 else False)
                if not (starts_across or starts_down):
                    cnt_nonstart_white += 1
                    pred = ""
                    conf = -1
                    writer.writerow([r, c, x0, y0, wcell, hcell, pred, conf])
                    txt = "-"
                    cv2.putText(overlay, txt, (x0 + 4, y0 + int(hcell * 0.22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    continue
                else:
                    cnt_start_candidates += 1

                # Build a top-left triangular ROI inset away from grid lines
                inset = max(2, int(min(wcell, hcell) * 0.06))
                xi = x0 + inset
                yi = y0 + inset
                roi_w = max(8, int(wcell * 0.34))
                roi_h = max(8, int(hcell * 0.34))
                roi_rect = warped_gray[yi : min(yi + roi_h, y1), xi : min(xi + roi_w, x1)]
                did_attempt = False
                if roi_rect.size == 0:
                    pred = ""
                    conf = -1
                else:
                    # Apply triangular mask to focus on clue-number region
                    hR, wR = roi_rect.shape[:2]
                    mask = np.zeros((hR, wR), dtype=np.uint8)
                    tri = np.array([[0, 0], [wR - 1, 0], [0, hR - 1]], dtype=np.int32)
                    cv2.fillConvexPoly(mask, tri, 255)
                    roi = cv2.bitwise_and(roi_rect, roi_rect, mask=mask)

                    # prepare digit image: resize & threshold
                    target_h = 80
                    scale = target_h / max(roi.shape[0], 1)
                    roi_rs = cv2.resize(roi, (max(8, int(roi.shape[1] * scale)), target_h), interpolation=cv2.INTER_CUBIC)
                    # Build thresholded variants and OCR configs (multi-pass ensemble)
                    variants = []
                    # Base Otsu binary (black digits on white)
                    _, roi_th = cv2.threshold(roi_rs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    variants.append(roi_th)
                    # CLAHE + Otsu + slight dilation
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    eq = clahe.apply(roi_rs)
                    _, th_eq = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    th_eq = cv2.dilate(th_eq, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
                    variants.append(th_eq)
                    # Adaptive Gaussian binary
                    th_ad = cv2.adaptiveThreshold(roi_rs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
                    variants.append(th_ad)

                    ocr_cfgs = [
                        "--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789",
                        "--psm 10 --oem 1 -c tessedit_char_whitelist=0123456789",
                        "--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789",
                    ]

                    pred = ""
                    conf = -1
                    did_attempt = False
                    for img_bin in variants:
                        # quick blank/solid check to skip expensive OCR calls
                        black_ratio = float((img_bin < 128).mean())
                        if black_ratio < 0.02 or black_ratio > 0.70:
                            cnt_blank_solid_skips += 1
                            continue
                        for config in ocr_cfgs:
                            try:
                                cnt_ocr_attempts += 1
                                did_attempt = True
                                data = pytesseract.image_to_data(
                                    img_bin, config=config, output_type=pytesseract.Output.DICT, timeout=roi_timeout
                                )
                            except Exception:
                                data = {"text": [], "conf": []}
                            # gather best text from this pass
                            best_conf = conf
                            best_pred = pred
                            for i, txt in enumerate(data.get("text", [])):
                                t = (txt or "").strip()
                                try:
                                    conf_i = int(data.get("conf", [])[i])
                                except Exception:
                                    try:
                                        conf_i = float(data.get("conf", [])[i])
                                        conf_i = int(conf_i)
                                    except Exception:
                                        conf_i = -1
                                if t and all(ch.isdigit() for ch in t) and 1 <= len(t) <= 3:
                                    if conf_i > best_conf:
                                        best_conf = conf_i
                                        best_pred = t
                            pred, conf = best_pred, best_conf
                            if conf >= conf_min:
                                break
                        if conf >= conf_min:
                            break

                    if conf < conf_min:
                        if did_attempt:
                            cnt_lowconf_rejects += 1
                        pred = ""
                        conf = -1
                    else:
                        cnt_accepts += 1
                # count only OCR attempts toward ROI cap
                if did_attempt:
                    processed += 1

                writer.writerow([r, c, x0, y0, wcell, hcell, pred, conf])
                # annotate overlay
                txt = pred if pred else "-"
                cv2.putText(overlay, txt, (x0 + 4, y0 + int(hcell * 0.22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                # only executed if inner loop didn't break
                pass
            # if we broke inner due to caps, also break outer
            if processed >= max_rois or (time.time() - start_ts) > total_timeout:
                break
    # save overlay
    save_image(overlay, out_base / "grid_digits_overlay.jpg")
    # Diagnostics summary
    print(
        f"[digits] summary: total_cells={total_cells}, white={cnt_white}, black={cnt_black}, "
        f"starts={cnt_start_candidates}, nonstart_white={cnt_nonstart_white}, ocr_attempts={cnt_ocr_attempts}, "
        f"lowconf_rejects={cnt_lowconf_rejects}, blank/solid_skips={cnt_blank_solid_skips}, accepts={cnt_accepts}"
    )
    print(f"Wrote digit candidates CSV -> {csv_path}")
    return csv_path


def process_image(
    path: Path,
    out_dir: Path = OUT_DIR,
    override_bbox: list | None = None,
    manual: bool = False,
    manual_corners: bool = False,
    digits_ocr: bool = False,
    digits_max_rois: int = 600,
    digits_timeout: float = 1.0,
    digits_total_seconds: float = 90.0,
    use_existing_warp: bool = False,
    digits_min_conf: int = 45,
):
    img = load_image(path)
    if img is None:
        print(f"ERROR: unable to read {path}")
        return
    img = resize_max(img, max_dim=1400)
    # compute gray/edged for diagnostics regardless of quad selection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

    pts = None
    if override_bbox:
        # override_bbox can be [x,y,w,h] or [x1,y1,...,x4,y4]
        coords = [float(v) for v in override_bbox]
        if len(coords) == 4:
            x, y, w, h = coords
            pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32")
        elif len(coords) == 8:
            pts = np.array([[coords[0], coords[1]], [coords[2], coords[3]], [coords[4], coords[5]], [coords[6], coords[7]]], dtype="float32")
        else:
            print("override_bbox must be 4 (x,y,w,h) or 8 (x1,y1,...,x4,y4) numbers; ignoring override.")
            pts = None
    # If the user requested manual selection, try it first so they can override
    if pts is None and manual:
        try:
            maybe_pts = manual_select_corners(img) if manual_corners else manual_select_quad(img)
            if maybe_pts is not None:
                pts = maybe_pts
            else:
                # user cancelled selection; fall back to automatic detection
                pts, gray, edged = find_largest_quad(img, try_manual=False)
        except cv2.error:
            # GUI support not available (e.g., opencv-python-headless). Provide guidance.
            print("Interactive selection failed: OpenCV GUI functions are unavailable in this environment.")
            print(f"Fallback: open '{OUT_DIR / path.stem / 'original.jpg'}' and use --bbox x,y,w,h or 8-point coords.")
            # ensure we have gray/edged computed for diagnostics
            pts, gray, edged = find_largest_quad(img, try_manual=False)
    elif pts is None:
        pts, gray, edged = find_largest_quad(img, try_manual=False)

    base = out_dir / path.stem
    base.mkdir(parents=True, exist_ok=True)

    save_image(img, base / "original.jpg")
    save_image(gray, base / "gray.jpg")
    save_image(edged, base / "edged.jpg")

    if pts is None:
        print(f"No puzzle contour found for {path.name} — saved intermediates.")
        thresh = adaptive_thresh(gray)
        save_image(thresh, base / "thresh.jpg")
        return

    # If only digits OCR is requested and an existing warp is present, allow bypassing detection/warp
    if use_existing_warp and (base / "warped.jpg").exists():
        warped = cv2.imread(str(base / "warped.jpg"))
        if warped is None:
            print("Found warped.jpg but could not read it; regenerating warp from detected/selected quad.")
        else:
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            warped_thresh = adaptive_thresh(warped_gray)
            # proceed directly to outputs/OCR
            goto_digits = True
        # if warped not readable, fall through to regenerate

    if 'warped' not in locals():
        warped = four_point_transform(img, pts, dst_size=1000)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped_thresh = adaptive_thresh(warped_gray)

    save_image(warped, base / "warped.jpg")
    save_image(warped_gray, base / "warped_gray.jpg")
    save_image(warped_thresh, base / "warped_thresh.jpg")
    # run digit extraction only if explicitly requested (avoids perceived freeze right after manual selection)
    if digits_ocr:
        try:
            print("[digits] extracting grid-number candidates ...")
            csv_path = _extract_digits_from_warp(
                warped,
                warped_gray,
                base,
                max_rois=digits_max_rois,
                roi_timeout=digits_timeout,
                total_timeout=digits_total_seconds,
                conf_min=digits_min_conf,
            )
            if csv_path:
                print(f"[digits] candidates saved to: {csv_path}")
        except Exception as e:
            print(f"[digits] extraction failed: {e}")

    ocr_text = quick_ocr(warped_gray)
    (base / "ocr.txt").write_text(ocr_text, encoding="utf-8")
    print(f"Processed {path.name} → outputs in {base}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess crossword photos (find puzzle, warp, threshold, quick OCR).")
    parser.add_argument("images", nargs="*", help="Path(s) to image files. If empty, all examples in src/examples will be processed.")
    parser.add_argument("--out", "-o", default=str(OUT_DIR), help="Output directory (default: data/temp)")
    parser.add_argument("--open", action="store_true", help="Open the output folder for the last processed image (macOS `open`).")
    parser.add_argument("--bbox", help="Override quad by specifying 4 numbers x,y,w,h or 8 numbers x1,y1,...,x4,y4 (comma-separated)")
    parser.add_argument("--manual", action="store_true", help="Open interactive ROI selector (drag box)")
    parser.add_argument("--manual-corners", action="store_true", help="Open 4-corner click selector (click 4 corners, u=undo, r=reset, ESC/c cancels)")
    parser.add_argument("--digits-ocr", action="store_true", help="Also run digit OCR on the grid numbers and write CSV + overlay")
    parser.add_argument("--digits-max-rois", type=int, default=600, help="Max number of digit ROIs to OCR (cap)")
    parser.add_argument("--digits-timeout", type=float, default=1.0, help="Per-ROI Tesseract timeout in seconds")
    parser.add_argument("--digits-total-seconds", type=float, default=90.0, help="Total time budget for digit OCR in seconds")
    parser.add_argument("--use-existing-warp", action="store_true", help="Use existing warped.jpg in the output folder (skip detection/warp)")
    parser.add_argument("--digits-min-conf", type=int, default=45, help="Minimum OCR confidence to accept a digit prediction (default: 45)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.images:
        for img_path in args.images:
            targets.append(Path(img_path))
    else:
        examples = list(EXAMPLES_DIR.glob("*.jpg")) + list(EXAMPLES_DIR.glob("*.JPG")) + list(EXAMPLES_DIR.glob("*.png"))
        targets = examples

    if not targets:
        print(f"No images found. Provide paths or add images to {EXAMPLES_DIR}")
        return 1

    last_base = None
    for target in targets:
        target = Path(target)
        if not target.exists():
            print(f"Skipping missing file: {target}")
            continue
        # parse bbox override once per run
        override = None
        if args.bbox:
            try:
                parts = [p for p in args.bbox.replace(" ", "").split(",") if p]
                override = parts
            except Exception:
                print("Could not parse --bbox; expected comma-separated numbers.")
                override = None
        process_image(
            target,
            out_dir=out_dir,
            override_bbox=override,
            manual=args.manual or args.manual_corners,
            manual_corners=args.manual_corners,
            digits_ocr=args.digits_ocr,
            digits_max_rois=args.digits_max_rois,
            digits_timeout=args.digits_timeout,
            digits_total_seconds=args.digits_total_seconds,
            use_existing_warp=args.use_existing_warp,
            digits_min_conf=args.digits_min_conf,
        )
        last_base = out_dir / target.stem

    if args.open and last_base is not None:
        # macOS: open the folder with Finder
        try:
            os.system(f'open "{last_base}"')
        except Exception:
            print(f"Could not open {last_base}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
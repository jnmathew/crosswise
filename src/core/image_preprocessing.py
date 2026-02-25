"""
Image preprocessing utilities for crossword puzzle extraction.

Pipeline:
1. Load and resize image
2. Detect puzzle quadrilateral (multi-strategy)
3. Apply perspective transform to warp to square
4. Generate grayscale and binary outputs
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from .config import Settings


def load_image(path: Path) -> np.ndarray:
    """Load image from path."""
    return cv2.imread(str(path))


def resize_max(image: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    """Resize image so maximum dimension is max_dim while preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def four_point_transform(image: np.ndarray, pts: np.ndarray, dst_size: int = 1000) -> np.ndarray:
    """
    Apply perspective transform to warp quadrilateral to square.

    Args:
        image: Input image
        pts: 4x2 array of corner points
        dst_size: Output image size (square)

    Returns:
        Warped square image of size (dst_size, dst_size)
    """
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


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in clockwise order: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: 4x2 array of points

    Returns:
        Ordered 4x2 array
    """
    rect = np.zeros((4, 2), dtype="float32")
    sums = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sums)]      # top-left (smallest sum)
    rect[2] = pts[np.argmax(sums)]      # bottom-right (largest sum)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]      # top-right (smallest diff)
    rect[3] = pts[np.argmax(diff)]      # bottom-left (largest diff)
    return rect


def correct_rotation(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Detect and correct rotation in warped grid image to make grid lines perfectly horizontal/vertical.

    Args:
        image: Warped image (grayscale or color)

    Returns:
        Tuple of (rotated image, rotation angle in degrees)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None or len(lines) < 10:
        logger.debug("Not enough lines detected for rotation correction, skipping")
        return image, 0.0

    # Separate horizontal and vertical lines
    h_angles = []
    v_angles = []

    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta)

        # Horizontal lines: angle near 0° or 180°
        if (angle_deg < 10 or angle_deg > 170):
            h_angles.append(theta)
        # Vertical lines: angle near 90°
        elif (80 < angle_deg < 100):
            v_angles.append(theta)

    # Calculate rotation angle
    rotation_angle = 0.0

    if h_angles:
        # Use horizontal lines to determine rotation
        median_h = np.median(h_angles)
        rotation_angle = np.degrees(median_h)
        # Normalize to [-45, 45] range
        if rotation_angle > 45:
            rotation_angle -= 180
    elif v_angles:
        # Use vertical lines to determine rotation
        median_v = np.median(v_angles)
        rotation_angle = np.degrees(median_v) - 90
        # Normalize to [-45, 45] range
        if rotation_angle > 45:
            rotation_angle -= 180
        elif rotation_angle < -45:
            rotation_angle += 180

    # Only apply rotation if it's significant (> 0.1 degrees)
    if abs(rotation_angle) < 0.1:
        logger.debug(f"Rotation angle {rotation_angle:.3f}° is negligible, skipping correction")
        return image, 0.0

    # Apply rotation
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    logger.info(f"Applied rotation correction: {rotation_angle:.3f}°")
    return rotated, rotation_angle


def detect_page_boundaries(image: np.ndarray, min_area_ratio: float = 0.3) -> Optional[np.ndarray]:
    """
    Detect newspaper/document page boundaries for perspective correction.

    Uses multiple strategies to find the page quad:
    1. Edge detection + largest quadrilateral contour
    2. Threshold-based detection (light page on dark background)
    3. Text density analysis (find bounding region of text content)

    Args:
        image: Input BGR image
        min_area_ratio: Minimum area of detected quad as fraction of image (default 0.3)

    Returns:
        4x2 array of corner points (TL, TR, BR, BL), or None if not detected
    """
    h, w = image.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Strategy 1: Edge-based detection
    # Good for scans with visible page edges against scanner bed
    logger.debug("Page detection: Trying edge-based strategy")
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    # Dilate edges to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Sort by area, try largest contours first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:5]:  # Check top 5 largest
            area = cv2.contourArea(contour)
            if area < img_area * min_area_ratio:
                continue

            # Approximate to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype("float32")
                logger.info("Page detection: Found quad via edge detection")
                return order_points(quad)

    # Strategy 2: Threshold-based (light content on dark background)
    logger.debug("Page detection: Trying threshold-based strategy")

    # Assume page content is lighter than background (scanner bed, dark margins)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up with morphological operations
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large)
    thresh_opened = cv2.morphologyEx(thresh_closed, cv2.MORPH_OPEN, kernel_large)

    contours, _ = cv2.findContours(thresh_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area >= img_area * min_area_ratio:
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype("float32")
                logger.info("Page detection: Found quad via threshold detection")
                return order_points(quad)

            # If not a clean quad, use minAreaRect
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect).astype("float32")
            logger.info("Page detection: Found quad via minAreaRect")
            return order_points(box)

    # Strategy 3: Text density - find bounding box of text regions
    logger.debug("Page detection: Trying text density strategy")

    # Detect text regions using MSER or simple threshold
    _, text_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find bounding box of all white pixels (text)
    coords = cv2.findNonZero(text_thresh)
    if coords is not None and len(coords) > 100:
        rect = cv2.minAreaRect(coords)
        box = cv2.boxPoints(rect).astype("float32")

        # Check if area is sufficient
        box_area = cv2.contourArea(box)
        if box_area >= img_area * min_area_ratio:
            logger.info("Page detection: Found quad via text density")
            return order_points(box)

    logger.warning("Page detection: Could not detect page boundaries")
    return None


def correct_page_perspective(
    image: np.ndarray,
    manual_corners: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Detect page boundaries and apply perspective correction to straighten the page.

    Unlike grid warping (which outputs a fixed 1000x1000), this preserves the
    original aspect ratio while correcting perspective distortion.

    Args:
        image: Input BGR image
        manual_corners: Optional 4x2 array of manually specified corners

    Returns:
        Tuple of:
        - Corrected image (or original if no correction needed)
        - Detected/used quad points (or None)
        - Metadata dict with correction info
    """
    h, w = image.shape[:2]
    meta = {
        "correction_applied": False,
        "detection_method": None,
        "quad_points": None
    }

    # Use manual corners if provided
    if manual_corners is not None:
        quad = order_points(manual_corners.astype("float32"))
        meta["detection_method"] = "manual"
    else:
        # Auto-detect page boundaries
        quad = detect_page_boundaries(image)
        if quad is None:
            logger.info("No page boundaries detected, skipping perspective correction")
            return image, None, meta
        meta["detection_method"] = "auto"

    meta["quad_points"] = quad.tolist()

    # Check if correction is actually needed
    # If the quad is nearly rectangular and axis-aligned, skip correction
    ordered = quad  # Already ordered: TL, TR, BR, BL
    tl, tr, br, bl = ordered

    # Calculate angles of top and bottom edges
    top_angle = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    bottom_angle = np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))
    left_angle = np.degrees(np.arctan2(bl[1] - tl[1], bl[0] - tl[0]))
    right_angle = np.degrees(np.arctan2(br[1] - tr[1], br[0] - tr[0]))

    # Check if edges are nearly horizontal/vertical (within 1 degree)
    horizontal_ok = abs(top_angle) < 1.0 and abs(bottom_angle) < 1.0
    vertical_ok = abs(left_angle - 90) < 1.0 and abs(right_angle - 90) < 1.0

    if horizontal_ok and vertical_ok:
        logger.info("Page already well-aligned, skipping perspective correction")
        return image, quad, meta

    # Calculate output dimensions preserving aspect ratio
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    out_w = int(max(width_top, width_bottom))
    out_h = int(max(height_left, height_right))

    # Destination points for perspective transform
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype="float32")

    # Apply perspective transform
    matrix = cv2.getPerspectiveTransform(quad, dst)
    corrected = cv2.warpPerspective(image, matrix, (out_w, out_h))

    meta["correction_applied"] = True
    meta["output_size"] = (out_w, out_h)
    meta["angles"] = {
        "top": float(top_angle),
        "bottom": float(bottom_angle),
        "left": float(left_angle),
        "right": float(right_angle)
    }

    logger.info(f"Applied perspective correction: {w}x{h} -> {out_w}x{out_h}")
    return corrected, quad, meta


def _contour_quad_from_mask(mask: np.ndarray, approx_eps: float = 0.02) -> Optional[np.ndarray]:
    """
    Extract largest quadrilateral contour from binary mask.

    If the largest contour has >4 vertices (e.g., L-shape from two adjacent
    grids connected by a fold) but is significantly larger than the first
    clean quad, extracts the largest 4-vertex sub-quad from its vertices.

    Args:
        mask: Binary mask image
        approx_eps: Epsilon for contour approximation (fraction of perimeter)

    Returns:
        4x2 array of corner points, or None if no quad found
    """
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find the first clean quad via polygon approximation
    first_quad = None
    first_quad_area = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, approx_eps * peri, True)
        if len(approx) == 4:
            first_quad = approx.reshape(4, 2).astype("float32")
            first_quad_area = cv2.contourArea(c)
            break

    # Check if the largest contour has >4 vertices but is significantly
    # larger than the first clean quad. This happens when folds/creases
    # create L-shaped contours connecting two adjacent grids.
    largest = contours[0]
    largest_area = cv2.contourArea(largest)
    largest_peri = cv2.arcLength(largest, True)
    largest_approx = cv2.approxPolyDP(largest, approx_eps * largest_peri, True)
    n_verts = len(largest_approx)

    if (
        5 <= n_verts <= 8
        and largest_area > first_quad_area * 1.5
    ):
        # Try all 4-vertex subsets to find the largest convex sub-quad
        from itertools import combinations

        verts = largest_approx.reshape(-1, 2).astype("float32")
        best_sub_area = 0
        best_sub_quad = None

        for combo in combinations(range(n_verts), 4):
            quad = verts[list(combo)]
            area = abs(cv2.contourArea(quad))
            if area <= best_sub_area:
                continue
            # Must be convex (no self-intersections)
            if len(cv2.convexHull(quad, returnPoints=False)) == 4:
                best_sub_area = area
                best_sub_quad = quad

        if best_sub_quad is not None and best_sub_area > first_quad_area * 1.3:
            logger.debug(
                f"Extracted sub-quad (area={best_sub_area:.0f}) from {n_verts}-vertex "
                f"contour, {best_sub_area / first_quad_area:.1f}x larger than first quad"
            )
            return best_sub_quad

    return first_quad


def _is_valid_quad(pts: np.ndarray, image_shape: Tuple[int, int, int], min_area_ratio: float = 0.05) -> bool:
    """
    Check if quad is valid (not too small).

    Args:
        pts: 4x2 array of points
        image_shape: Shape of source image
        min_area_ratio: Minimum area as fraction of image area

    Returns:
        True if quad is valid
    """
    img_area = image_shape[0] * image_shape[1]
    area = abs(cv2.contourArea(pts.reshape(4, 1, 2)))
    if area <= 0:
        return False
    ratio = area / float(img_area)
    return ratio >= min_area_ratio


def _line_intersection(line1: Tuple[float, float, float, float],
                       line2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float]]:
    """
    Find intersection point of two lines.

    Args:
        line1: (x1, y1, x2, y2) for first line
        line2: (x1, y1, x2, y2) for second line

    Returns:
        (x, y) intersection point, or None if lines are parallel
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom

    return (px, py)


def find_quad_with_hough(image: np.ndarray, approx_quad: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Find puzzle quad using Hough line detection, optionally refined from an approximate quad.

    This method detects actual grid lines rather than the outer boundary,
    producing better perspective correction for skewed images.

    Algorithm:
    1. If approx_quad provided, create a mask to focus on that region
    2. Detect all lines using HoughLinesP
    3. Classify lines as horizontal or vertical (by angle)
    4. Find the most consistent grid cluster (lines with regular spacing)
    5. Select outer boundary lines of the main grid
    6. Compute 4 intersections = quad corners

    Args:
        image: Input BGR image
        approx_quad: Optional approximate quad from contour detection to focus search

    Returns:
        4x2 array of corner points, or None if detection fails
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Create mask if approximate quad provided
    mask = None
    if approx_quad is not None:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [approx_quad.astype(np.int32)], 255)
        logger.debug("Hough: Using approximate quad to focus search")

    # Strong edge detection to find grid lines
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 150)

    # Apply mask if provided
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Hough line detection with stricter parameters
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=80,  # Lower threshold to catch more lines
        minLineLength=max(50, min(h, w) // 15),
        maxLineGap=30
    )

    if lines is None or len(lines) == 0:
        logger.debug("Hough: No lines detected")
        return None

    # Classify lines as horizontal or vertical based on angle
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        # Calculate angle
        if x2 - x1 == 0:
            angle = 90.0
        else:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        # Horizontal: angle close to 0 or 180
        if angle < 15 or angle > 165:
            horizontal_lines.append((x1, y1, x2, y2, length))
        # Vertical: angle close to 90
        elif 75 < angle < 105:
            vertical_lines.append((x1, y1, x2, y2, length))

    logger.debug(f"Hough: {len(horizontal_lines)} horizontal, {len(vertical_lines)} vertical lines")

    if len(horizontal_lines) < 4 or len(vertical_lines) < 4:
        logger.debug("Hough: Insufficient lines for grid detection")
        return None

    # Cluster horizontal lines by y-position (adaptive tolerance based on expected grid cell size)
    # For a typical crossword, cells are about 1000/20 = 50px, so tolerance should be ~10-15px
    h_tolerance = max(10, min(h * 0.02, 20))  # 10-20px, max 2% of image height
    h_clusters = []
    for x1, y1, x2, y2, length in horizontal_lines:
        y_avg = (y1 + y2) / 2
        placed = False

        for cluster in h_clusters:
            if abs(cluster['y_avg'] - y_avg) < h_tolerance:
                cluster['lines'].append((x1, y1, x2, y2, length))
                cluster['y_sum'] += length
                cluster['y_avg'] = np.mean([l[1] + l[3] for l in cluster['lines']]) / 2
                placed = True
                break

        if not placed:
            h_clusters.append({
                'y_avg': y_avg,
                'lines': [(x1, y1, x2, y2, length)],
                'y_sum': length
            })

    # Cluster vertical lines by x-position
    v_tolerance = max(10, min(w * 0.02, 20))
    v_clusters = []
    for x1, y1, x2, y2, length in vertical_lines:
        x_avg = (x1 + x2) / 2
        placed = False

        for cluster in v_clusters:
            if abs(cluster['x_avg'] - x_avg) < v_tolerance:
                cluster['lines'].append((x1, y1, x2, y2, length))
                cluster['x_sum'] += length
                cluster['x_avg'] = np.mean([l[0] + l[2] for l in cluster['lines']]) / 2
                placed = True
                break

        if not placed:
            v_clusters.append({
                'x_avg': x_avg,
                'lines': [(x1, y1, x2, y2, length)],
                'x_sum': length
            })

    # Sort clusters by position
    h_clusters.sort(key=lambda c: c['y_avg'])
    v_clusters.sort(key=lambda c: c['x_avg'])

    logger.debug(f"Hough: {len(h_clusters)} h_clusters, {len(v_clusters)} v_clusters")

    if len(h_clusters) < 4 or len(v_clusters) < 4:
        logger.debug("Hough: Insufficient line clusters for grid")
        return None

    # Use all detected clusters - assume they're all part of the grid
    # The clustering already filtered out outliers by requiring tight spacing tolerance
    h_start, h_end = 0, len(h_clusters)
    v_start, v_end = 0, len(v_clusters)

    logger.debug(f"Hough: Grid bounds h[{h_start}:{h_end}], v[{v_start}:{v_end}]")

    # Select boundary lines from the main grid region
    top_h = max(h_clusters[h_start]['lines'], key=lambda l: l[4])[:4]
    bottom_h = max(h_clusters[h_end-1]['lines'], key=lambda l: l[4])[:4]
    left_v = max(v_clusters[v_start]['lines'], key=lambda l: l[4])[:4]
    right_v = max(v_clusters[v_end-1]['lines'], key=lambda l: l[4])[:4]

    # Compute 4 intersections
    tl = _line_intersection(top_h, left_v)
    tr = _line_intersection(top_h, right_v)
    br = _line_intersection(bottom_h, right_v)
    bl = _line_intersection(bottom_h, left_v)

    if None in [tl, tr, br, bl]:
        logger.debug("Hough: Failed to compute all intersections")
        return None

    quad = np.array([tl, tr, br, bl], dtype="float32")

    # Validate quad is reasonable (higher threshold than contour methods)
    if not _is_valid_quad(quad, image.shape, min_area_ratio=0.15):
        logger.debug("Hough: Quad failed validation (area too small)")
        return None

    logger.debug("Hough: Successfully detected quad from grid lines")
    return quad


def adaptive_thresh(gray: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding to grayscale image."""
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
    )
    return thresh


def find_largest_quad(image: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Try multiple strategies to find the puzzle quad.

    Strategies:
      A) Contour + Hough refinement (best - combines robustness with accuracy)
      B) Hough line detection alone (for clear grids)
      C) Canny edges + contour approx (fast fallback)
      D) Adaptive threshold + morphological close + contour approx
      E) minAreaRect on largest contour (last resort)

    Args:
        image: Input BGR image

    Returns:
        (quad_points, gray, edged) where:
            - quad_points: 4x2 float32 array or None if not found
            - gray: Grayscale version of image
            - edged: Edge detection image for debugging
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    # Strategy A: Get approximate quad from contours, refine with Hough
    logger.debug("Trying Strategy A: Contour + Hough refinement")

    # Try contour detection first
    approx_quad = _contour_quad_from_mask(edged, approx_eps=0.02)
    if approx_quad is None or not _is_valid_quad(approx_quad, image.shape):
        # Try with adaptive threshold
        thresh = adaptive_thresh(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        approx_quad = _contour_quad_from_mask(closed, approx_eps=0.03)

    # If we have an approximate quad, refine it with Hough
    if approx_quad is not None and _is_valid_quad(approx_quad, image.shape):
        logger.debug("Got approximate quad from contours, refining with Hough")
        refined_quad = find_quad_with_hough(image, approx_quad=approx_quad)
        if refined_quad is not None:
            logger.info("Strategy A succeeded: Contour + Hough refinement")
            return refined_quad, gray, edged
        else:
            logger.debug("Hough refinement failed, using approximate quad")
            logger.info("Strategy A succeeded (contour only): Using approximate quad")
            return approx_quad, gray, edged

    # Strategy B: Hough line detection alone
    logger.debug("Trying Strategy B: Hough line detection alone")
    quad = find_quad_with_hough(image)
    if quad is not None:
        logger.info("Strategy B succeeded: Hough line detection alone")
        return quad, gray, edged

    # Strategy C: blurred Canny + contour approx (fallback)
    logger.debug("Trying Strategy C: Canny edges + contour")
    quad = _contour_quad_from_mask(edged, approx_eps=0.02)
    if quad is not None and _is_valid_quad(quad, image.shape):
        logger.info("Strategy C succeeded: Canny edges")
        return quad, gray, edged

    # Strategy D: adaptive threshold + morphological close
    logger.debug("Trying Strategy D: Adaptive threshold + morph close")
    thresh = adaptive_thresh(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    quad = _contour_quad_from_mask(closed, approx_eps=0.03)
    if quad is not None and _is_valid_quad(quad, image.shape):
        edged_from_thresh = cv2.Canny(cv2.GaussianBlur(closed, (3, 3), 0), 30, 100)
        logger.info("Strategy D succeeded: Adaptive threshold + morph close")
        return quad, gray, edged_from_thresh

    # Strategy E: largest contour -> minAreaRect fallback
    logger.debug("Trying Strategy E: minAreaRect on largest contour")
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect).astype("float32")
        if _is_valid_quad(box, image.shape):
            logger.info("Strategy E succeeded: minAreaRect fallback")
            return box, gray, edged

    logger.warning("All quad detection strategies failed")
    return None, gray, edged


def preprocess(input_path: Path, config: Settings, manual_quad: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Load image, detect puzzle quad, warp to square.

    Args:
        input_path: Path to input image
        config: Settings object
        manual_quad: Optional 4x2 array of manually specified corner points (TL, TR, BR, BL)

    Returns:
        Dictionary with:
            - original: np.ndarray (resized original)
            - warped: np.ndarray (1000x1000 color warped)
            - warped_gray: np.ndarray (grayscale warped)
            - warped_thresh: np.ndarray (binary threshold warped)
            - quad_points: np.ndarray (4x2 corner points)
            - meta: dict (processing metadata)

    Raises:
        ValueError: If image cannot be read or quad not found
    """
    logger.info(f"Preprocessing {input_path.name}")

    # Load and resize
    image = load_image(input_path)
    if image is None:
        raise ValueError(f"Cannot read image: {input_path}")

    original = resize_max(image, max_dim=1400)
    logger.debug(f"Resized to {original.shape[:2]}")

    # Find puzzle quad (use manual if provided, otherwise auto-detect)
    if manual_quad is not None:
        quad = manual_quad
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        logger.info("Using manually specified quad corners")
    else:
        quad, gray, edged = find_largest_quad(original)
        if quad is None:
            raise ValueError("Could not detect puzzle quadrilateral")

    # Calculate quad area ratio for metadata
    img_area = original.shape[0] * original.shape[1]
    quad_area = abs(cv2.contourArea(quad.reshape(4, 1, 2)))
    area_ratio = quad_area / img_area

    logger.info(f"Detected quad with area ratio: {area_ratio:.2%}")

    # Warp to square
    warped = four_point_transform(original, quad, dst_size=1000)

    # Apply rotation correction to align grid perfectly
    # TEMPORARILY DISABLED - applying too much rotation
    # warped, rotation_angle = correct_rotation(warped)
    rotation_angle = 0.0

    # Generate grayscale and threshold images
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_thresh = adaptive_thresh(warped_gray)

    logger.success("Preprocessing complete")

    return {
        "original": original,
        "warped": warped,
        "warped_gray": warped_gray,
        "warped_thresh": warped_thresh,
        "quad_points": quad,
        "meta": {
            "strategy_used": "multi-strategy",  # Could track which strategy worked
            "quad_area_ratio": float(area_ratio),
            "max_dimension": max(original.shape[:2]),
            "rotation_correction_degrees": float(rotation_angle)
        }
    }

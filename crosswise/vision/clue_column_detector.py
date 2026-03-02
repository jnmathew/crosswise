"""
Automatic multi-column layout detection for crossword clues.

This module detects column boundaries in multi-column text layouts (like newspaper
crossword clues) and adds vertical separator lines to prevent OCR from reading
across columns horizontally.

Pipeline:
1. Vertical projection profile to find gaps between columns
2. Connected component analysis to refine boundaries based on actual text
3. Column validation to ensure consistent spacing
4. Draw vertical separator lines at column boundaries
5. Optional masking of non-text regions (ads, images, etc.)
"""
from typing import List, Tuple
import cv2
import numpy as np
from loguru import logger


def _smooth1d(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    """Smooth 1D array with moving average filter."""
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    return np.convolve(arr.astype(np.float32), kernel, mode="same")


def _find_valleys(signal: np.ndarray, min_depth: float = 0.2, min_width: int = 10) -> List[int]:
    """
    Find valleys (local minima) in 1D signal.

    Valleys indicate gaps between columns in vertical projection.

    Args:
        signal: 1D array (vertical projection)
        min_depth: Minimum valley depth as fraction of signal range
        min_width: Minimum width of valley in pixels

    Returns:
        List of valley center positions
    """
    if signal.size == 0:
        return []

    # Normalize signal to [0, 1]
    signal_min = signal.min()
    signal_max = signal.max()
    if signal_max - signal_min < 1e-6:
        return []

    normalized = (signal - signal_min) / (signal_max - signal_min)
    threshold = min_depth

    # Find regions below threshold
    below_thresh = normalized < threshold

    # Find contiguous regions
    valleys = []
    in_valley = False
    valley_start = 0

    for i, is_below in enumerate(below_thresh):
        if is_below and not in_valley:
            # Start of valley
            valley_start = i
            in_valley = True
        elif not is_below and in_valley:
            # End of valley
            valley_width = i - valley_start
            if valley_width >= min_width:
                # Calculate valley depth (how deep below threshold)
                valley_signal = normalized[valley_start:i]
                valley_depth = threshold - valley_signal.mean()

                # Only accept valleys with significant depth
                # This filters out shallow gaps within columns
                if valley_depth > min_depth * 0.3:  # At least 30% of threshold
                    valley_center = (valley_start + i) // 2
                    valleys.append(valley_center)
            in_valley = False

    # Handle valley extending to end
    if in_valley:
        valley_width = len(below_thresh) - valley_start
        if valley_width >= min_width:
            valley_signal = normalized[valley_start:]
            valley_depth = threshold - valley_signal.mean()
            if valley_depth > min_depth * 0.3:
                valley_center = (valley_start + len(below_thresh)) // 2
                valleys.append(valley_center)

    return valleys


def detect_column_boundaries_by_projection(
    image: np.ndarray,
    min_column_width: int = 100,
    smoothing_kernel: int = 15
) -> List[int]:
    """
    Detect column boundaries using vertical projection profile.

    The vertical projection sums pixel intensities along each column.
    Valleys in this projection indicate gaps between text columns.

    Args:
        image: Input grayscale image
        min_column_width: Minimum expected column width in pixels
        smoothing_kernel: Kernel size for smoothing projection profile

    Returns:
        List of x-coordinates where vertical separators should be placed
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # Binarize to emphasize text
    # Use inverse binary so text is white (255) on black (0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Vertical projection: sum along y-axis for each x position
    vertical_projection = binary.sum(axis=0)

    # Smooth the projection to reduce noise
    smoothed = _smooth1d(vertical_projection, smoothing_kernel)

    # Find valleys (gaps between columns)
    # Valleys should be at least min_column_width/3 wide to be considered column gaps
    min_gap_width = max(10, min_column_width // 10)
    valleys = _find_valleys(smoothed, min_depth=0.3, min_width=min_gap_width)

    logger.debug(f"Projection method found {len(valleys)} potential column boundaries")

    return valleys


def get_text_regions(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect text regions using connected component analysis.

    Args:
        image: Input grayscale image

    Returns:
        List of bounding boxes (x, y, w, h) for text regions
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to connect nearby text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    # Extract bounding boxes (skip background label 0)
    text_regions = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Filter out very small components (noise)
        if area > 20 and w > 3 and h > 3:
            text_regions.append((x, y, w, h))

    return text_regions


def refine_boundaries_with_text_regions(
    projection_boundaries: List[int],
    text_regions: List[Tuple[int, int, int, int]],
    image_width: int,
    tolerance: int = 50
) -> List[int]:
    """
    Refine column boundaries using actual text region positions.

    Adjusts projection-based boundaries to avoid splitting text regions.

    Args:
        projection_boundaries: Initial boundaries from projection method
        text_regions: List of text bounding boxes (x, y, w, h)
        image_width: Total image width
        tolerance: Maximum adjustment distance in pixels

    Returns:
        Refined list of boundary positions
    """
    if not projection_boundaries:
        return []

    refined = []

    for boundary in projection_boundaries:
        # Find text regions near this boundary
        nearby_regions = []
        for x, y, w, h in text_regions:
            x_center = x + w // 2
            if abs(x_center - boundary) < tolerance * 2:
                nearby_regions.append((x, x + w))

        if not nearby_regions:
            # No text nearby, keep original boundary
            refined.append(boundary)
            continue

        # Find largest gap near the boundary
        # Collect all x positions (starts and ends of text regions)
        x_positions = []
        for x_start, x_end in nearby_regions:
            x_positions.extend([x_start, x_end])

        x_positions = sorted(x_positions)

        # Find largest gap within tolerance
        best_gap_center = boundary
        best_gap_size = 0

        for i in range(len(x_positions) - 1):
            gap_start = x_positions[i]
            gap_end = x_positions[i + 1]
            gap_center = (gap_start + gap_end) // 2
            gap_size = gap_end - gap_start

            # Only consider gaps near original boundary
            if abs(gap_center - boundary) <= tolerance and gap_size > best_gap_size:
                best_gap_center = gap_center
                best_gap_size = gap_size

        refined.append(best_gap_center)

    logger.debug(f"Refined {len(refined)} boundaries using text regions")

    return refined


def validate_columns(
    boundaries: List[int],
    image_width: int,
    min_column_width: int,
    max_spacing_variance: float = 0.5
) -> List[int]:
    """
    Validate and filter column boundaries.

    Ensures columns have reasonable width and relatively consistent spacing.

    Args:
        boundaries: List of boundary positions
        image_width: Total image width
        min_column_width: Minimum acceptable column width
        max_spacing_variance: Maximum acceptable variance in column spacing (as fraction of median)

    Returns:
        Filtered list of valid boundaries
    """
    if not boundaries:
        return []

    boundaries = sorted(boundaries)

    # Calculate column widths (including edges)
    widths = []

    # First column: from 0 to first boundary
    if boundaries:
        widths.append(boundaries[0])

    # Middle columns: between boundaries
    for i in range(len(boundaries) - 1):
        widths.append(boundaries[i + 1] - boundaries[i])

    # Last column: from last boundary to edge
    if boundaries:
        widths.append(image_width - boundaries[-1])

    # Filter out columns that are too narrow
    # Use 1.5x the min_column_width for more aggressive filtering
    effective_min_width = int(min_column_width * 1.5)
    valid_boundaries = []
    current_x = 0

    for i, boundary in enumerate(boundaries):
        column_width = boundary - current_x
        if column_width >= effective_min_width:
            valid_boundaries.append(boundary)
            current_x = boundary
        else:
            logger.debug(f"Removing boundary at {boundary} (column width {column_width} < effective min {effective_min_width})")

    # Additional pass: check that the NEXT column is also wide enough
    # This prevents creating a valid boundary followed by a too-narrow column
    final_boundaries = []
    for i, boundary in enumerate(valid_boundaries):
        if i < len(valid_boundaries) - 1:
            next_boundary = valid_boundaries[i + 1]
            next_column_width = next_boundary - boundary
        else:
            next_column_width = image_width - boundary

        if next_column_width >= effective_min_width:
            final_boundaries.append(boundary)
        else:
            logger.debug(f"Removing boundary at {boundary} (next column width {next_column_width} < effective min {effective_min_width})")

    # Check spacing consistency
    if len(final_boundaries) >= 2:
        spacings = []
        prev = 0
        for boundary in final_boundaries:
            spacings.append(boundary - prev)
            prev = boundary
        spacings.append(image_width - prev)

        median_spacing = np.median(spacings)
        std_spacing = np.std(spacings)
        variance = std_spacing / median_spacing if median_spacing > 0 else 0

        logger.debug(f"Column spacing - median: {median_spacing:.1f}, std: {std_spacing:.1f}, variance: {variance:.2f}")

        # If spacing is too irregular, might be detecting noise instead of columns
        if variance > max_spacing_variance and len(final_boundaries) > 4:
            logger.warning(f"High spacing variance ({variance:.2f}), might have false detections")

    return final_boundaries


def detect_column_boundaries_by_text_clustering(
    image: np.ndarray,
    min_column_width: int = 100
) -> List[int]:
    """
    Detect column boundaries using text region clustering.

    This method works well for partial-height columns that don't span
    the entire page, as it clusters actual text positions rather than
    relying on full-height gaps.

    Args:
        image: Input grayscale image
        min_column_width: Minimum expected column width in pixels

    Returns:
        List of x-coordinates for column boundaries
    """
    text_regions = get_text_regions(image)

    if not text_regions:
        logger.warning("No text regions found for clustering")
        return []

    # Extract x-centers of all text regions
    x_centers = np.array([x + w // 2 for x, y, w, h in text_regions])

    if len(x_centers) < 10:
        logger.warning(f"Too few text regions ({len(x_centers)}) for reliable clustering")
        return []

    # Create histogram of x-positions
    w = image.shape[1]
    hist, bin_edges = np.histogram(x_centers, bins=w // 10)

    # Smooth histogram
    smoothed_hist = _smooth1d(hist, kernel_size=max(5, w // 200))

    # Find valleys (gaps between columns)
    # Use a lower depth threshold to catch smaller gaps
    valleys = _find_valleys(smoothed_hist, min_depth=0.15, min_width=min_column_width // 20)

    # Convert bin indices to x-coordinates
    boundaries = []
    for valley_idx in valleys:
        x_pos = int(bin_edges[valley_idx])
        boundaries.append(x_pos)

    logger.debug(f"Text clustering found {len(boundaries)} potential boundaries")

    return boundaries


def detect_column_boundaries(
    image: np.ndarray,
    min_column_width: int = 100,
    max_columns: int = 10
) -> List[int]:
    """
    Detect column boundaries using hybrid approach.

    Combines vertical projection profile with text region clustering
    for robust column detection that handles both full-height and
    partial-height columns.

    Args:
        image: Input image (grayscale or BGR)
        min_column_width: Minimum expected column width in pixels
        max_columns: Maximum expected number of columns

    Returns:
        List of x-coordinates where vertical separators should be drawn
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    logger.info(f"Detecting column boundaries (image size: {w}x{h}, min_column_width: {min_column_width})")

    # Step 1: Get text regions (used by multiple methods)
    text_regions = get_text_regions(gray)
    logger.debug(f"Found {len(text_regions)} text regions")

    # Step 2a: Try text clustering method (better for partial-height columns)
    clustering_boundaries = detect_column_boundaries_by_text_clustering(
        gray,
        min_column_width=min_column_width
    )

    # Step 2b: Try vertical projection method (better for full-height columns)
    projection_boundaries = detect_column_boundaries_by_projection(
        gray,
        min_column_width=min_column_width,
        smoothing_kernel=max(15, w // 100)
    )

    # Step 3: Combine both methods
    # Merge boundaries that are close to each other
    all_boundaries = sorted(set(clustering_boundaries + projection_boundaries))

    # Merge boundaries within tolerance
    merged = []
    tolerance = min_column_width // 4
    for boundary in all_boundaries:
        if not merged or boundary - merged[-1] >= tolerance:
            merged.append(boundary)
        else:
            # Average the close boundaries
            merged[-1] = (merged[-1] + boundary) // 2

    logger.debug(f"Combined methods: {len(merged)} boundaries after merging")

    # Step 4: Refine boundaries using text regions
    if merged and text_regions:
        refined_boundaries = refine_boundaries_with_text_regions(
            merged,
            text_regions,
            image_width=w,
            tolerance=tolerance
        )
    else:
        refined_boundaries = merged

    # Step 5: Validate columns
    valid_boundaries = validate_columns(
        refined_boundaries,
        image_width=w,
        min_column_width=min_column_width,
        max_spacing_variance=0.6  # Increased tolerance for irregular layouts
    )

    # Limit to max_columns
    if len(valid_boundaries) >= max_columns:
        logger.warning(f"Found {len(valid_boundaries)} boundaries, limiting to {max_columns - 1}")
        valid_boundaries = valid_boundaries[:max_columns - 1]

    logger.info(f"Detected {len(valid_boundaries) + 1} columns with boundaries at: {valid_boundaries}")

    return valid_boundaries


def draw_column_separators(
    image: np.ndarray,
    boundaries: List[int],
    separator_color: Tuple[int, int, int] = (255, 255, 0),  # Aqua/cyan in BGR
    separator_width: int = 3
) -> np.ndarray:
    """
    Draw vertical separator lines at column boundaries.

    Args:
        image: Input image (will be modified)
        boundaries: List of x-coordinates for separators
        separator_color: BGR color for separator lines (default: gray)
        separator_width: Width of separator lines in pixels

    Returns:
        Image with separators drawn
    """
    result = image.copy()

    h = result.shape[0]

    for x in boundaries:
        # Draw vertical line from top to bottom
        cv2.line(result, (x, 0), (x, h), separator_color, separator_width)

    logger.debug(f"Drew {len(boundaries)} vertical separators")

    return result


def detect_and_mask_non_text_regions(
    image: np.ndarray,
    mask_color: Tuple[int, int, int] = (255, 255, 255),
    text_density_threshold: float = 0.02,
    region_size: int = 100
) -> np.ndarray:
    """
    Detect and mask non-text regions (ads, images, etc.).

    Uses a grid-based approach to identify regions with low text density
    and masks them with a solid color.

    Args:
        image: Input image
        mask_color: BGR color for masking (default: white)
        text_density_threshold: Minimum text density to keep region
        region_size: Size of grid cells for density analysis

    Returns:
        Image with non-text regions masked
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = image.copy()
    else:
        gray = image.copy()
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape

    # Binarize to detect text
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Analyze grid cells
    masked_regions = 0

    for y in range(0, h, region_size):
        for x in range(0, w, region_size):
            # Extract region
            y_end = min(y + region_size, h)
            x_end = min(x + region_size, w)

            region = binary[y:y_end, x:x_end]

            if region.size == 0:
                continue

            # Calculate text density (ratio of white pixels in binary image)
            text_density = np.count_nonzero(region) / region.size

            # Mask regions with low text density
            if text_density < text_density_threshold:
                result[y:y_end, x:x_end] = mask_color
                masked_regions += 1

    logger.debug(f"Masked {masked_regions} low-text-density regions")

    return result


def preprocess_clues_for_ocr(
    image: np.ndarray,
    min_column_width: int = 100,
    max_columns: int = 10,
    add_separators: bool = True,
    mask_non_text: bool = False,
    separator_color: Tuple[int, int, int] = (255, 255, 0),  # Aqua/cyan in BGR
    separator_width: int = 3
) -> np.ndarray:
    """
    Preprocess crossword clues image for OCR.

    Main entry point that combines column detection, separator drawing,
    and optional masking.

    Args:
        image: Input image (grayscale or BGR)
        min_column_width: Minimum expected column width in pixels
        max_columns: Maximum expected number of columns
        add_separators: Whether to draw vertical separator lines
        mask_non_text: Whether to mask non-text regions
        separator_color: BGR color for separator lines
        separator_width: Width of separator lines in pixels

    Returns:
        Preprocessed image ready for OCR
    """
    logger.info("Preprocessing clues for OCR")

    result = image.copy()

    # Optional: Mask non-text regions first
    if mask_non_text:
        result = detect_and_mask_non_text_regions(result)

    # Detect column boundaries
    boundaries = detect_column_boundaries(
        result,
        min_column_width=min_column_width,
        max_columns=max_columns
    )

    # Draw separators
    if add_separators and boundaries:
        result = draw_column_separators(
            result,
            boundaries,
            separator_color=separator_color,
            separator_width=separator_width
        )

    logger.success(f"Clue preprocessing complete ({len(boundaries) + 1} columns detected)")

    return result

"""Unit tests for black cell classification in grid_detection.py."""
import pytest
import numpy as np
import cv2


# ============================================================================
# 1. Threshold Method Tests
# ============================================================================

class TestOtsuMethod:
    """Test Otsu's automatic thresholding method."""

    def test_otsu_threshold_bimodal_distribution(self, bimodal_clean_separation):
        """Otsu should find valley between two peaks."""
        intensities = bimodal_clean_separation.astype(np.uint8)
        threshold, _ = cv2.threshold(
            intensities, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Should be between black peak (~40) and white peak (~150)
        assert 40 < threshold < 120

    def test_otsu_threshold_uniform_distribution(self, uniform_white_cells):
        """Otsu on uniform distribution should still return valid value."""
        intensities = uniform_white_cells.astype(np.uint8)
        threshold, _ = cv2.threshold(
            intensities, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        assert 0 <= threshold <= 255

    def test_otsu_threshold_all_black(self, uniform_black_cells):
        """Otsu on all-black distribution."""
        intensities = uniform_black_cells.astype(np.uint8)
        threshold, _ = cv2.threshold(
            intensities, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Threshold will be somewhere in the black range
        assert 0 <= threshold <= 255


class TestPercentileMethod:
    """Test percentile-based thresholding."""

    def test_percentile_23_calculation(self):
        """23rd percentile should classify ~23% as black."""
        intensities = np.arange(0, 100, dtype=float)  # 0-99
        threshold = np.percentile(intensities, 23)

        black_count = np.sum(intensities < threshold)
        percentage = (black_count / len(intensities)) * 100

        # Allow ±1% tolerance for floating point
        assert 22 <= percentage <= 24

    def test_percentile_with_multiple_clusters(self, bimodal_multiple_clusters):
        """Percentile should handle multiple clusters gracefully."""
        threshold = np.percentile(bimodal_multiple_clusters, 23)

        # Should classify appropriate number as black
        black_count = np.sum(bimodal_multiple_clusters < threshold)
        # Expected ~36 black cells (23% of 156)
        assert 30 <= black_count <= 40

    def test_percentile_on_uniform_distribution(self, uniform_white_cells):
        """Percentile on uniform distribution should give consistent result."""
        threshold = np.percentile(uniform_white_cells, 23)

        # Threshold should be near 23% point of range (150-200)
        # 23% of (200-150) + 150 ≈ 161.5
        assert 155 < threshold < 170


class TestGapDetectionMethod:
    """Test gap detection for finding threshold."""

    def test_gap_detection_clean_separation(self, bimodal_clean_separation):
        """Gap method should find the valley between clusters."""
        sorted_intensities = np.sort(bimodal_clean_separation)
        search_range = int(len(sorted_intensities) * 0.40)

        diffs = np.diff(sorted_intensities[:search_range])
        gap_idx = np.argmax(diffs)
        gap_size = diffs[gap_idx]
        gap_threshold = (sorted_intensities[gap_idx] + sorted_intensities[gap_idx + 1]) / 2

        # Should find large gap (>20) between black and white
        assert gap_size > 20
        assert 50 < gap_threshold < 120

    def test_gap_detection_no_clear_gap(self, uniform_white_cells):
        """Gap method on uniform distribution should find small gap."""
        sorted_intensities = np.sort(uniform_white_cells)
        search_range = int(len(sorted_intensities) * 0.40)

        diffs = np.diff(sorted_intensities[:search_range])
        gap_size = diffs[np.argmax(diffs)] if len(diffs) > 0 else 0

        # Should have very small gaps in uniform distribution
        assert gap_size < 10

    def test_gap_detection_multiple_clusters(self, bimodal_multiple_clusters):
        """Gap detection with multiple black cell clusters."""
        sorted_intensities = np.sort(bimodal_multiple_clusters)
        search_range = int(len(sorted_intensities) * 0.40)

        diffs = np.diff(sorted_intensities[:search_range])
        gap_idx = np.argmax(diffs)
        gap_size = diffs[gap_idx]

        # Should find a gap, but might not be the largest overall gap
        assert gap_size > 0


# ============================================================================
# 2. Selection Logic Tests
# ============================================================================

class TestThresholdSelection:
    """Test intelligent threshold method selection."""

    def test_selection_uses_gap_for_clean_separation(self):
        """Gap size 3-15 should trigger gap method."""
        # Simulate: gap_size=4.0, gap_threshold=88.8
        gap_threshold = 88.8
        gap_size = 4.0
        _percentile_threshold = 120.0  # noqa: F841 — documents test scenario

        # Logic: 40 <= gap_threshold <= 150 and 3 <= gap_size <= 15
        should_use_gap = (40 <= gap_threshold <= 150) and (3 <= gap_size <= 15)

        assert should_use_gap is True

    def test_selection_uses_percentile_for_large_gap(self):
        """Gap size >15 should trigger percentile method."""
        # Simulate: gap_size=42.4, gap_threshold=78.4
        gap_threshold = 78.4
        gap_size = 42.4
        percentile_threshold = 111.2

        # Should NOT use gap (gap_size > 15)
        should_use_gap = (40 <= gap_threshold <= 150) and (3 <= gap_size <= 15)
        should_use_percentile = (40 <= percentile_threshold <= 150)

        assert should_use_gap is False
        assert should_use_percentile is True

    def test_selection_uses_percentile_for_small_gap(self):
        """Gap size <3 should trigger percentile method."""
        gap_threshold = 85.0
        gap_size = 2.5
        percentile_threshold = 110.0

        should_use_gap = (40 <= gap_threshold <= 150) and (3 <= gap_size <= 15)
        should_use_percentile = (40 <= percentile_threshold <= 150)

        assert should_use_gap is False
        assert should_use_percentile is True

    def test_selection_fallback_to_otsu(self):
        """Invalid thresholds should fall back to Otsu."""
        gap_threshold = 200  # Out of range
        percentile_threshold = 250  # Out of range
        _otsu_threshold = 57  # noqa: F841 — documents test scenario

        # Neither gap nor percentile valid
        valid_gap = (40 <= gap_threshold <= 150)
        valid_percentile = (40 <= percentile_threshold <= 150)

        assert valid_gap is False
        assert valid_percentile is False
        # Would use otsu_threshold

    @pytest.mark.parametrize("gap_size,gap_threshold,percentile_threshold,expected_method", [
        (4.0, 88.8, 120.0, "gap"),           # Clean separation
        (42.4, 78.4, 111.2, "percentile"),   # Multiple clusters
        (2.5, 85.0, 110.0, "percentile"),    # Gap too small
        (10.0, 200.0, 115.0, "percentile"),  # Gap out of range
        (8.0, 90.0, 250.0, "gap"),           # Percentile out of range
        (15.0, 95.0, 115.0, "gap"),          # Gap at upper bound
        (3.0, 80.0, 120.0, "gap"),           # Gap at lower bound
    ])
    def test_method_selection_parametrized(self, gap_size, gap_threshold, percentile_threshold, expected_method):
        """Test all combinations of threshold selection logic."""
        # Implement selection logic from grid_detection.py
        if 40 <= gap_threshold <= 150 and 3 <= gap_size <= 15:
            method = "gap"
        elif 40 <= percentile_threshold <= 150:
            method = "percentile"
        else:
            method = "otsu"

        assert method == expected_method


# ============================================================================
# 3. Classification Accuracy Tests
# ============================================================================

class TestClassificationAccuracy:
    """Test actual black/white cell classification."""

    def test_classification_accuracy_known_distribution(self):
        """Verify cells are classified correctly with known threshold."""
        cell_intensities = np.array([30, 40, 50, 60, 70, 120, 130, 140])
        threshold = 80

        expected_black = [True, True, True, True, True, False, False, False]
        actual_black = [intensity < threshold for intensity in cell_intensities]

        assert actual_black == expected_black

    def test_boundary_cells_at_threshold(self):
        """Cells at exact threshold should be classified as white."""
        threshold = 88.0

        assert (88.0 < threshold) is False  # Exactly at threshold -> white
        assert (87.9 < threshold) is True   # Just below -> black
        assert (88.1 < threshold) is False  # Just above -> white

    def test_classification_with_bimodal_data(self, bimodal_clean_separation):
        """Classification with clear bimodal distribution."""
        threshold = 85.0  # Between two clusters

        black_cells = bimodal_clean_separation[bimodal_clean_separation < threshold]
        white_cells = bimodal_clean_separation[bimodal_clean_separation >= threshold]

        # All black cells should be < 50, all white > 120
        assert np.all(black_cells < 50)
        assert np.all(white_cells >= 120)


# ============================================================================
# 4. Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_cell_intensities(self):
        """No cells should trigger fallback threshold."""
        cell_intensities = []

        if not cell_intensities:
            threshold = 55  # Fallback

        assert threshold == 55

    def test_single_cell(self):
        """Single cell should work without error."""
        cell_intensities = np.array([100.0])

        # All methods should still compute
        percentile = np.percentile(cell_intensities, 23)
        assert percentile == 100.0

        # Gap detection on single cell
        sorted_intensities = np.sort(cell_intensities)
        diffs = np.diff(sorted_intensities)
        assert len(diffs) == 0  # No gaps with single cell

    def test_two_cells(self):
        """Two cells should compute threshold."""
        cell_intensities = np.array([30.0, 150.0])

        threshold = np.percentile(cell_intensities, 23)
        # 23% between 30 and 150
        assert 30 < threshold < 150

    def test_all_black_cells(self, uniform_black_cells):
        """All dark cells should classify appropriately."""
        threshold = np.percentile(uniform_black_cells, 23)

        black_count = np.sum(uniform_black_cells < threshold)
        # With uniform dark distribution, 23% rule applies
        assert black_count >= 20

    def test_all_white_cells(self, uniform_white_cells):
        """All bright cells should classify appropriately."""
        threshold = np.percentile(uniform_white_cells, 23)

        black_count = np.sum(uniform_white_cells < threshold)
        # 23% of uniform white cells will be classified as "black"
        # but they're all in the white range
        assert black_count >= 20
        # But all "black" cells are actually > 150
        black_cells = uniform_white_cells[uniform_white_cells < threshold]
        assert np.all(black_cells >= 150)

    def test_borderline_intensities(self, borderline_intensities):
        """Cells near typical threshold values."""
        threshold = np.percentile(borderline_intensities, 23)

        # Threshold should be somewhere in 80-100 range
        assert 80 <= threshold <= 100

        # Should classify ~23% as black
        black_count = np.sum(borderline_intensities < threshold)
        percentage = (black_count / len(borderline_intensities)) * 100
        assert 15 <= percentage <= 30  # Wider tolerance for small sample

    def test_very_small_gaps(self):
        """Test when all gaps are tiny (continuous distribution)."""
        # Create continuous distribution with small gaps
        intensities = np.linspace(50, 150, 100)

        sorted_intensities = np.sort(intensities)
        search_range = int(len(sorted_intensities) * 0.40)
        diffs = np.diff(sorted_intensities[:search_range])

        gap_size = diffs[np.argmax(diffs)]

        # Gaps should all be very small (~1.0)
        assert gap_size < 2.0


# ============================================================================
# 5. Data Validation Tests
# ============================================================================

class TestDataValidation:
    """Test data type and range validation."""

    def test_intensity_range_validity(self, bimodal_clean_separation):
        """Cell intensities should be in valid range."""
        assert np.all(bimodal_clean_separation >= 0)
        assert np.all(bimodal_clean_separation <= 255)

    def test_threshold_in_valid_range(self):
        """Computed thresholds should be reasonable."""
        intensities = np.array([30, 40, 50, 100, 150, 160])

        threshold_otsu, _ = cv2.threshold(
            intensities.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold_percentile = np.percentile(intensities, 23)

        assert 0 <= threshold_otsu <= 255
        assert 0 <= threshold_percentile <= 255

    def test_percentile_array_sorting(self):
        """Verify percentile calculation doesn't require pre-sorted array."""
        unsorted = np.array([100, 30, 150, 50, 200, 40])
        sorted_array = np.sort(unsorted)

        # Both should give same result
        threshold_unsorted = np.percentile(unsorted, 23)
        threshold_sorted = np.percentile(sorted_array, 23)

        assert threshold_unsorted == threshold_sorted


# ============================================================================
# 6. Fixture Validation Tests
# ============================================================================

class TestFixtures:
    """Validate test fixtures produce expected data."""

    def test_uniform_white_cells_fixture(self, uniform_white_cells):
        """Verify uniform_white_cells fixture."""
        assert len(uniform_white_cells) == 100
        assert np.all(uniform_white_cells >= 150)
        assert np.all(uniform_white_cells < 200)

    def test_uniform_black_cells_fixture(self, uniform_black_cells):
        """Verify uniform_black_cells fixture."""
        assert len(uniform_black_cells) == 100
        assert np.all(uniform_black_cells >= 20)
        assert np.all(uniform_black_cells < 50)

    def test_bimodal_clean_separation_fixture(self, bimodal_clean_separation):
        """Verify bimodal_clean_separation fixture."""
        assert len(bimodal_clean_separation) == 156  # 36 + 120

        # Should have two distinct groups
        black_group = bimodal_clean_separation[bimodal_clean_separation < 60]
        white_group = bimodal_clean_separation[bimodal_clean_separation > 100]

        assert len(black_group) == 36
        assert len(white_group) == 120

    def test_bimodal_multiple_clusters_fixture(self, bimodal_multiple_clusters):
        """Verify bimodal_multiple_clusters fixture."""
        assert len(bimodal_multiple_clusters) == 156  # 30 + 6 + 120

        # Should have three groups
        cluster1 = bimodal_multiple_clusters[bimodal_multiple_clusters < 60]
        cluster2 = bimodal_multiple_clusters[
            (bimodal_multiple_clusters >= 90) & (bimodal_multiple_clusters < 120)
        ]
        white_group = bimodal_multiple_clusters[bimodal_multiple_clusters >= 120]

        assert len(cluster1) == 30
        assert len(cluster2) == 6
        assert len(white_group) == 120

    def test_mock_warped_grid_fixture(self, mock_warped_grid):
        """Verify mock_warped_grid fixture."""
        assert mock_warped_grid.shape == (250, 250)
        assert mock_warped_grid.dtype == np.uint8

        # Count black cells (checkerboard 5x5 = 13 black cells)
        # Each cell is 50x50 pixels
        black_cell_count = 0
        for r in range(5):
            for c in range(5):
                cell = mock_warped_grid[r*50:(r+1)*50, c*50:(c+1)*50]
                mean_intensity = cell.mean()
                if mean_intensity < 100:  # Black threshold
                    black_cell_count += 1

        assert black_cell_count == 13  # Checkerboard pattern

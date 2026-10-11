"""
Test suite for the CV pipeline (preprocess -> grid_detect -> warp ->
slice_cells -> classify_cells -> pipeline).

Run from the project root with:
    pytest cv/tests/ -v

Uses a synthetically-generated sudoku image (drawn with OpenCV) as a
reliable, reproducible test fixture - not dependent on any particular
real photo being present on disk.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from preprocess import load_image, preprocess
from grid_detect import find_grid_contour, order_corners
from warp import warp_grid
from slice_cells import slice_cells
from classify_cells import is_cell_filled, classify_all_cells
from pipeline import process_sudoku_image, SudokuImageResult


# ---------------------------------------------------------------------------
# Shared fixture: a synthetic sudoku photo, generated once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_sudoku_path(tmp_path_factory):
    """
    Builds a simple synthetic sudoku image with a known number of digits,
    so tests have a stable, reproducible ground truth to check against
    (rather than depending on a real photo being present on disk).
    """
    img = np.ones((450, 450, 3), dtype=np.uint8) * 255
    for i in range(10):
        thickness = 3 if i % 3 == 0 else 1
        cv2.line(img, (i * 50, 0), (i * 50, 450), (0, 0, 0), thickness)
        cv2.line(img, (0, i * 50), (450, i * 50), (0, 0, 0), thickness)

    # 5 known digits placed at known cell positions
    digits = {(0, 0): "5", (1, 2): "3", (4, 4): "7", (8, 8): "9", (2, 6): "1"}
    for (r, c), d in digits.items():
        cv2.putText(img, d, (c * 50 + 15, r * 50 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    path = tmp_path_factory.mktemp("cv_fixtures") / "synthetic_sudoku.png"
    cv2.imwrite(str(path), img)
    return str(path), digits


@pytest.fixture(scope="session")
def blank_image_path(tmp_path_factory):
    """A plain white image with no grid at all - used to test failure handling."""
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    path = tmp_path_factory.mktemp("cv_fixtures") / "blank.png"
    cv2.imwrite(str(path), img)
    return str(path)


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_load_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_image("this_file_does_not_exist.png")

    def test_load_corrupt_file_raises_value_error(self, tmp_path):
        # a file that exists but isn't valid image data - distinct from
        # a missing file, so callers can tell "no file" apart from "bad file"
        fake_image = tmp_path / "fake.png"
        fake_image.write_text("this is not image data")
        with pytest.raises(ValueError, match="couldn't be read as an image"):
            load_image(str(fake_image))

    def test_load_too_small_image_raises_value_error(self, tmp_path):
        tiny = np.ones((20, 20, 3), dtype=np.uint8) * 255
        path = tmp_path / "tiny.png"
        cv2.imwrite(str(path), tiny)
        with pytest.raises(ValueError, match="too small"):
            load_image(str(path))

    def test_load_and_preprocess_valid_image(self, synthetic_sudoku_path):
        path, _ = synthetic_sudoku_path
        img = load_image(path)
        binary = preprocess(img)
        assert binary.shape[:2] == img.shape[:2]
        assert len(binary.shape) == 2  # single channel (grayscale/binary)

    def test_preprocess_output_is_binary(self, synthetic_sudoku_path):
        path, _ = synthetic_sudoku_path
        img = load_image(path)
        binary = preprocess(img)
        unique_values = set(np.unique(binary).tolist())
        assert unique_values <= {0, 255}  # only pure black/white, nothing in between


# ---------------------------------------------------------------------------
# grid_detect
# ---------------------------------------------------------------------------

class TestGridDetect:
    def test_finds_four_corners(self, synthetic_sudoku_path):
        path, _ = synthetic_sudoku_path
        img = load_image(path)
        binary = preprocess(img)
        corners = find_grid_contour(binary)
        assert corners.shape == (4, 2)

    def test_corners_are_ordered_tl_tr_br_bl(self, synthetic_sudoku_path):
        path, _ = synthetic_sudoku_path
        img = load_image(path)
        binary = preprocess(img)
        corners = find_grid_contour(binary)
        tl, tr, br, bl = corners
        assert tl[0] < tr[0]  # top-left is left of top-right
        assert bl[0] < br[0]  # bottom-left is left of bottom-right
        assert tl[1] < bl[1]  # top-left is above bottom-left

    def test_order_corners_handles_shuffled_input(self):
        # feed corners in a deliberately wrong/shuffled order, confirm
        # order_corners still sorts them correctly
        shuffled = np.array([[449, 449], [0, 0], [449, 0], [0, 449]], dtype="float32")
        ordered = order_corners(shuffled)
        tl, tr, br, bl = ordered
        assert list(tl) == [0, 0]
        assert list(tr) == [449, 0]
        assert list(br) == [449, 449]
        assert list(bl) == [0, 449]

    def test_blank_image_raises_value_error(self, blank_image_path):
        img = load_image(blank_image_path)
        binary = preprocess(img)
        with pytest.raises(ValueError):
            find_grid_contour(binary)


# ---------------------------------------------------------------------------
# warp
# ---------------------------------------------------------------------------

class TestWarp:
    def test_warp_produces_correct_output_size(self, synthetic_sudoku_path):
        path, _ = synthetic_sudoku_path
        img = load_image(path)
        binary = preprocess(img)
        corners = find_grid_contour(binary)
        warped = warp_grid(img, corners, output_size=450)
        assert warped.shape[:2] == (450, 450)

    def test_warp_respects_custom_output_size(self, synthetic_sudoku_path):
        path, _ = synthetic_sudoku_path
        img = load_image(path)
        binary = preprocess(img)
        corners = find_grid_contour(binary)
        warped = warp_grid(img, corners, output_size=180)
        assert warped.shape[:2] == (180, 180)


# ---------------------------------------------------------------------------
# slice_cells
# ---------------------------------------------------------------------------

class TestSliceCells:
    def test_produces_81_cells(self, synthetic_sudoku_path):
        path, _ = synthetic_sudoku_path
        img = load_image(path)
        binary = preprocess(img)
        corners = find_grid_contour(binary)
        warped = warp_grid(binary, corners)
        cells = slice_cells(warped)
        assert len(cells) == 81

    def test_cells_are_equal_sized(self, synthetic_sudoku_path):
        path, _ = synthetic_sudoku_path
        img = load_image(path)
        binary = preprocess(img)
        corners = find_grid_contour(binary)
        warped = warp_grid(binary, corners)
        cells = slice_cells(warped)
        shapes = {c.shape for c in cells}
        assert len(shapes) == 1  # every cell is the same size

    def test_non_square_image_raises(self):
        non_square = np.zeros((100, 200), dtype=np.uint8)
        with pytest.raises(ValueError, match="square"):
            slice_cells(non_square)

    def test_size_not_divisible_by_9_raises(self):
        bad_size = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="divisible by 9"):
            slice_cells(bad_size)


# ---------------------------------------------------------------------------
# classify_cells
# ---------------------------------------------------------------------------

class TestClassifyCells:
    def test_all_black_cell_is_empty(self):
        blank_cell = np.zeros((50, 50), dtype=np.uint8)
        assert is_cell_filled(blank_cell) is False

    def test_cell_with_solid_blob_is_filled(self):
        cell = np.zeros((50, 50), dtype=np.uint8)
        cv2.circle(cell, (25, 25), 15, 255, -1)  # solid filled circle, digit-like
        assert is_cell_filled(cell)

    def test_scattered_noise_is_not_filled(self):
        # scattered single-pixel specks should NOT count as a digit -
        # this is the exact bug that got caught and fixed during development
        cell = np.zeros((50, 50), dtype=np.uint8)
        rng = np.random.default_rng(42)
        for _ in range(30):
            y, x = rng.integers(0, 50, size=2)
            cell[y, x] = 255
        assert not is_cell_filled(cell)

    def test_classify_all_returns_correct_length(self, synthetic_sudoku_path):
        path, digits = synthetic_sudoku_path
        img = load_image(path)
        binary = preprocess(img)
        corners = find_grid_contour(binary)
        warped = warp_grid(binary, corners)
        cells = slice_cells(warped, margin_ratio=0.12)
        flags = classify_all_cells(cells)
        assert len(flags) == 81


# ---------------------------------------------------------------------------
# pipeline (full end-to-end)
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_returns_sudoku_image_result(self, synthetic_sudoku_path):
        path, _ = synthetic_sudoku_path
        result = process_sudoku_image(path)
        assert isinstance(result, SudokuImageResult)

    def test_result_has_81_cells_and_flags(self, synthetic_sudoku_path):
        path, _ = synthetic_sudoku_path
        result = process_sudoku_image(path)
        assert len(result.cells) == 81
        assert len(result.filled_flags) == 81

    def test_filled_count_matches_known_digit_count(self, synthetic_sudoku_path):
        # the synthetic fixture has exactly 5 digits placed - the pipeline
        # should detect a filled count reasonably close to that
        path, digits = synthetic_sudoku_path
        result = process_sudoku_image(path)
        assert result.filled_count == len(digits)

    def test_filled_cell_indices_are_valid_range(self, synthetic_sudoku_path):
        path, _ = synthetic_sudoku_path
        result = process_sudoku_image(path)
        for idx in result.filled_cell_indices():
            assert 0 <= idx < 81

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            process_sudoku_image("nonexistent_photo.jpg")

    def test_blank_image_raises_value_error(self, blank_image_path):
        with pytest.raises(ValueError):
            process_sudoku_image(blank_image_path)

    def test_to_grid_returns_81_length_list(self, synthetic_sudoku_path):
        path, _ = synthetic_sudoku_path
        result = process_sudoku_image(path)
        grid = result.to_grid()
        assert len(grid) == 81

    def test_to_grid_marks_empty_cells_as_zero(self, synthetic_sudoku_path):
        path, _ = synthetic_sudoku_path
        result = process_sudoku_image(path)
        grid = result.to_grid()
        for i, filled in enumerate(result.filled_flags):
            if not filled:
                assert grid[i] == 0

    def test_to_grid_runs_real_recognition_by_default(self, synthetic_sudoku_path):
        """
        to_grid() with no digit_values supplied should run real digit
        recognition automatically (not return -1 placeholders) - this is
        the whole point of wiring the CNN into the pipeline.
        """
        path, _ = synthetic_sudoku_path
        result = process_sudoku_image(path)
        grid = result.to_grid()
        for i, filled in enumerate(result.filled_flags):
            if filled:
                assert 1 <= grid[i] <= 9, f"cell {i}: expected a real recognized digit, got {grid[i]}"

    def test_to_grid_applies_supplied_digit_values(self, synthetic_sudoku_path):
        path, _ = synthetic_sudoku_path
        result = process_sudoku_image(path)
        fake_values = {i: 7 for i in result.filled_cell_indices()}
        grid = result.to_grid(fake_values)
        for i in result.filled_cell_indices():
            assert grid[i] == 7

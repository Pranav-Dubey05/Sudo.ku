"""
Sub-step 6: package the full CV pipeline into one clean function.

This is the single entry point the FastAPI backend (main roadmap step 4)
will actually call: photo in -> list of 81 cell images out, each tagged
with whether it's empty or contains a digit.

Ties together every earlier sub-step:
    1. preprocess    - clean binary image
    2. grid_detect   - find the grid's 4 corners
    3. warp          - straighten into a flat square
    4. slice_cells   - split into 81 individual cells
    5. classify_cells - tag each cell empty/filled

Digit recognition itself (reading *which* digit) is intentionally NOT
part of this file - that's a separate model/step (main roadmap step 3),
kept isolated so this module's only job is finding and isolating cells.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_image, preprocess
from grid_detect import find_grid_contour
from warp import warp_grid
from slice_cells import slice_cells
from classify_cells import classify_all_cells

_digit_model = None  # lazy-loaded singleton - avoid reloading the model on every call


def _get_digit_model():
    """
    Loads the trained CNN once and reuses it across calls. Digit
    recognition is optional: if torch or the model file aren't
    available, callers can still use the CV pipeline (grid detection,
    cell slicing) without digit recognition - to_grid() just falls
    back to -1 placeholders in that case, same as before.
    """
    global _digit_model
    if _digit_model is None:
        import torch
        from train_cnn import DigitCNN

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_cnn_mega.pt")
        model = DigitCNN()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        _digit_model = model
    return _digit_model


class SudokuImageResult:
    """
    Holds everything extracted from one sudoku photo.

    Attributes:
        cells: list of 81 cell images (row-major order, index = row*9+col)
        filled_flags: list of 81 booleans, True = cell has a digit
        warped_color: the straightened color image (useful for debug overlays)
        corners: the 4 detected grid corners in the original photo
    """

    def __init__(self, cells, filled_flags, warped_color, corners):
        self.cells = cells
        self.filled_flags = filled_flags
        self.warped_color = warped_color
        self.corners = corners

    @property
    def filled_count(self):
        return sum(self.filled_flags)

    def filled_cell_indices(self):
        """Indices (0-80) of cells that contain a digit - what digit recognition needs to process."""
        return [i for i, f in enumerate(self.filled_flags) if f]

    def recognize_digits(self):
        """
        Runs the trained CNN on every filled cell, returns a dict
        {cell_index: recognized_digit}. This is what turns the CV
        pipeline's "here's where the digits are" output into actual
        digit values.
        """
        from train_cnn import predict_digit

        model = _get_digit_model()
        digit_values = {}
        for idx in self.filled_cell_indices():
            digit, confidence = predict_digit(model, self.cells[idx])
            digit_values[idx] = digit
        return digit_values

    def to_grid(self, digit_values=None):
        """
        Convert this result into the flat 81-int grid shape sudoku_core.py
        expects (0 = blank). This is the handoff point between the CV
        pipeline and the solver/generator module.

        Args:
            digit_values: optional dict mapping cell index -> recognized
                digit (1-9). If not provided, digit recognition runs
                automatically via recognize_digits() - so calling
                to_grid() with no arguments gives you a real, usable
                grid, not placeholders.

        Returns:
            list of 81 ints: 0 = empty, 1-9 = recognized digit.
        """
        if digit_values is None:
            digit_values = self.recognize_digits()

        grid = []
        for i, filled in enumerate(self.filled_flags):
            if not filled:
                grid.append(0)
            else:
                grid.append(digit_values.get(i, -1))
        return grid


def process_sudoku_image(path):
    """
    The single entry point for the whole CV pipeline.

    Args:
        path: path to a photo of a sudoku puzzle.

    Returns:
        SudokuImageResult

    Raises:
        FileNotFoundError: if the image can't be loaded.
        ValueError: if no plausible grid could be found in the image.
    """
    img = load_image(path)
    binary = preprocess(img)
    corners = find_grid_contour(binary)

    warped_binary = warp_grid(binary, corners)
    warped_color = warp_grid(img, corners)

    cells = slice_cells(warped_binary, margin_ratio=0.12)
    filled_flags = classify_all_cells(cells)

    return SudokuImageResult(cells, filled_flags, warped_color, corners)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to_sudoku_photo>")
        sys.exit(1)

    # allow importing sudoku_core.py from the project root, one level up
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sudoku_core import Sudoku

    path = sys.argv[1]
    result = process_sudoku_image(path)

    print(f"Pipeline complete for: {path}")
    print(f"Cells extracted: {len(result.cells)}")
    print(f"Filled: {result.filled_count} / 81  |  Empty: {81 - result.filled_count}")

    print("\nRunning digit recognition...")
    grid = result.to_grid()

    print("\nRecognized grid:")
    for r in range(9):
        row = grid[r * 9:(r + 1) * 9]
        print(" ".join(str(v) if v != 0 else "." for v in row))

    print("\nHanding off to the solver...")
    try:
        game = Sudoku(grid)
        solution = game.solve()
        print("\nSOLVED:")
        for r in range(9):
            row = solution[r * 9:(r + 1) * 9]
            print(" ".join(str(v) for v in row))
        print("\nEnd-to-end pipeline succeeded: photo -> digits -> solved puzzle.")
    except ValueError as e:
        print(f"\nCould not solve: {e}")
        print("(This usually means a digit was misread - digit recognition isn't perfect.)")

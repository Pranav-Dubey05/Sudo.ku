"""
The missing link: photo in -> solved sudoku out, using everything built
so far chained together for the first time:

    process_sudoku_image()  (CV pipeline: find grid, slice cells, classify)
        -> predict_digit()  (CNN: read each filled cell's actual digit)
            -> Sudoku(grid) (validates the recognized grid is legal)
                -> .solve() (backtracking solver)

This is the real end-to-end proof the whole project works together, not
just three components that each work in isolation.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import process_sudoku_image
from train_cnn import DigitCNN, predict_digit
from sudoku_core import Sudoku

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_cnn_mega.pt")

_model_cache = None


def load_model(path=DEFAULT_MODEL_PATH):
    """Loads the digit recognition model once and caches it - avoids
    reloading weights from disk on every call in a long-running server."""
    global _model_cache
    if _model_cache is None:
        model = DigitCNN()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        _model_cache = model
    return _model_cache


class SolveResult:
    """
    Everything about one solve attempt: the recognized puzzle, the
    solution (if solvable), and diagnostics about cells the model
    wasn't confident about - useful for surfacing "please double check
    this cell" in a UI, rather than silently trusting every prediction.
    """

    def __init__(self, puzzle, solution, low_confidence_cells, clues, error=None):
        self.puzzle = puzzle
        self.solution = solution
        self.low_confidence_cells = low_confidence_cells
        self.clues = clues
        self.error = error

    @property
    def success(self):
        return self.error is None and self.solution is not None


def solve_photo(image_path, model=None, confidence_threshold=0.5):
    """
    The full pipeline, start to finish. Returns a SolveResult - check
    .success before trusting .solution, since a misread digit can produce
    an unsolvable or illegal grid (this becomes rarer as recognition
    accuracy improves, but at ~96% on a 25-81 digit puzzle it can still
    happen occasionally, so this must be handled, not assumed away).
    """
    model = model or load_model()

    try:
        cv_result = process_sudoku_image(image_path)
    except (FileNotFoundError, ValueError) as e:
        return SolveResult(None, None, [], 0, error=f"Could not read grid from image: {e}")

    digit_values = {}
    low_confidence = []
    for idx in cv_result.filled_cell_indices():
        digit, confidence = predict_digit(model, cv_result.cells[idx])
        digit_values[idx] = digit
        if confidence < confidence_threshold:
            low_confidence.append({"cell": idx, "predicted_digit": digit, "confidence": round(confidence, 3)})

    grid = cv_result.to_grid(digit_values)

    try:
        game = Sudoku(grid)  # validates: legal grid, no rule violations
    except ValueError as e:
        return SolveResult(grid, None, low_confidence, cv_result.filled_count,
                            error=f"Recognized grid is invalid (likely a misread digit): {e}")

    try:
        solution = game.solve()
    except ValueError as e:
        return SolveResult(grid, None, low_confidence, cv_result.filled_count,
                            error=f"Recognized grid has no solution (likely a misread digit): {e}")

    return SolveResult(grid, solution, low_confidence, cv_result.filled_count)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python solve_photo.py <path_to_sudoku_photo>")
        sys.exit(1)

    result = solve_photo(sys.argv[1])

    if not result.success:
        print(f"FAILED: {result.error}")
        sys.exit(1)

    from sudoku_core import to_string

    print(f"Recognized {result.clues} clues from the photo.")
    if result.low_confidence_cells:
        print(f"\n{len(result.low_confidence_cells)} cell(s) had low confidence, worth double-checking:")
        for c in result.low_confidence_cells:
            print(f"  cell {c['cell']}: predicted {c['predicted_digit']} (confidence {c['confidence']})")

    print("\nRecognized puzzle:")
    print(to_string(result.puzzle))
    print("\nSolved:")
    print(to_string(result.solution))

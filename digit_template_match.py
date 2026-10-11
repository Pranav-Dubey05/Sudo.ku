"""
Sub-step 1: template matching baseline for digit recognition.

Approach: render one clean reference image per digit (1-9), then for any
given cell crop, compare it against all 9 templates and pick whichever
correlates best. No training required - this is a fast, simple baseline
to get digit recognition working end-to-end, and something to measure
the CNN (sub-step 3) against later.

Why this can work reasonably well for printed sudoku puzzles specifically:
printed digits from a fixed font are far more visually consistent than
handwriting, so simple shape correlation goes a long way here - it would
do much worse on handwritten digits.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


TEMPLATE_SIZE = 32  # all templates and cell crops get resized to this before comparing


def build_templates(font=cv2.FONT_HERSHEY_SIMPLEX, thickness=2):
    """
    Renders one reference template per digit 1-9.
    Returns dict: {digit: normalized 32x32 binary image}
    """
    templates = {}
    for digit in range(1, 10):
        canvas = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.uint8)
        text = str(digit)
        (text_w, text_h), baseline = cv2.getTextSize(text, font, 1.0, thickness)
        # scale font so the digit fills most of the canvas
        scale = min((TEMPLATE_SIZE * 0.7) / text_w, (TEMPLATE_SIZE * 0.7) / text_h)
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = (TEMPLATE_SIZE - text_w) // 2
        y = (TEMPLATE_SIZE + text_h) // 2
        cv2.putText(canvas, text, (x, y), font, scale, 255, thickness)
        templates[digit] = canvas
    return templates


def _prepare_cell(cell_img):
    """Resize a cell crop to match template size, ensure it's binary single-channel."""
    if len(cell_img.shape) == 3:
        cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(cell_img, (TEMPLATE_SIZE, TEMPLATE_SIZE))
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return binary


def match_digit(cell_img, templates):
    """
    Compares a cell crop against all 9 digit templates.
    Returns (best_digit, confidence_score) - higher score = better match.
    """
    prepared = _prepare_cell(cell_img)

    best_digit, best_score = None, -1.0
    for digit, template in templates.items():
        result = cv2.matchTemplate(prepared, template, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        if score > best_score:
            best_digit, best_score = digit, score

    return best_digit, best_score


def recognize_all(cells, filled_flags, templates=None):
    """
    Given the 81 cell crops and their filled/empty flags, returns a dict
    {cell_index: recognized_digit} for every filled cell - exactly the
    shape SudokuImageResult.to_grid() expects as its digit_values arg.
    """
    templates = templates or build_templates()
    results = {}
    for i, (cell, filled) in enumerate(zip(cells, filled_flags)):
        if filled:
            digit, _score = match_digit(cell, templates)
            results[i] = digit
    return results


if __name__ == "__main__":
    from pipeline import process_sudoku_image

    if len(sys.argv) < 2:
        print("Usage: python digit_template_match.py <path_to_sudoku_photo>")
        sys.exit(1)

    path = sys.argv[1]
    result = process_sudoku_image(path)
    templates = build_templates()

    digit_values = recognize_all(result.cells, result.filled_flags, templates)

    print("Recognized digits (row-major grid, . = empty):")
    for r in range(9):
        row_str = []
        for c in range(9):
            idx = r * 9 + c
            row_str.append(str(digit_values.get(idx, ".")))
        print(" ".join(row_str))

    grid = result.to_grid(digit_values)
    print(f"\nsudoku_core-compatible grid:\n{grid}")

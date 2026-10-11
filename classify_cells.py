"""
Sub-step 5: decide whether each sliced cell is empty or contains a digit.

Input: the 81 cell crops from slice_cells.slice_cells()
Output: a list of 81 booleans (True = has a digit, False = empty)

Approach: count the proportion of white pixels in the *center* of each
cell (a smaller inner region, not the full crop). We deliberately ignore
a border around the edges, because even after slice_cells' margin trim,
faint remnants of grid lines can survive right at a cell's edge - if we
counted those, some genuinely empty cells would get misclassified as
"filled". A digit's ink lives in the middle of the cell, so restricting
the check to the center is a simple, robust filter for both problems.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_image, preprocess
from grid_detect import find_grid_contour
from warp import warp_grid
from slice_cells import slice_cells, save_contact_sheet


def is_cell_filled(cell_img, center_ratio=0.5, min_blob_area_ratio=0.20):
    """
    Args:
        cell_img: single-channel binary cell crop (white=ink, black=background)
        center_ratio: fraction of the cell (centered) to inspect.
        min_blob_area_ratio: minimum size (as a fraction of the inspected
            region's area) that the LARGEST connected white blob must reach
            to count as a digit.

    Why connected components instead of a raw white-pixel count: paper
    texture and photo noise show up as many small, scattered white specks
    after thresholding. A simple pixel-count check treats scattered speckle
    the same as a solid digit stroke and produces false positives on empty
    cells. A real digit is always one large, single connected shape, so
    checking the size of the biggest connected blob (not the total white
    pixel count) reliably tells noise apart from an actual digit.

    Why 0.20 instead of the original 0.06: on clean rendered / digital
    sudoku images, slicing a grid can leave thin line fragments or edge
    pixels inside the "center" region even with margin trimming. A thin
    line fragment can easily cover 6-10% of the center area - above the
    old threshold, which caused empty cells to be flagged as filled.
    Real digits, even in a small font, fill at least 20-30% of the
    center region, so raising the threshold to 0.20 rejects line
    fragments while still accepting every genuine digit.
    """
    h, w = cell_img.shape[:2]
    margin_h = int(h * (1 - center_ratio) / 2)
    margin_w = int(w * (1 - center_ratio) / 2)
    center = cell_img[margin_h:h - margin_h, margin_w:w - margin_w]

    if center.size == 0:
        return False

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(center, connectivity=8)
    if num_labels <= 1:
        return False  # only the background component exists - nothing here

    # stats[0] is always the background; find the largest non-background blob
    largest_blob_area = max(stats[1:, cv2.CC_STAT_AREA])
    return (largest_blob_area / center.size) > min_blob_area_ratio


def classify_all_cells(cells):
    """Returns a list of 81 booleans, one per cell, True = filled."""
    return [is_cell_filled(cell) for cell in cells]


def save_classification_debug(cells, filled_flags, out_path="filled_debug.png"):
    """
    Same contact-sheet layout as slice_cells, but each filled cell gets a
    green border and each empty cell gets a red border - fast visual check
    of whether the empty/filled calls look right.
    """
    annotated = []
    for cell, filled in zip(cells, filled_flags):
        c = cell.copy()
        if len(c.shape) == 2:
            c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        color = (0, 200, 0) if filled else (0, 0, 200)
        cv2.rectangle(c, (0, 0), (c.shape[1] - 1, c.shape[0] - 1), color, 2)
        annotated.append(c)
    return save_contact_sheet(annotated, out_path=out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_cells.py <path_to_sudoku_photo>")
        sys.exit(1)

    path = sys.argv[1]
    img = load_image(path)
    binary = preprocess(img)
    corners = find_grid_contour(binary)
    warped_binary = warp_grid(binary, corners)
    cells = slice_cells(warped_binary, margin_ratio=0.12)

    filled_flags = classify_all_cells(cells)
    filled_count = sum(filled_flags)
    print(f"Classified {len(cells)} cells: {filled_count} filled, {81 - filled_count} empty")

    out = save_classification_debug(cells, filled_flags)
    print(f"\nDebug visualization saved to: {out}")
    print("Open it and check: cells with a digit should have a GREEN border,")
    print("blank cells should have a RED border. Every digit should be green,")
    print("every true blank should be red.")

    # print as a 9x9 grid of . and X for a quick terminal-only sanity check too
    print("\nGrid layout (X = filled, . = empty):")
    for r in range(9):
        row = filled_flags[r * 9:(r + 1) * 9]
        print(" ".join("X" if f else "." for f in row))
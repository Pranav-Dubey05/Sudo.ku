"""
Sub-step 4: slice the warped grid into 81 individual cell images.

Input: the warped (straightened) 450x450 image from warp.warp_grid()
Output: a flat list of 81 cell images, in row-major order (same order
        sudoku_core.py expects: index 0 = row0/col0, index 1 = row0/col1, ...)

Why 450x450 specifically: 450 / 9 = 50 exactly, so every cell is a clean
50x50 pixel crop with zero rounding error - if we'd picked an output size
that didn't divide evenly by 9, cell boundaries would drift slightly by
the time we reached row/col 8, causing some crops to clip digits.

A small inward margin is trimmed from each cell crop to avoid grid-line
pixels leaking into the corners of the cell - those lines aren't part of
the digit and would only confuse digit recognition in sub-step 3 of the
main roadmap.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_image, preprocess
from grid_detect import find_grid_contour
from warp import warp_grid


def slice_cells(warped_img, margin_ratio=0.08):
    """
    Splits a warped NxN grid image into 81 cell images (row-major order).

    Args:
        warped_img: square image, side length must be divisible by 9.
        margin_ratio: fraction of each cell's size trimmed off all 4 sides,
            to avoid grid-line pixels contaminating the cell crop.

    Returns:
        list of 81 cell images (numpy arrays), index = row * 9 + col.
    """
    h, w = warped_img.shape[:2]
    if h != w:
        raise ValueError(f"Expected a square image, got {w}x{h}")
    if h % 9 != 0:
        raise ValueError(f"Image size {h} isn't evenly divisible by 9 - re-warp with a multiple of 9")

    cell_size = h // 9
    margin = int(cell_size * margin_ratio)

    cells = []
    for r in range(9):
        for c in range(9):
            y0, y1 = r * cell_size, (r + 1) * cell_size
            x0, x1 = c * cell_size, (c + 1) * cell_size
            cell = warped_img[y0 + margin:y1 - margin, x0 + margin:x1 - margin]
            cells.append(cell)

    return cells


def save_contact_sheet(cells, out_path="cells_contact_sheet.png", cols=9):
    """
    Lays all 81 cell crops out in a 9x9 grid with thin separator lines,
    saved as one image - much faster to visually sanity-check than
    opening 81 separate files.
    """
    rows = (len(cells) + cols - 1) // cols
    cell_h, cell_w = cells[0].shape[:2]
    pad = 2

    sheet_h = rows * (cell_h + pad) + pad
    sheet_w = cols * (cell_w + pad) + pad
    is_color = len(cells[0].shape) == 3
    shape = (sheet_h, sheet_w, 3) if is_color else (sheet_h, sheet_w)
    sheet = np.full(shape, 128, dtype=np.uint8)  # gray background as separator

    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        y = pad + r * (cell_h + pad)
        x = pad + c * (cell_w + pad)
        sheet[y:y + cell_h, x:x + cell_w] = cell

    cv2.imwrite(out_path, sheet)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python slice_cells.py <path_to_sudoku_photo>")
        sys.exit(1)

    path = sys.argv[1]
    img = load_image(path)
    binary = preprocess(img)
    corners = find_grid_contour(binary)

    warped_binary = warp_grid(binary, corners)
    cells = slice_cells(warped_binary)

    print(f"Sliced into {len(cells)} cells, each {cells[0].shape[1]}x{cells[0].shape[0]} px")

    out = save_contact_sheet(cells)
    print(f"Contact sheet saved to: {out}")
    print("Open it and check: 81 evenly-spaced cells, each digit centered")
    print("and fully visible with no grid-line clipping into the digit.")

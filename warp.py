"""
Sub-step 3: perspective correction.

Input: the original image + 4 corner points from grid_detect.find_grid_contour()
Output: a clean, straight-on square image of just the grid - ready to be
        sliced into 81 cells in sub-step 4.

Why this matters: a real phone photo is rarely taken dead-on. The grid
usually appears as a skewed quadrilateral (closer edge bigger, farther
edge smaller/rotated). A perspective transform maps those 4 corners onto
the 4 corners of a perfect square, which "un-skews" everything inside it -
including the cell boundaries and digits - as if it had been photographed
straight-on.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_image, preprocess
from grid_detect import find_grid_contour, draw_debug


def warp_grid(img, corners, output_size=450):
    """
    Warps `img` so the quadrilateral defined by `corners` (in
    [top-left, top-right, bottom-right, bottom-left] order) becomes a
    flat output_size x output_size square.

    output_size=450 isn't arbitrary: it's evenly divisible by 9, so each
    cell in sub-step 4 comes out to a clean 50x50 pixels with no rounding.
    """
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, matrix, (output_size, output_size))
    return warped


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python warp.py <path_to_sudoku_photo>")
        sys.exit(1)

    path = sys.argv[1]
    img = load_image(path)
    binary = preprocess(img)

    corners = find_grid_contour(binary)
    print("Detected corners:")
    print(corners)

    warped = warp_grid(img, corners)
    cv2.imwrite("warped_debug.png", warped)
    print("\nWarped (straightened) grid saved to: warped_debug.png")
    print("Open it and check: the grid should now look like a perfect")
    print("straight-on square, with no skew/rotation, and roughly equal")
    print("cell sizes across the whole image.")

    # also warp the binary version - this is what sub-step 4 will actually
    # slice into cells, so worth checking it looks clean too
    warped_binary = warp_grid(binary, corners)
    cv2.imwrite("warped_binary_debug.png", warped_binary)
    print("Warped binary version saved to: warped_binary_debug.png")

"""
Sub-step 2: find the sudoku grid's outer boundary in a preprocessed image.

Input: the binary (black/white) image from preprocess.preprocess()
Output: 4 corner points of the grid, in order [top-left, top-right,
        bottom-right, bottom-left] - ready for perspective correction
        in sub-step 3.

Approach:
- Find all contours in the binary image.
- The grid border is (almost always) the largest connected white shape,
  since it's the biggest continuous structure in a sudoku photo.
- Approximate that contour's shape - if it simplifies down to 4 points,
  we've found the grid's corners directly.
- If the largest contour doesn't simplify to exactly 4 points (common
  with slight photo noise/damage), fall back to using its bounding
  rotated rectangle instead - less precise, but keeps the pipeline going
  rather than failing outright.

Note on the aspect-ratio window below: it used to be 0.8-1.25, which
works for clean printed scans but is a bit too strict for colored /
app-rendered grids whose rounded or slightly clipped corners can
produce a bounding box that isn't quite that square. Loosened to
0.7-1.4, which is still tight enough to reject a full-page outline
(the actual thing this check exists to guard against) while accepting
slightly-less-square colored grids.
"""

import os
import sys

import cv2
import numpy as np

# ensure this script can find preprocess.py sitting next to it, regardless
# of which directory the terminal happens to be in when it's run
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_image, preprocess


def find_grid_contour(binary_img):
    """
    Returns 4 corner points (float32, shape (4,2)) of the detected grid,
    ordered [top-left, top-right, bottom-right, bottom-left].
    Raises ValueError if no plausible grid-sized contour is found.
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found - is the preprocessed image empty/blank?")

    img_area = binary_img.shape[0] * binary_img.shape[1]

    # sort every contour by area, largest first, and take the first one
    # that's both big enough AND roughly square. Just taking the single
    # largest contour breaks on photos with a lot of surrounding page
    # whitespace (URLs, captions, watermarks) - in those cases the biggest
    # connected shape is often the whole page boundary, not the grid. A
    # sudoku grid is always close to square (width ~= height), while a
    # full page usually isn't, so squareness is what actually tells them apart.
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)

    largest = None
    for c in candidates:
        area = cv2.contourArea(c)
        if area < img_area * 0.10:
            break  # remaining contours are too small to be the grid

        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / h if h > 0 else 0
        # loosened from 0.8-1.25: colored / app-rendered grids sometimes
        # have rounded or slightly clipped corners, producing a bounding
        # box that isn't quite as square as a clean printed scan's. The
        # aspect-ratio guard is still the main defence against picking
        # a full-page outline instead of the grid, just slightly more
        # permissive.
        if 0.7 <= aspect_ratio <= 1.4:
            largest = c
            break

    if largest is None:
        raise ValueError(
            "No plausibly square, grid-sized contour found "
            "(check lighting/focus, or that the whole grid is in frame)"
        )

    perimeter = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)

    if len(approx) == 4:
        points = approx.reshape(4, 2).astype("float32")
    else:
        # fallback: use the minimum-area rotated bounding rect instead
        rect = cv2.minAreaRect(largest)
        points = cv2.boxPoints(rect).astype("float32")

    return order_corners(points)


def order_corners(pts):
    """
    Sort 4 arbitrary corner points into [top-left, top-right,
    bottom-right, bottom-left] order, regardless of the order
    contour detection happened to return them in.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]  # bottom-right: largest x+y

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest y-x
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest y-x
    return rect


def draw_debug(original_img, corners, out_path="grid_corners_debug.png"):
    """Draw the detected corners + outline on the original image for visual sanity-checking."""
    debug = original_img.copy()
    pts = corners.astype(int)
    cv2.polylines(debug, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    for (x, y) in pts:
        cv2.circle(debug, (x, y), 8, (0, 0, 255), -1)
    cv2.imwrite(out_path, debug)
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python grid_detect.py <path_to_sudoku_photo>")
        sys.exit(1)

    path = sys.argv[1]
    img = load_image(path)
    binary = preprocess(img)

    corners = find_grid_contour(binary)
    print("Detected grid corners (top-left, top-right, bottom-right, bottom-left):")
    print(corners)

    out = draw_debug(img, corners)
    print(f"\nDebug visualization saved to: {out}")
    print("Open it and check: the green outline should trace the sudoku's outer border,")
    print("with a red dot on each of the 4 true corners.")
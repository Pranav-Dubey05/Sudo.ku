"""
Sub-step 1: load and preprocess a photo of a sudoku puzzle.

Goal: turn a raw, possibly noisy/unevenly-lit phone photo into a clean
black-and-white image where grid lines and digits are crisp - this is
what grid detection (sub-step 2) will operate on.

Pipeline:
    load (with EXIF rotation applied)
      -> resize if too large
      -> per-channel threshold -> OR
      -> morphological open

Why per-channel thresholding instead of a single grayscale pass:
    Sudoku puzzles (especially app-generated / printed-on-color ones)
    frequently use COLORED grid lines - yellow, pink, cyan, green, etc.
    Converting straight to grayscale collapses a saturated colored line
    to a light shade of gray, which an adaptive threshold with c=7 then
    treats as background and silently drops. The result is a broken,
    fragmented grid border that grid_detect can't find.

    The fix: threshold each BGR channel separately and OR the results.
    Any colored line is dark in AT LEAST one channel (yellow is dark in
    B, cyan is dark in R, green is dark in R and B, ...), so OR-ing the
    three binaries recovers every line regardless of hue. Black lines
    are dark in all three channels, so they're still recovered exactly
    as before.

Why resize before thresholding:
    Adaptive threshold's block_size is a pixel count, not a fraction of
    the image. A block_size tuned for a ~900px test image (21px
    neighborhood) is far too small for a 4000px phone camera capture:
    the neighborhood ends up smaller than the grid-line spacing, so
    each line is treated as isolated and the output fragments. Resizing
    the longest edge down to a fixed maximum makes preprocessing behave
    the same regardless of source resolution. INTER_AREA is used
    (rather than INTER_LINEAR/CUBIC) because it's the correct choice
    for downscaling - it averages pixels and avoids aliasing artifacts.

Why EXIF rotation matters:
    Phone cameras usually save portrait photos in landscape orientation
    with an EXIF "rotate me" tag. cv2.imread silently IGNORES that tag,
    so the pipeline sees a sideways grid and detection fails. Loading
    via PIL with exif_transpose() bakes the rotation into the pixels
    before OpenCV ever sees them.
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageOps


# Longest edge is scaled down to this on load. Chosen to be big enough
# that a grid still has plenty of pixels per cell (a 9x9 grid at 1200px
# gives ~130px per cell - way more than the CNN needs) but small enough
# that block_size=21 in adaptive threshold still sees a sensible local
# neighborhood. Test images at ~900px are unchanged by this.
MAX_DIM = 1200


def load_image(path, min_size=100):
    """
    Load an image from disk, applying EXIF orientation and downscaling
    if the source is very large.

    Raises:
        FileNotFoundError: if the path doesn't exist on disk at all.
        ValueError: if the path exists but isn't a readable/valid image,
            or if it's readable but too small to plausibly contain a
            usable sudoku photo.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at: {path}")

    try:
        pil_img = Image.open(path)
        pil_img = ImageOps.exif_transpose(pil_img)   # bake in EXIF rotation
        pil_img = pil_img.convert("RGB")
    except Exception as e:
        raise ValueError(
            f"File exists at {path} but couldn't be read as an image "
            f"(it may be corrupted or not actually an image file): {e}"
        )

    # PIL gives us RGB; OpenCV expects BGR
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        raise ValueError(
            f"Image is only {w}x{h}px, too small to reliably process "
            f"(need at least {min_size}x{min_size}px)"
        )

    # Resize if the longest edge exceeds MAX_DIM
    longest = max(h, w)
    if longest > MAX_DIM:
        scale = MAX_DIM / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def preprocess(img, blur_ksize=7, block_size=21, c=7):
    """
    Convert a raw BGR image into a clean binary (black/white) image ready
    for grid-line detection.

    Color-robust: works on black-on-white printed grids AND on grids
    whose lines are colored (yellow, pink, cyan, green, etc.).

    Args:
        img: raw image as loaded by load_image() (already EXIF-corrected
            and resized to <= MAX_DIM).
        blur_ksize: Gaussian blur kernel size (must be odd).
        block_size: neighborhood size adaptive threshold looks at when
            deciding the local black/white cutoff (must be odd).
        c: constant subtracted from the local mean in adaptive threshold.

    Returns:
        binary: single-channel image, grid lines/digits are white (255)
                on a black (0) background.
    """
    # split into B, G, R - threshold each independently. A colored line
    # that's light in grayscale is still dark relative to its local
    # neighborhood in at least one of these channels.
    channel_binaries = []
    for channel in cv2.split(img):
        blurred = cv2.GaussianBlur(channel, (blur_ksize, blur_ksize), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, c,
        )
        channel_binaries.append(cv2.bitwise_not(thresh))   # ink = white

    combined = cv2.bitwise_or(
        channel_binaries[0],
        cv2.bitwise_or(channel_binaries[1], channel_binaries[2]),
    )

    # gentle opening - thin colored lines hate aggressive morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    return cleaned


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <path_to_sudoku_photo>")
        sys.exit(1)

    path = sys.argv[1]
    img = load_image(path)
    print(f"Loaded image: {path}, shape={img.shape}")

    binary = preprocess(img)
    out_path = "preprocessed_debug.png"
    cv2.imwrite(out_path, binary)
    print(f"Preprocessed image saved to: {out_path}")
    print("Open that file and check: grid lines and digits should be clearly")
    print("white on black, without being washed out or overly speckled.")
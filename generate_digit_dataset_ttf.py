"""
Improved synthetic digit generator using real TTF/OTF font files.

Why this is better than the original OpenCV-Hershey-font approach:
OpenCV's built-in fonts (FONT_HERSHEY_*) are simple vector line drawings -
they don't look much like any real printed font. Real sudoku puzzles
(newspapers, books, apps) are set in actual typefaces like Times New
Roman, Arial, Calibri, etc. Rendering training digits with real font
files closes a real, measurable gap between synthetic training data and
what the model sees in production.

Font files sourced from a public sudoku-recognition reference project
(77 fonts across ~15 typeface families - Times New Roman, Arial, Calibri,
Garamond, Rockwell, Futura, Bodoni, Helvetica, and others).
"""

import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

IMAGE_SIZE = 32
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def load_available_fonts(fonts_dir=FONTS_DIR):
    """
    Finds every .ttf/.otf file under fonts_dir. Skips any that PIL can't
    actually load (some font files are broken/unsupported), rather than
    crashing the whole dataset generation over one bad file.
    """
    paths = glob.glob(os.path.join(fonts_dir, "**", "*.ttf"), recursive=True)
    paths += glob.glob(os.path.join(fonts_dir, "**", "*.otf"), recursive=True)

    working_fonts = []
    for path in paths:
        try:
            ImageFont.truetype(path, 20)  # test-load
            working_fonts.append(path)
        except Exception:
            continue  # skip broken/unsupported font files silently

    return working_fonts


def render_digit_pil(digit, font_path, font_size, angle=0, offset_x=0, offset_y=0):
    """
    Renders one digit using a real TTF/OTF font via PIL, returns a 32x32
    grayscale numpy array (white digit on black background) - same output
    contract as the original OpenCV-based render_digit().
    """
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(font_path, font_size)

    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x = (IMAGE_SIZE - text_w) // 2 - bbox[0] + offset_x
    y = (IMAGE_SIZE - text_h) // 2 - bbox[1] + offset_y
    draw.text((x, y), text, fill=255, font=font)

    img = np.array(canvas)

    if angle != 0:
        matrix = cv2.getRotationMatrix2D((IMAGE_SIZE / 2, IMAGE_SIZE / 2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (IMAGE_SIZE, IMAGE_SIZE))

    return img


def add_noise(img, rng, noise_level=15):
    noise = rng.integers(0, noise_level, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def generate_dataset_ttf(samples_per_digit=150, seed=42, fonts_dir=FONTS_DIR):
    """
    Same output contract as generate_digit_dataset.generate_dataset():
    returns (X, y) - X: float32 (N,32,32) normalized 0-1, y: int (N,) labels 1-9.
    """
    fonts = load_available_fonts(fonts_dir)
    if not fonts:
        raise RuntimeError(f"No usable font files found in {fonts_dir}")
    print(f"Using {len(fonts)} font files")

    rng = np.random.default_rng(seed)
    images, labels = [], []

    for digit in range(1, 10):
        for _ in range(samples_per_digit):
            font_path = fonts[rng.integers(0, len(fonts))]
            font_size = int(rng.integers(20, 28))
            angle = rng.uniform(-10, 10)
            offset_x = int(rng.integers(-2, 3))
            offset_y = int(rng.integers(-2, 3))

            try:
                img = render_digit_pil(digit, font_path, font_size, angle, offset_x, offset_y)
            except Exception:
                continue  # skip if this particular font/digit combo fails to render

            if rng.random() < 0.5:
                img = add_noise(img, rng)

            images.append(img)
            labels.append(digit)

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int64)

    perm = rng.permutation(len(X))
    return X[perm], y[perm]


if __name__ == "__main__":
    print("Generating synthetic printed-digit dataset using real TTF fonts...")
    X, y = generate_dataset_ttf(samples_per_digit=150)
    print(f"Generated {len(X)} samples, shape={X.shape}")

    np.savez_compressed("digit_dataset_ttf.npz", X=X, y=y)
    print("Saved to: digit_dataset_ttf.npz")

    unique, counts = np.unique(y, return_counts=True)
    print(f"Class balance: {dict(zip(unique.tolist(), counts.tolist()))}")

    # visual sanity check, same contact-sheet style as before
    cell = IMAGE_SIZE
    sheet = np.zeros((9 * (cell + 4), 10 * (cell + 4)), dtype=np.uint8)
    for digit in range(1, 10):
        idxs = np.where(y == digit)[0][:10]
        for col, idx in enumerate(idxs):
            img = (X[idx] * 255).astype(np.uint8)
            y0 = (digit - 1) * (cell + 4)
            x0 = col * (cell + 4)
            sheet[y0:y0 + cell, x0:x0 + cell] = img
    cv2.imwrite("dataset_ttf_sample.png", sheet)
    print("Saved visual sample to: dataset_ttf_sample.png")

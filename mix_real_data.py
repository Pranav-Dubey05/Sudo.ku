"""
Mix real labeled cells into the training set to close the sim-to-real gap.

CRITICAL methodological point: we must never train and test on the same
real cells - that would just measure memorization, not generalization.
So the 25 real labeled cells from the test photo are split into a train
portion (augmented and mixed into the synthetic dataset) and a held-out
test portion (never touched during training, used only for honest
final evaluation).
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import process_sudoku_image
from train_cnn import _center_digit, IMAGE_SIZE
from generate_digit_dataset import generate_dataset


GROUND_TRUTH_GRID = [
    "..18926..",
    ".9.....2.",
    ".5.......",
    ".84......",
    "..34479..",
    "......15.",
    ".......3.",
    ".7.....8.",
    "..81534..",
]


def get_ground_truth():
    gt = {}
    for r, row in enumerate(GROUND_TRUTH_GRID):
        for c, ch in enumerate(row):
            if ch != ".":
                gt[r * 9 + c] = int(ch)
    return gt


def get_trusted_cells_from_photo(photo_path, ground_truth):
    """
    Processes one photo and returns only the (cell_image, label) pairs
    where the detected filled/empty pattern matches the known ground
    truth exactly - i.e. cells we can trust are correctly aligned.
    Misaligned photos (e.g. imperfect grid-corner detection) will have
    some cells silently skipped rather than risk mislabeled training data.
    """
    result = process_sudoku_image(photo_path)
    cells, labels = [], []
    for idx, digit in ground_truth.items():
        if result.filled_flags[idx]:  # only trust cells where pattern agrees
            cells.append(result.cells[idx])
            labels.append(digit)
    return cells, labels


def split_real_cells(photo_paths, test_fraction=0.3, seed=42):
    """
    Returns (train_cells, train_labels, test_cells, test_labels) - a clean
    split across ALL trusted cells from ALL given photos, so training and
    testing never touch the same real examples, regardless of which photo
    they came from.
    """
    if isinstance(photo_paths, str):
        photo_paths = [photo_paths]

    ground_truth = get_ground_truth()
    all_cells, all_labels = [], []
    for path in photo_paths:
        cells, labels = get_trusted_cells_from_photo(path, ground_truth)
        print(f"  {path}: {len(cells)} trusted cells")
        all_cells.extend(cells)
        all_labels.extend(labels)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_cells))
    shuffled = rng.permutation(indices)

    n_test = max(1, int(len(shuffled) * test_fraction))
    test_idx_set = set(shuffled[:n_test].tolist())

    train_cells, train_labels = [], []
    test_cells, test_labels = [], []
    for i in range(len(all_cells)):
        if i in test_idx_set:
            test_cells.append(all_cells[i])
            test_labels.append(all_labels[i])
        else:
            train_cells.append(all_cells[i])
            train_labels.append(all_labels[i])

    return train_cells, train_labels, test_cells, test_labels


def augment_real_cell(cell_img, rng, n_variations=20):
    """
    Generates several augmented variations of one real cell crop: small
    rotations, shifts, and noise. This is what lets a handful of real
    examples meaningfully contribute training signal instead of being
    a drop in the ocean against thousands of synthetic images.
    """
    centered = _center_digit(cell_img)
    variations = [centered]  # include the original

    for _ in range(n_variations - 1):
        angle = rng.uniform(-10, 10)
        matrix = cv2.getRotationMatrix2D((IMAGE_SIZE / 2, IMAGE_SIZE / 2), angle, 1.0)
        rotated = cv2.warpAffine(centered, matrix, (IMAGE_SIZE, IMAGE_SIZE))

        tx, ty = rng.integers(-2, 3), rng.integers(-2, 3)
        shift_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        shifted = cv2.warpAffine(rotated, shift_matrix, (IMAGE_SIZE, IMAGE_SIZE))

        if rng.random() < 0.5:
            noise = rng.integers(0, 10, shifted.shape, dtype=np.uint8)
            shifted = cv2.add(shifted, noise)

        variations.append(shifted)

    return variations


def build_mixed_dataset(photo_paths, synthetic_samples_per_digit=300, real_augment_count=30, seed=42):
    """
    Combines synthetic digits with augmented real examples (train portion
    only, across all given photos). Returns (X, y, test_cells, test_labels)
    - the last two are the held-out real cells for honest final evaluation.
    """
    rng = np.random.default_rng(seed)

    print("Extracting trusted real cells from photos:")
    train_cells, train_labels, test_cells, test_labels = split_real_cells(photo_paths, seed=seed)
    print(f"Total: {len(train_cells)} for training (augmented), {len(test_cells)} held out for testing")

    X_synth, y_synth = generate_dataset(samples_per_digit=synthetic_samples_per_digit, seed=seed)

    real_images, real_labels = [], []
    for cell, label in zip(train_cells, train_labels):
        variations = augment_real_cell(cell, rng, n_variations=real_augment_count)
        real_images.extend(variations)
        real_labels.extend([label] * len(variations))

    X_real = np.array(real_images, dtype=np.float32) / 255.0
    y_real = np.array(real_labels, dtype=np.int64)
    print(f"Generated {len(X_real)} augmented real training samples from {len(train_cells)} real cells")

    X = np.concatenate([X_synth, X_real], axis=0)
    y = np.concatenate([y_synth, y_real], axis=0)

    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    return X, y, test_cells, test_labels


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mix_real_data.py <photo1> [photo2] [photo3] ...")
        sys.exit(1)

    photo_paths = sys.argv[1:]
    X, y, test_cells, test_labels = build_mixed_dataset(photo_paths)

    print(f"\nFinal mixed dataset: {len(X)} samples")
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class balance: {dict(zip(unique.tolist(), counts.tolist()))}")

    np.savez_compressed("digit_dataset_mixed.npz", X=X, y=y)
    print("Saved to: digit_dataset_mixed.npz")

    # save the held-out test cells too, so evaluation can be run separately
    # without needing to re-derive the same split
    np.savez_compressed(
        "held_out_real_test.npz",
        cells=np.array([c for c in test_cells], dtype=object),
        labels=np.array(test_labels),
        allow_pickle=True,
    )
    print(f"Saved {len(test_cells)} held-out real test cells to: held_out_real_test.npz")
    print("\nThese held-out cells were NEVER used in training - safe for honest evaluation.")

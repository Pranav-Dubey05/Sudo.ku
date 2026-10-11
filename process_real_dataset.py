"""
Process the real labeled sudoku photo dataset (v1_training / v1_test,
360 real camera photos with exact ground-truth grids) through our own
CV pipeline, building a genuinely large real training set and an
honestly held-out real test set.

Methodology:
- v1_training images -> training cells only
- v1_test images -> test cells only (never touched during training)
This mirrors the dataset's own original train/test split, so there's
no risk of leakage between the two.

For each image, our pipeline's own filled/empty detection is compared
against the known ground truth pattern. Only cells where they agree are
trusted - this protects against grid-detection misalignment (which we
saw happen on a real photo earlier) silently mislabeling training data.
Images where too few cells align are skipped entirely, since that
usually means grid/corner detection failed on that particular photo.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import process_sudoku_image


def parse_dat_file(path):
    """Parses one .dat ground truth file into an 81-int flat grid."""
    with open(path) as f:
        lines = f.readlines()
    grid_lines = lines[2:11]  # first 2 lines are camera metadata, next 9 are the grid
    grid = []
    for line in grid_lines:
        grid.extend(int(v) for v in line.split())
    return grid


def process_image_folder(folder, min_match_ratio=0.85, verbose=True):
    """
    Processes every .jpg in folder that has a matching .dat file.
    Returns list of (cell_image, label) for every trusted filled cell
    across all successfully-processed images, plus stats on how many
    images were skipped due to poor alignment.
    """
    jpg_files = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))

    all_cells, all_labels = [], []
    processed, skipped_align, skipped_error = 0, 0, 0

    for fname in jpg_files:
        base = fname[:-4]
        dat_path = os.path.join(folder, base + ".dat")
        jpg_path = os.path.join(folder, fname)
        if not os.path.exists(dat_path):
            continue

        try:
            ground_truth_grid = parse_dat_file(dat_path)
            result = process_sudoku_image(jpg_path)
        except Exception:
            skipped_error += 1
            continue

        expected_filled = [v != 0 for v in ground_truth_grid]
        detected_filled = result.filled_flags
        matches = sum(1 for e, d in zip(expected_filled, detected_filled) if e == d)
        match_ratio = matches / 81

        if match_ratio < min_match_ratio:
            skipped_align += 1
            continue

        for idx in range(81):
            if expected_filled[idx] and detected_filled[idx]:
                all_cells.append(result.cells[idx])
                all_labels.append(ground_truth_grid[idx])

        processed += 1

    if verbose:
        print(f"  {folder}: {processed} images used, {skipped_align} skipped (alignment), "
              f"{skipped_error} skipped (errors) -> {len(all_cells)} trusted cells")

    return all_cells, all_labels


if __name__ == "__main__":
    train_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_dataset_train", "v1_training")
    test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_dataset_test", "v1_test")

    print("Processing training images...")
    train_cells, train_labels = process_image_folder(train_folder)

    print("Processing test images (held out, never used in training)...")
    test_cells, test_labels = process_image_folder(test_folder)

    print(f"\nTotal: {len(train_cells)} training cells, {len(test_cells)} held-out test cells")

    np.savez_compressed(
        "large_real_train.npz",
        cells=np.array(train_cells, dtype=object),
        labels=np.array(train_labels),
        allow_pickle=True,
    )
    np.savez_compressed(
        "large_real_test.npz",
        cells=np.array(test_cells, dtype=object),
        labels=np.array(test_labels),
        allow_pickle=True,
    )
    print("Saved to large_real_train.npz and large_real_test.npz")

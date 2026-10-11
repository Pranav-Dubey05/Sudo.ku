"""
Tunes classify_cells' is_cell_filled() parameters against a large sample
of real, labeled cells - prioritizing a LOW FALSE-NEGATIVE rate specifically,
since a missed digit (false negative) is the dangerous failure mode: it
looks identical to a genuinely blank cell, so the solver has no way to
detect it's wrong and can silently produce an incorrect "solved" puzzle.
A false positive (phantom digit on an empty cell) is comparatively safe -
digit recognition will likely produce inconsistent results on pure
background noise, which validate_grid() or the solver can catch.

This script produced the tuned defaults currently in classify_cells.py
(center_ratio=0.5, min_blob_area_ratio=0.06), found by grid search across
~8,700 labeled real cells from the v1+v2 training photos.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_image, preprocess
from grid_detect import find_grid_contour
from warp import warp_grid
from slice_cells import slice_cells


def parse_dat_file(path):
    with open(path) as f:
        lines = f.readlines()
    grid = []
    for line in lines[2:11]:
        grid.extend(int(v) for v in line.split())
    return grid


def collect_labeled_cells(folders, sample_per_folder=60):
    """Gathers (cell_image, is_actually_filled) pairs from real labeled photos."""
    all_cells, all_truth = [], []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        jpgs = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))[:sample_per_folder]
        for fname in jpgs:
            base = fname[:-4]
            dat_path = os.path.join(folder, base + ".dat")
            jpg_path = os.path.join(folder, fname)
            if not os.path.exists(dat_path):
                continue
            try:
                truth_grid = parse_dat_file(dat_path)
                img = load_image(jpg_path)
                binary = preprocess(img)
                corners = find_grid_contour(binary)
                warped_binary = warp_grid(binary, corners)
                cells = slice_cells(warped_binary, margin_ratio=0.12)
            except Exception:
                continue
            for cell, truth_val in zip(cells, truth_grid):
                all_cells.append(cell)
                all_truth.append(truth_val != 0)
    return all_cells, np.array(all_truth)


def is_filled_with_params(cell, center_ratio, min_blob_ratio):
    h, w = cell.shape[:2]
    mh, mw = int(h * (1 - center_ratio) / 2), int(w * (1 - center_ratio) / 2)
    center = cell[mh:h - mh, mw:w - mw]
    if center.size == 0:
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(center, connectivity=8)
    largest = max(stats[1:, cv2.CC_STAT_AREA]) if n > 1 else 0
    return (largest / center.size) > min_blob_ratio


def grid_search(cells, truth, fn_penalty=2.0):
    """
    Searches center_ratio x min_blob_ratio combinations. Score weights
    false-negative rate more heavily than raw accuracy, since missed
    digits are the dangerous failure mode (see module docstring).
    """
    best = None
    results = []
    for center_ratio in [0.4, 0.5, 0.6, 0.7]:
        for min_blob_ratio in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
            preds = np.array([is_filled_with_params(c, center_ratio, min_blob_ratio) for c in cells])
            accuracy = np.mean(preds == truth)
            false_negatives = np.sum((truth) & (~preds))
            false_positives = np.sum((~truth) & (preds))
            fn_rate = false_negatives / np.sum(truth)
            score = accuracy - fn_rate * fn_penalty

            results.append((center_ratio, min_blob_ratio, accuracy, fn_rate, false_positives))
            if best is None or score > best[0]:
                best = (score, center_ratio, min_blob_ratio, accuracy, fn_rate)
    return best, results


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    folders = [
        os.path.join(base, "real_dataset_train", "v1_training"),
        os.path.join(base, "real_dataset_v2_train", "v2_train"),
    ]

    print("Collecting labeled cells from real training photos...")
    cells, truth = collect_labeled_cells(folders)
    print(f"Collected {len(cells)} labeled cells ({np.sum(truth)} filled, {np.sum(~truth)} empty)")

    print("\nGrid search (prioritizing low false-negative rate)...")
    best, results = grid_search(cells, truth)

    print("\nAll results (center_ratio, min_blob_ratio, accuracy, FN_rate, FP_count):")
    for r in results:
        print(f"  {r}")

    _, center_ratio, min_blob_ratio, accuracy, fn_rate = best
    print(f"\nBEST: center_ratio={center_ratio}, min_blob_area_ratio={min_blob_ratio}")
    print(f"  accuracy={accuracy:.3f}, false_negative_rate={fn_rate:.3f}")
    print("\nUpdate these as the defaults in classify_cells.py's is_cell_filled().")

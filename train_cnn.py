"""
Sub-step 3: build and train a small CNN for digit recognition (PyTorch).

Switched from TensorFlow to PyTorch because TensorFlow doesn't yet publish
wheels for Python 3.14 (too new at time of writing), while PyTorch does.

Architecture: kept deliberately small - this is a 32x32 grayscale,
9-class classification problem (digits 1-9; blank cells are already
handled separately by classify_cells.py, so 0 isn't a class here).

Trains on generate_digit_dataset.py's synthetic output, with a held-out
validation split to catch overfitting honestly rather than just trusting
training accuracy.
"""

import os
import sys

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

IMAGE_SIZE = 32
NUM_CLASSES = 9  # digits 1-9


class DigitCNN(nn.Module):
    """Two conv blocks (conv + pool) followed by a dense classifier, with dropout."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))   # 16x16 -> 8x8
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # raw logits - loss function applies softmax internally


def load_dataset(path="digit_dataset.npz"):
    """
    Loads the synthetic dataset and prepares PyTorch tensors:
    - adds the channel dimension PyTorch conv layers expect (N, C, H, W)
    - shifts labels from 1-9 down to 0-8 (PyTorch class indices are 0-based)
    """
    data = np.load(path)
    X, y = data["X"], data["y"]
    X = X.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)  # (N, C, H, W)
    y = y - 1  # 1-9 -> 0-8

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    return X_tensor, y_tensor


def train(dataset_path="digit_dataset.npz", epochs=15, val_split=0.15, batch_size=32, lr=0.001, seed=42):
    torch.manual_seed(seed)  # reproducibility: same weights init + same train/val split every run

    X, y = load_dataset(dataset_path)
    print(f"Loaded {len(X)} samples for training")

    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DigitCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_correct += (out.argmax(1) == yb).sum().item()
            train_total += xb.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch}/{epochs} - loss: {train_loss/train_total:.4f} "
              f"- train_acc: {train_acc:.3f} - val_acc: {val_acc:.3f}")

    print(f"\nFinal training accuracy: {train_acc:.3f}")
    print(f"Final validation accuracy: {val_acc:.3f}")
    if train_acc - val_acc > 0.10:
        print("NOTE: training accuracy is notably higher than validation accuracy -")
        print("this suggests some overfitting, worth keeping in mind.")

    return model


def _center_digit(cell_img, canvas_size=IMAGE_SIZE, padding_ratio=0.15):
    """
    Finds the actual digit's ink within a cell crop, then re-centers it on
    a clean square canvas before resizing - matching how the synthetic
    training digits are already centered.

    Why this matters: without it, a digit that happens to sit slightly
    off-center or occupies only a small portion of the raw cell crop gets
    squashed/stretched inconsistently compared to the synthetic training
    images (which are always centered by construction). That mismatch is
    exactly the kind of thing that produces confidently-wrong predictions -
    the model was never trained on off-center digits, so it doesn't know
    it should be uncertain.

    Why we isolate the largest connected component (not just findNonZero):
    a single stray white pixel anywhere in the crop - a grid-line remnant
    at the edge, a JPEG speckle, ink bleed from an adjacent cell - would
    otherwise make findNonZero + boundingRect span the full image. The
    real digit then gets scaled down to a tiny speck on a mostly-black
    canvas, which the CNN has never seen and reads as garbage. Taking
    the largest connected white blob's bounding box instead ignores
    every such stray pixel and gives us the true digit extent.
    """
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img

    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        # nothing but background - just fall back to a plain resize
        return cv2.resize(gray, (canvas_size, canvas_size))

    # stats[0] is always the background component; find the largest foreground one
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]

    if w <= 0 or h <= 0:
        # degenerate component - fall back
        return cv2.resize(gray, (canvas_size, canvas_size))

    # mask out everything except the largest component, then crop it
    mask = (labels == largest_label).astype(np.uint8) * 255
    digit_crop = cv2.bitwise_and(gray, gray, mask=mask)[y:y + h, x:x + w]

    # scale the digit to fit within the canvas, leaving a small padding margin
    target_size = int(canvas_size * (1 - padding_ratio))
    scale = min(target_size / w, target_size / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized_digit = cv2.resize(digit_crop, (new_w, new_h))

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    x_off = (canvas_size - new_w) // 2
    y_off = (canvas_size - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized_digit
    return canvas


def predict_digit(model, cell_img):
    """
    Given a trained model and a single cell crop (any size, single or
    3-channel), returns (predicted_digit 1-9, confidence 0-1).
    """
    import cv2
    centered = _center_digit(cell_img)
    normalized = centered.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).float().reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        predicted_class = int(torch.argmax(probs).item())
        confidence = float(probs[predicted_class].item())

    return predicted_class + 1, confidence  # shift back 0-8 -> 1-9


if __name__ == "__main__":
    model = train()

    torch.save(model.state_dict(), "digit_cnn.pt")
    print("\nModel saved to: digit_cnn.pt")
    print("Next: test this model against real cell crops from an actual photo")
    print("(sub-step 4), not just the synthetic validation split above.")
"""
API test suite for the sudoku scanner backend.

Uses FastAPI's TestClient (in-process, no real server/network needed) -
fast and reliable, and this is exactly how the bugs in main.py were
actually found and fixed during development (a numpy-serialization bug
in /solve-from-image, and a misleading error-message bug), so these
tests are a direct record of that debugging work, not just boilerplate.

Run from the backend/ folder with:
    pytest tests/ -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


# A known, hand-verified, solvable puzzle - used across several tests
KNOWN_PUZZLE = [
    5, 3, 0, 0, 7, 0, 0, 0, 0,
    6, 0, 0, 1, 9, 5, 0, 0, 0,
    0, 9, 8, 0, 0, 0, 0, 6, 0,
    8, 0, 0, 0, 6, 0, 0, 0, 3,
    4, 0, 0, 8, 0, 3, 0, 0, 1,
    7, 0, 0, 0, 2, 0, 0, 0, 6,
    0, 6, 0, 0, 0, 0, 2, 8, 0,
    0, 0, 0, 4, 1, 9, 0, 0, 5,
    0, 0, 0, 0, 8, 0, 0, 7, 9,
]
KNOWN_SOLUTION = [
    5, 3, 4, 6, 7, 8, 9, 1, 2,
    6, 7, 2, 1, 9, 5, 3, 4, 8,
    1, 9, 8, 3, 4, 2, 5, 6, 7,
    8, 5, 9, 7, 6, 1, 4, 2, 3,
    4, 2, 6, 8, 5, 3, 7, 9, 1,
    7, 1, 3, 9, 2, 4, 8, 5, 6,
    9, 6, 1, 5, 3, 7, 2, 8, 4,
    2, 8, 7, 4, 1, 9, 6, 3, 5,
    3, 4, 5, 2, 8, 6, 1, 7, 9,
]


def make_image_bytes(width=300, height=300, all_white=True):
    """Builds a simple in-memory PNG for testing, no disk I/O needed."""
    img = np.ones((height, width, 3), dtype=np.uint8) * (255 if all_white else 0)
    success, buf = cv2.imencode(".png", img)
    assert success
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------------------------

class TestBasicEndpoints:
    def test_root_returns_ok(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_check(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "healthy"}


# ---------------------------------------------------------------------------
# /solve
# ---------------------------------------------------------------------------

class TestSolveEndpoint:
    def test_solves_known_puzzle_correctly(self):
        r = client.post("/solve", json={"grid": KNOWN_PUZZLE})
        assert r.status_code == 200
        body = r.json()
        assert body["solution"] == KNOWN_SOLUTION
        assert body["puzzle"] == KNOWN_PUZZLE
        assert body["difficulty"] in ("easy", "medium", "hard", "expert")

    def test_wrong_length_returns_422(self):
        r = client.post("/solve", json={"grid": [1, 2, 3]})
        assert r.status_code == 422

    def test_empty_grid_returns_422(self):
        r = client.post("/solve", json={"grid": []})
        assert r.status_code == 422

    def test_out_of_range_value_returns_422(self):
        r = client.post("/solve", json={"grid": [15] * 81})
        assert r.status_code == 422

    def test_negative_value_returns_422(self):
        r = client.post("/solve", json={"grid": [-1] * 81})
        assert r.status_code == 422

    def test_conflicting_givens_returns_422(self):
        broken = list(KNOWN_PUZZLE)
        broken[1] = 5  # duplicate 5 in row 0 alongside broken[0]
        r = client.post("/solve", json={"grid": broken})
        assert r.status_code == 422

    def test_float_values_rejected(self):
        r = client.post("/solve", json={"grid": [5.5] * 81})
        assert r.status_code == 422

    def test_string_values_rejected(self):
        r = client.post("/solve", json={"grid": ["a"] * 81})
        assert r.status_code == 422

    def test_missing_grid_field_rejected(self):
        r = client.post("/solve", json={})
        assert r.status_code == 422

    def test_already_solved_grid_returns_itself(self):
        r = client.post("/solve", json={"grid": KNOWN_SOLUTION})
        assert r.status_code == 200
        assert r.json()["solution"] == KNOWN_SOLUTION


# ---------------------------------------------------------------------------
# /generate
# ---------------------------------------------------------------------------

class TestGenerateEndpoint:
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "expert"])
    def test_generates_valid_puzzle_for_each_difficulty(self, difficulty):
        r = client.get(f"/generate?difficulty={difficulty}")
        assert r.status_code == 200
        body = r.json()
        assert len(body["puzzle"]) == 81
        assert len(body["solution"]) == 81
        assert body["requested_difficulty"] == difficulty
        # every given clue must match the solution at that position
        assert all(
            p == s for p, s in zip(body["puzzle"], body["solution"]) if p != 0
        )

    def test_default_difficulty_is_medium(self):
        r = client.get("/generate")
        assert r.status_code == 200
        assert r.json()["requested_difficulty"] == "medium"

    def test_invalid_difficulty_returns_400(self):
        r = client.get("/generate?difficulty=impossible")
        assert r.status_code == 400

    def test_wrong_case_difficulty_returns_400(self):
        r = client.get("/generate?difficulty=Easy")
        assert r.status_code == 400

    def test_each_generated_puzzle_is_different(self):
        r1 = client.get("/generate?difficulty=medium")
        r2 = client.get("/generate?difficulty=medium")
        assert r1.json()["puzzle"] != r2.json()["puzzle"]


# ---------------------------------------------------------------------------
# /solve-from-image
# ---------------------------------------------------------------------------

class TestSolveFromImageEndpoint:
    def test_non_image_content_type_returns_400(self):
        r = client.post(
            "/solve-from-image",
            files={"file": ("notanimage.txt", b"hello world", "text/plain")},
        )
        assert r.status_code == 400

    def test_missing_file_returns_422(self):
        r = client.post("/solve-from-image")
        assert r.status_code == 422

    def test_corrupt_file_returns_422_with_clear_message(self):
        r = client.post(
            "/solve-from-image",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert r.status_code == 422
        assert "couldn't be read" in r.json()["detail"]

    def test_too_small_image_returns_422_with_clear_message(self):
        tiny_bytes = make_image_bytes(width=20, height=20)
        r = client.post(
            "/solve-from-image",
            files={"file": ("tiny.png", tiny_bytes, "image/png")},
        )
        assert r.status_code == 422
        assert "too small" in r.json()["detail"]

    def test_valid_image_with_no_grid_returns_422_with_clear_message(self):
        blank_bytes = make_image_bytes(width=300, height=300, all_white=True)
        r = client.post(
            "/solve-from-image",
            files={"file": ("blank.png", blank_bytes, "image/png")},
        )
        assert r.status_code == 422
        # message should describe the actual problem (no grid found),
        # not a generic/misleading catch-all
        assert "contour" in r.json()["detail"].lower() or "grid" in r.json()["detail"].lower()

    def test_error_messages_are_distinguishable(self):
        """
        Regression test for a real bug: corrupt file, too-small image,
        and no-grid-found used to all return the same generic message.
        They must be distinguishable so the app can show a helpful,
        accurate message to the user.
        """
        r1 = client.post("/solve-from-image", files={"file": ("empty.jpg", b"", "image/jpeg")})
        r2 = client.post(
            "/solve-from-image",
            files={"file": ("tiny.png", make_image_bytes(20, 20), "image/png")},
        )
        assert r1.json()["detail"] != r2.json()["detail"]
        
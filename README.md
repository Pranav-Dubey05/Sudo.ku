# Sudoku Scanner Project — File Guide

## Core solver (root folder)
- `sudoku_core.py` — Sudoku class, solver, generator, difficulty rating, validation, JSON I/O
- `tests/test_sudoku_core.py` — 31 tests

## Computer vision + digit recognition pipeline (cv/ folder)

The full pipeline, in order:
1. `preprocess.py` — image cleanup (grayscale, adaptive threshold, denoise)
2. `grid_detect.py` — finds the sudoku's 4 corners in a photo
3. `warp.py` — straightens the grid into a flat square
4. `slice_cells.py` — cuts the grid into 81 individual cell images
5. `classify_cells.py` — decides which cells are empty vs filled
6. `train_cnn.py` — CNN architecture, training, and single-cell prediction
7. `pipeline.py` — **the main entry point.** Ties everything together:
   `process_sudoku_image(path)` → `.to_grid()` gives you a real, solved-ready
   81-int grid with digit recognition already applied automatically.

### One-line usage
```python
from pipeline import process_sudoku_image
from sudoku_core import Sudoku

result = process_sudoku_image("my_sudoku_photo.jpg")
grid = result.to_grid()          # real recognized digits, not placeholders
solution = Sudoku(grid).solve()  # raises ValueError if a digit was misread
```

## Digit recognition support files
- `digit_template_match.py` — simple baseline (~16% accuracy, kept for the report's comparison table)
- `generate_digit_dataset.py` — synthetic digits using OpenCV fonts
- `generate_digit_dataset_ttf.py` — synthetic digits using real TTF fonts (`fonts/` folder)
- `mix_real_data.py` — combines small numbers of real labeled photos into training data
- `process_real_dataset.py` — processes the large real academic sudoku dataset
- `tune_classify_cells.py` — the methodology used to tune empty/filled detection
  against ~8,700 real labeled cells, prioritizing a low false-negative rate

## ⭐ THE MODEL THAT MATTERS
**`digit_cnn_mega.pt`** — the final model `pipeline.py` actually loads and uses.

- Trained on ~8,855 samples: real academic dataset (v1+v2, ~6,200 real cells) +
  our own captured photos + synthetic data
- Tested on 1,680 genuinely held-out real cells: **96.4% cell-level accuracy**
- Full photo→solved-puzzle integration tests: see `cv/tests/test_end_to_end.py`

## backend/ — FastAPI server (for the Android app)

`backend/main.py` — the API the Android app will talk to.

**Endpoints so far:**
- `GET /` — basic status check
- `GET /health` — lightweight ping for the app to confirm the server is reachable
- `POST /solve-from-image` — upload a photo, get back the recognized grid +
  solved puzzle. Returns proper HTTP status codes: `400` for a bad/unreadable
  image, `422` if no grid was found or the recognized puzzle can't be solved
  (usually a misread digit), `200` with full JSON on success.

**Run it:**
```
cd backend
pip install fastapi uvicorn python-multipart
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Then visit `http://localhost:8000/docs` for an interactive test page —
you can upload a photo directly in the browser to try `/solve-from-image`.

`--host 0.0.0.0` (not `127.0.0.1`) matters: it's what makes the server
reachable from your phone over the same WiFi network, not just from your
own laptop — needed once the Android app is built.

**Still to come:** `/solve` (raw grid JSON in, no image), `/generate`
(new puzzle by difficulty), full API test suite.

## Tests
- `tests/test_sudoku_core.py` — 31 tests (solver, generator, validation, I/O)
- `cv/tests/test_cv_pipeline.py` — 27 tests (every CV sub-step + digit recognition wiring)
- `cv/tests/test_end_to_end.py` — 6 integration tests: real photo → solved puzzle

Run everything: `pytest tests/ cv/tests/ -v` → **66 passed** (6 of those need
the real labeled photo dataset, not included in this zip — see below; they
skip gracefully rather than fail when it's absent)

## fonts/
77 real TTF/OTF font files (Times New Roman, Arial, Calibri, etc.) used by
`generate_digit_dataset_ttf.py`.

## Full accuracy journey (for the report)
| Stage | Metric | Result |
|---|---|---|
| Template matching | cell accuracy | 16.0% |
| CNN, synthetic only | cell accuracy | 48.0% |
| CNN, small real-data mix | cell accuracy | 57.1% |
| CNN, large combined real+synthetic dataset | cell accuracy | **96.4%** (1,680 held-out cells) |
| Empty/filled classification, before tuning | silent-wrongness rate | ~33% |
| Empty/filled classification, after tuning against 8,700 real cells | silent-wrongness rate | **2.5%** |
| Full pipeline (photo→solved puzzle) | success rate | 48% |
| Full pipeline | safe-failure rate (caught, not silent) | 50% |

**Key finding for the report:** cell-level digit accuracy (96%) and full-puzzle
success rate (48%) are different numbers for a real reason — a puzzle with
~25 clues needs *every single* recognized digit correct to solve properly, so
even a small per-cell error rate compounds across many cells.

## Known limitations (honest, for the report)
- Digit recognition generalizes better within a training-similar photo
  distribution than to genuinely novel camera/print sources
- Grid detection assumes the sudoku roughly fills the frame; heavy
  surrounding page content (captions, watermarks) can reduce corner-detection
  accuracy
- A missed digit (empty/filled false negative) is structurally undetectable
  by `validate_grid()`, since a missed digit is indistinguishable from a
  genuinely blank cell from the solver's point of view

## Not included in this zip
The raw real-photo dataset folders (`real_dataset_train/`, `real_dataset_test/`,
`real_dataset_v2_train/`, `real_dataset_v2_test/`) are excluded to keep zip
size reasonable — they were only needed to *build* the training data, not to
run the trained model. That work is already baked into `digit_cnn_mega.pt`.

## Project roadmap status
1. ✅ Solver/generator core
2. ✅ CV pipeline (grid detection)
3. ✅ Digit recognition (wired in, tested end-to-end)
4. 🔨 FastAPI backend — in progress (`/solve-from-image` done, more endpoints coming)
5. ⬜ Android app (Kotlin + CameraX)
6. ⬜ Full integration + written report

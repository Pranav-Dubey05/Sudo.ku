"""
FastAPI backend. Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
--host 0.0.0.0 makes it reachable from phone over WiFi, not just localhost.
"""

import os, sys, tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cv"))
from pydantic import BaseModel, field_validator
from sudoku_core import Sudoku
from pipeline import process_sudoku_image

app = FastAPI(title="Sudoku Scanner API")


class SolveRequest(BaseModel):
    grid: list[int]

    @field_validator("grid")
    @classmethod
    def check_length(cls, v):
        if len(v) != 81:
            raise ValueError(f"grid must have 81 cells, got {len(v)}")
        return v


@app.get("/")
def root():
    return {"status": "ok", "message": "Sudoku Scanner API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/solve")
def solve_grid(request: SolveRequest):
    """Raw 81-int grid in (0=blank), solution + difficulty out."""
    try:
        game = Sudoku(request.grid)
        solution = game.solve()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "puzzle": list(request.grid),
        "solution": list(solution),
        "difficulty": game.rate(),
    }


VALID_DIFFICULTIES = {"easy", "medium", "hard", "expert"}


@app.get("/generate")
def generate_puzzle(difficulty: str = "medium"):
    if difficulty not in VALID_DIFFICULTIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid difficulty. Must be one of: {sorted(VALID_DIFFICULTIES)}",
        )
    game = Sudoku.generate(difficulty)
    return {
        "puzzle": list(game.puzzle),
        "solution": list(game.solution),
        "requested_difficulty": difficulty,
        "measured_difficulty": game.difficulty,
        "clues": game.clues,
    }


@app.post("/solve-from-image")
async def solve_from_image(file: UploadFile = File(...)):
    """Upload photo → recognized grid + solution.
    400=bad image, 422=no grid found or unsolvable, 200=success."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Not an image")
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = process_sudoku_image(tmp_path)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Could not read image")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)

    recognized_grid = result.to_grid()
    try:
        game = Sudoku(recognized_grid)
        solution = game.solve()
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Recognized puzzle could not be solved (likely misread digit): {e}",
        )
    return {
        "recognized_grid": list(recognized_grid),
        "solution": list(solution),
        "cells_filled": int(result.filled_count),
        "difficulty": Sudoku(recognized_grid, solution).rate(),
    }


@app.post("/recognize-from-image")
async def recognize_from_image(file: UploadFile = File(...)):
    """Upload photo → recognized grid + per-cell confidence.
    Does NOT solve. Used by the Android review screen so the user can
    correct low-confidence cells before posting to /solve.
    400=bad image, 422=no grid found, 200=success."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Not an image")
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = process_sudoku_image(tmp_path)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Could not read image")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)

    from pipeline import _get_digit_model
    from train_cnn import predict_digit

    model = _get_digit_model()
    digit_values = {}
    digit_confidences = {}
    for idx in result.filled_cell_indices():
        digit, conf = predict_digit(model, result.cells[idx])
        digit_values[idx] = digit
        digit_confidences[str(idx)] = round(conf, 3)

    recognized_grid = result.to_grid(digit_values)
    return {
        "recognized_grid": recognized_grid,
        "digit_confidences": digit_confidences,
        "cells_filled": int(result.filled_count),
        "low_confidence_threshold": 0.7,
    }
"""
Test suite for sudoku_core.py

Run from the project root with:
    pytest tests/

Or with more detail:
    pytest tests/ -v
"""

import os
import sys

# allow running pytest from the project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sudoku_core import (
    Sudoku,
    solve,
    has_unique_solution,
    generate_full_grid,
    generate_puzzle,
    rate_difficulty,
    validate_grid,
    save_puzzle,
    load_puzzle,
)

# A known-good, hand-verified solved grid, used across several tests
KNOWN_SOLVED = [
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

# The same grid with one clue removed - still has a unique solution
KNOWN_PUZZLE_ONE_BLANK = list(KNOWN_SOLVED)
KNOWN_PUZZLE_ONE_BLANK[0] = 0


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class TestSolver:
    def test_solves_known_puzzle_correctly(self):
        result = solve(KNOWN_PUZZLE_ONE_BLANK, count_solutions_limit=1)
        assert len(result) == 1
        assert result[0] == KNOWN_SOLVED

    def test_already_solved_grid_returns_itself(self):
        result = solve(KNOWN_SOLVED, count_solutions_limit=1)
        assert result[0] == KNOWN_SOLVED

    def test_unsolvable_grid_returns_empty(self):
        broken = list(KNOWN_SOLVED)
        broken[1] = broken[0]  # create an immediate row conflict, unsolvable
        result = solve(broken, count_solutions_limit=1)
        assert result == []

    def test_empty_grid_is_solvable(self):
        empty = [0] * 81
        result = solve(empty, count_solutions_limit=1)
        assert len(result) == 1
        validate_grid(result[0], allow_blanks=False)  # must be a legal full grid


class TestUniqueness:
    def test_single_blank_is_unique(self):
        assert has_unique_solution(KNOWN_PUZZLE_ONE_BLANK) is True

    def test_fully_blank_grid_is_not_unique(self):
        # an empty grid has millions of valid solutions
        assert has_unique_solution([0] * 81) is False


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_generate_full_grid_is_legal_and_complete(self):
        grid = generate_full_grid()
        validate_grid(grid, allow_blanks=False)  # no zeros, no duplicates

    def test_generate_full_grid_is_random(self):
        # two generations should (almost certainly) differ
        g1 = generate_full_grid()
        g2 = generate_full_grid()
        assert g1 != g2

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "expert"])
    def test_generate_puzzle_has_unique_solution(self, difficulty):
        puzzle, solution = generate_puzzle(difficulty)
        validate_grid(puzzle)
        validate_grid(solution, allow_blanks=False)
        assert has_unique_solution(puzzle)
        # the puzzle's clues must all agree with the solution
        for given, solved in zip(puzzle, solution):
            if given != 0:
                assert given == solved

    def test_easier_puzzles_have_more_clues_than_harder_ones(self):
        easy_puzzle, _ = generate_puzzle("easy")
        expert_puzzle, _ = generate_puzzle("expert")
        easy_clues = sum(1 for v in easy_puzzle if v != 0)
        expert_clues = sum(1 for v in expert_puzzle if v != 0)
        assert easy_clues > expert_clues


# ---------------------------------------------------------------------------
# Difficulty rating
# ---------------------------------------------------------------------------

class TestDifficultyRating:
    def test_returns_valid_rating_label(self):
        puzzle, _ = generate_puzzle("medium")
        rating, stats = rate_difficulty(puzzle)
        assert rating in ("easy", "medium", "hard", "expert")
        assert stats["clues"] == sum(1 for v in puzzle if v != 0)

    def test_nearly_full_grid_rates_easy(self):
        # a grid with only 2 blanks should always be trivially easy
        nearly_full = list(KNOWN_SOLVED)
        nearly_full[0] = 0
        nearly_full[1] = 0
        rating, stats = rate_difficulty(nearly_full)
        assert rating == "easy"
        assert stats["solved_by_pure_logic"] is True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_grid_passes(self):
        validate_grid(KNOWN_SOLVED, allow_blanks=False)  # should not raise

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="81 cells"):
            validate_grid([1, 2, 3])

    def test_out_of_range_value_raises(self):
        broken = list(KNOWN_SOLVED)
        broken[0] = 10
        with pytest.raises(ValueError, match="Invalid value"):
            validate_grid(broken)

    def test_negative_value_raises(self):
        broken = list(KNOWN_SOLVED)
        broken[0] = -1
        with pytest.raises(ValueError, match="Invalid value"):
            validate_grid(broken)

    def test_duplicate_in_row_raises(self):
        broken = list(KNOWN_SOLVED)
        broken[1] = broken[0]  # duplicate within row 0
        with pytest.raises(ValueError, match="row 0"):
            validate_grid(broken)

    def test_duplicate_in_column_raises(self):
        # isolated grid: only cells 0 and 27 are set (column 0), nothing
        # else - avoids accidentally also tripping a row conflict
        broken = [0] * 81
        broken[0] = 5
        broken[27] = 5  # row 3, col 0 - same column as cell 0
        with pytest.raises(ValueError, match="column 0"):
            validate_grid(broken)

    def test_duplicate_in_box_raises(self):
        broken = list(KNOWN_SOLVED)
        broken[10] = broken[0]  # row1,col1 within same 3x3 box as row0,col0
        with pytest.raises(ValueError):
            validate_grid(broken)

    def test_blanks_disallowed_when_flag_set(self):
        with pytest.raises(ValueError, match="Invalid value 0"):
            validate_grid(KNOWN_PUZZLE_ONE_BLANK, allow_blanks=False)


# ---------------------------------------------------------------------------
# Sudoku class
# ---------------------------------------------------------------------------

class TestSudokuClass:
    def test_generate_produces_valid_rated_puzzle(self):
        game = Sudoku.generate("medium")
        assert game.difficulty is not None
        assert game.solution is not None
        assert game.clues == sum(1 for v in game.puzzle if v != 0)

    def test_construction_validates_input(self):
        with pytest.raises(ValueError):
            Sudoku([1, 2, 3])  # wrong length

    def test_solve_matches_known_solution(self):
        game = Sudoku(KNOWN_PUZZLE_ONE_BLANK)
        result = game.solve()
        assert result == KNOWN_SOLVED
        assert game.solution == KNOWN_SOLVED

    def test_solve_raises_on_unsolvable_puzzle(self):
        game = Sudoku(KNOWN_SOLVED)  # constructs fine - it's legal
        broken = list(KNOWN_SOLVED)
        broken[1] = broken[0]
        game.puzzle = broken  # simulate a corrupted in-memory state post-construction
        with pytest.raises(ValueError, match="no solution"):
            game.solve()

    def test_is_unique_reflects_puzzle_state(self):
        game = Sudoku(KNOWN_PUZZLE_ONE_BLANK)
        assert game.is_unique() is True

    def test_str_returns_9x9_display(self):
        game = Sudoku(KNOWN_SOLVED)
        text = str(game)
        assert len(text.split("\n")) == 9


# ---------------------------------------------------------------------------
# Save / load round trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_load_round_trip(self, tmp_path):
        game = Sudoku.generate("easy")
        path = tmp_path / "puzzle.json"
        game.save(str(path))

        reloaded = Sudoku.from_file(str(path))
        assert reloaded.puzzle == game.puzzle
        assert reloaded.solution == game.solution
        assert reloaded.difficulty == game.difficulty
        assert reloaded.clues == game.clues

    def test_load_rejects_corrupted_puzzle(self, tmp_path):
        path = tmp_path / "corrupt.json"
        save_puzzle(str(path), KNOWN_SOLVED)
        # manually corrupt the saved file by introducing a duplicate
        import json
        with open(path) as f:
            data = json.load(f)
        data["puzzle"][1] = data["puzzle"][0]
        with open(path, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError):
            load_puzzle(str(path))

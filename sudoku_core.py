"""
Sudoku core: solver, generator, and difficulty rating.

Design notes:
- Grid is a flat list of 81 ints, 0 = empty. Row-major order.
- Solver uses backtracking + constraint propagation (candidate elimination)
  so it's fast enough to both solve and to verify uniqueness while generating.
- Generator: start from a random full solved grid, then remove cells one by
  one, checking after each removal that the puzzle still has exactly one
  solution. This guarantees every generated puzzle is valid.
- Difficulty is estimated from how many "naked/hidden single" steps a human-style
  solver can do before it needs backtracking, plus number of givens.
"""

import json
import random

SIZE = 9
BOX = 3


# ---------------------------------------------------------------------------
# Basic grid helpers
# ---------------------------------------------------------------------------

def row_of(i):
    return i // SIZE


def col_of(i):
    return i % SIZE


def box_of(i):
    r, c = row_of(i), col_of(i)
    return (r // BOX) * BOX + (c // BOX)


def peers(i):
    """Indices sharing a row, column, or box with cell i (excluding i)."""
    r, c, b = row_of(i), col_of(i), box_of(i)
    br, bc = (b // BOX) * BOX, (b % BOX) * BOX
    result = set()
    for k in range(SIZE):
        result.add(r * SIZE + k)          # same row
        result.add(k * SIZE + c)          # same column
    for dr in range(BOX):
        for dc in range(BOX):
            result.add((br + dr) * SIZE + (bc + dc))  # same box
    result.discard(i)
    return result


PEERS = [peers(i) for i in range(81)]


def candidates(grid, i):
    if grid[i] != 0:
        return set()
    used = {grid[p] for p in PEERS[i] if grid[p] != 0}
    return set(range(1, 10)) - used


def is_valid_placement(grid, i, val):
    return all(grid[p] != val for p in PEERS[i])


# ---------------------------------------------------------------------------
# Solver (backtracking with minimum-remaining-values heuristic)
# ---------------------------------------------------------------------------

def solve(grid, count_solutions_limit=1):
    """
    Returns a list of solutions (up to count_solutions_limit) found.
    Uses MRV: always branch on the empty cell with fewest candidates first,
    which prunes the search tree hard and makes 9x9 sudoku solve almost
    instantly even for near-empty grids.
    """
    solutions = []
    work = list(grid)

    # if the given (pre-filled) cells conflict with each other, the puzzle
    # is unsolvable outright - catch this before searching, since the MRV
    # loop below only ever looks at empty cells and would otherwise miss it
    for i, v in enumerate(work):
        if v != 0 and any(work[p] == v for p in PEERS[i]):
            return solutions  # empty list - conflicting givens, no solution

    def backtrack():
        if len(solutions) >= count_solutions_limit:
            return

        # find empty cell with fewest candidates (MRV heuristic)
        best_i, best_cands = -1, None
        for i in range(81):
            if work[i] == 0:
                c = candidates(work, i)
                if best_cands is None or len(c) < len(best_cands):
                    best_i, best_cands = i, c
                    if len(c) == 0:
                        return  # dead end, prune immediately
                    if len(c) == 1:
                        break  # can't do better than 1 candidate

        if best_i == -1:
            solutions.append(list(work))  # no empty cells left -> solved
            return

        for val in best_cands:
            work[best_i] = val
            backtrack()
            work[best_i] = 0
            if len(solutions) >= count_solutions_limit:
                return

    backtrack()
    return solutions


def has_unique_solution(grid):
    return len(solve(grid, count_solutions_limit=2)) == 1


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_full_grid():
    """Produce a random, fully solved, valid 81-cell grid."""
    grid = [0] * 81
    cells = list(range(81))

    def backtrack(pos=0):
        if pos == 81:
            return True
        i = cells[pos]
        if grid[i] != 0:
            return backtrack(pos + 1)
        vals = list(range(1, 10))
        random.shuffle(vals)
        for v in vals:
            if is_valid_placement(grid, i, v):
                grid[i] = v
                if backtrack(pos + 1):
                    return True
                grid[i] = 0
        return False

    backtrack()
    return grid


DIFFICULTY_TARGET_CLUES = {
    "easy": 40,
    "medium": 32,
    "hard": 26,
    "expert": 22,
}


def generate_puzzle(difficulty="medium"):
    """
    Removes cells from a full grid one at a time (random order), keeping
    the removal only if the puzzle still has a unique solution, until we
    hit the target clue count for the requested difficulty (or run out of
    safe removals, whichever comes first).
    """
    target_clues = DIFFICULTY_TARGET_CLUES.get(difficulty, 32)
    full = generate_full_grid()
    puzzle = list(full)

    order = list(range(81))
    random.shuffle(order)
    clues = 81

    for i in order:
        if clues <= target_clues:
            break
        backup = puzzle[i]
        puzzle[i] = 0
        if has_unique_solution(puzzle):
            clues -= 1
        else:
            puzzle[i] = backup  # revert, removal broke uniqueness

    return puzzle, full


# ---------------------------------------------------------------------------
# Human-style difficulty rating
# ---------------------------------------------------------------------------

def rate_difficulty(puzzle):
    """
    Simulates a human solver using only naked singles and hidden singles.
    Returns (rating, stats). If those two techniques alone can't finish the
    puzzle, it needs guessing/backtracking, so it's rated 'expert'.
    """
    grid = list(puzzle)
    steps = 0

    def naked_singles():
        nonlocal steps
        progressed = False
        for i in range(81):
            if grid[i] == 0:
                c = candidates(grid, i)
                if len(c) == 1:
                    grid[i] = next(iter(c))
                    steps += 1
                    progressed = True
        return progressed

    def hidden_singles():
        nonlocal steps
        progressed = False
        groups = []
        for r in range(9):
            groups.append([r * 9 + c for c in range(9)])
        for c in range(9):
            groups.append([r * 9 + c for r in range(9)])
        for b in range(9):
            br, bc = (b // 3) * 3, (b % 3) * 3
            groups.append([(br + dr) * 9 + (bc + dc) for dr in range(3) for dc in range(3)])

        for group in groups:
            for val in range(1, 10):
                spots = [i for i in group if grid[i] == 0 and val in candidates(grid, i)]
                if len(spots) == 1:
                    grid[spots[0]] = val
                    steps += 1
                    progressed = True
        return progressed

    while True:
        if naked_singles() or hidden_singles():
            continue
        break

    solved_by_logic = all(v != 0 for v in grid)
    clue_count = sum(1 for v in puzzle if v != 0)

    if not solved_by_logic:
        rating = "expert"
    elif steps < 35:
        rating = "easy"
    elif steps < 55:
        rating = "medium"
    else:
        rating = "hard"

    return rating, {"logic_steps": steps, "clues": clue_count, "solved_by_pure_logic": solved_by_logic}


def validate_grid(grid, allow_blanks=True):
    """
    Full sudoku legality check. Raises ValueError with a specific message
    on anything malformed. This is the safety net for CV/digit-recognition
    input, which will occasionally misread a cell and produce an illegal
    grid — this must fail loudly rather than let the solver silently choke.
    """
    if len(grid) != 81:
        raise ValueError(f"Grid must have 81 cells, got {len(grid)}")

    allowed = set(range(0, 10)) if allow_blanks else set(range(1, 10))
    for i, v in enumerate(grid):
        if v not in allowed:
            raise ValueError(f"Invalid value {v} at cell {i} (row {row_of(i)}, col {col_of(i)})")

    # check no duplicates in any row, column, or box
    def check_group(indices, label):
        seen = {}
        for i in indices:
            v = grid[i]
            if v == 0:
                continue
            if v in seen:
                raise ValueError(
                    f"Duplicate value {v} in {label}: cells {seen[v]} and {i} "
                    f"(row {row_of(i)}, col {col_of(i)})"
                )
            seen[v] = i

    for r in range(9):
        check_group([r * 9 + c for c in range(9)], f"row {r}")
    for c in range(9):
        check_group([r * 9 + c for r in range(9)], f"column {c}")
    for b in range(9):
        br, bc = (b // 3) * 3, (b % 3) * 3
        check_group([(br + dr) * 9 + (bc + dc) for dr in range(3) for dc in range(3)], f"box {b}")


# ---------------------------------------------------------------------------
# Save / load (JSON) — this is the exact shape the FastAPI backend will
# later send/receive over HTTP, so building it now isn't throwaway work.
# ---------------------------------------------------------------------------

def save_puzzle(path, puzzle, solution=None, difficulty=None):
    """Write a puzzle (and optionally its solution/difficulty) to a JSON file."""
    data = {"puzzle": list(puzzle)}
    if solution is not None:
        data["solution"] = list(solution)
    if difficulty is not None:
        data["difficulty"] = difficulty
    data["clues"] = sum(1 for v in puzzle if v != 0)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_puzzle(path):
    """
    Read a puzzle back from a JSON file. Returns the parsed dict.
    Runs validate_grid() on the puzzle so a corrupted/tampered file
    fails loudly here rather than silently breaking the solver later.
    """
    with open(path) as f:
        data = json.load(f)
    validate_grid(data["puzzle"])
    if "solution" in data:
        validate_grid(data["solution"], allow_blanks=False)
    return data


# ---------------------------------------------------------------------------
# Sudoku class — OOP wrapper around the functions above. This is what the
# FastAPI backend and any other caller should actually use; the free
# functions above remain as the underlying engine.
# ---------------------------------------------------------------------------

class Sudoku:
    """
    Represents one puzzle instance.

    Attributes:
        puzzle: flat list of 81 ints (0 = blank) — the starting grid
        solution: flat list of 81 ints — the filled solution (or None until solved)
        difficulty: str rating, e.g. "medium" (or None until rated)
        clues: number of non-zero cells in puzzle
    """

    def __init__(self, puzzle, solution=None, difficulty=None):
        validate_grid(puzzle)
        self.puzzle = list(puzzle)
        self.solution = list(solution) if solution is not None else None
        self.difficulty = difficulty
        self.clues = sum(1 for v in self.puzzle if v != 0)

    # -- construction --------------------------------------------------

    @classmethod
    def generate(cls, difficulty="medium"):
        """Create a brand new puzzle at the requested difficulty."""
        puzzle, solution = generate_puzzle(difficulty)
        instance = cls(puzzle, solution)
        instance.rate()  # fills in the actual measured difficulty
        return instance

    @classmethod
    def from_file(cls, path):
        data = load_puzzle(path)
        return cls(data["puzzle"], data.get("solution"), data.get("difficulty"))

    # -- behavior ---------------------------------------------------------

    def solve(self):
        """Solve self.puzzle, store and return the solution. Raises if unsolvable."""
        results = solve(self.puzzle, count_solutions_limit=1)
        if not results:
            raise ValueError("Puzzle has no solution — check for input errors")
        self.solution = results[0]
        return self.solution

    def is_unique(self):
        return has_unique_solution(self.puzzle)

    def rate(self):
        """Compute and store self.difficulty, return it."""
        self.difficulty, _stats = rate_difficulty(self.puzzle)
        return self.difficulty

    def save(self, path):
        save_puzzle(path, self.puzzle, self.solution, self.difficulty)

    def __str__(self):
        return to_string(self.puzzle)


# ---------------------------------------------------------------------------
# Pretty printing / demo
# ---------------------------------------------------------------------------

def to_string(grid):
    lines = []
    for r in range(9):
        row = grid[r * 9:(r + 1) * 9]
        lines.append(" ".join(str(v) if v != 0 else "." for v in row))
    return "\n".join(lines)


if __name__ == "__main__":
    game = Sudoku.generate("medium")
    print("Puzzle:")
    print(game)
    print("\nSolution:")
    print(to_string(game.solution))

    print(f"\nDifficulty: {game.difficulty}")
    print(f"Clues: {game.clues}")
    print(f"Unique solution: {game.is_unique()}")

    game.save("last_puzzle.json")
    print("\nSaved to last_puzzle.json")

    reloaded = Sudoku.from_file("last_puzzle.json")
    print(f"Reloaded ok, clues={reloaded.clues}, difficulty={reloaded.difficulty}")

    # demonstrate solving a puzzle that came in with no solution attached
    blank_solution_game = Sudoku(game.puzzle)
    blank_solution_game.solve()
    print(f"\nRe-solved from scratch matches original: {blank_solution_game.solution == game.solution}")

    # demonstrate validation catching a broken grid (simulates a CV misread)
    broken = list(game.puzzle)
    broken[0] = broken[1] if broken[1] != 0 else 5  # force a duplicate in row 0
    try:
        validate_grid(broken)
        print("\nValidation did not catch the broken grid (unexpected)")
    except ValueError as e:
        print(f"\nValidation correctly caught bad input: {e}")

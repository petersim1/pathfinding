import random
from queue import LifoQueue

from .search import get_neighbors
from .types import Grid_t, Position_t


def is_path_valid(start: Position_t, target: Position_t, matrix: Grid_t):
    """
    DFS for an adjacency matrix.

    Args:
    - start (int, int)
    - end (int, int)
    - matrix: adjacency matrix

    Returns:
    - valid_path boolean
    - stack int[]: current stack, can be complete for path, or incomplete.
    """

    max_size = len(matrix) * len(matrix[0])

    visited = set([start])
    stack = LifoQueue(maxsize=max_size)
    stack.put(start)
    is_valid_path = False

    while stack.qsize():
        cur_node = stack.queue[-1]
        if cur_node == target:
            is_valid_path = True
            break

        is_available = False
        neighbors = get_neighbors(cur_node, matrix)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.put(neighbor)
                is_available = True
                break
        if not is_available:
            stack.get()

    return is_valid_path


def generate_random_grid(
    n: int, m: int, percent_blocked: float = 0.3
) -> tuple[Position_t, Position_t, Grid_t]:
    """
    Create an N x M grid with a starting and ending position.
    Randomly block off some squares with a given chance, ensuring there's at least one
    valid path.
    Prevent start and end from being neighbors.
    We'll use DFS for ensuring a valid path, and randomly block of squares.

    Args:
    - n (int): The height of the grid.
    - m (int): The width of the grid.
    - percent_blocked (float): The probability of a square being blocked.
    Defaults to 0.3.

    Returns:
    - tuple[position_t, position_t, grid_t]: A tuple containing the start position, end
    position, and the grid.
    """

    if n < 2:
        raise ValueError("N is too small")
    if m < 2:
        raise ValueError("M is too small")
    percent_blocked = max(0, min(1, percent_blocked))

    available_positions = [(r, c) for c in range(m) for r in range(n)]

    start = random.choice(available_positions)
    available_positions.remove(start)
    end = random.choice(
        [
            pos
            for pos in available_positions
            if abs(pos[0] - start[0]) + abs(pos[1] - start[1]) > 1
        ]
    )
    available_positions.remove(end)

    grid = [[1 for _ in range(m)] for _ in range(n)]

    for r in range(n):
        for c in range(m):
            pos = (r, c)
            if pos == start:
                continue
            if pos == end:
                continue
            if random.random() < percent_blocked:
                grid[r][c] = 0
                valid_path = is_path_valid(start, end, grid)
                if not valid_path:
                    grid[r][c] = 1  # unblock it if there isn't a valid path.

    return start, end, grid


def generate_fixed_grid(n: int, m: int, variant: int):

    if n < 4:
        raise ValueError("N is too small for the fixed grid")
    if m < 4:
        raise ValueError("M is too small for the fixed grid")

    variant = max(0, min(variant, 2))

    grid = [[1 for _ in range(m)] for _ in range(n)]

    row = int(n / 2)
    col_block = 2 * int(m / 3)

    start = (row, 0)
    end = (row, col_block + 1)

    if variant == 0:
        for i in range(1, n - 1):
            grid[i][col_block] = 0
    elif variant == 1:
        for i in range(2, n - 2):
            grid[i][int(m / 2)] = 0
        for i in range(2, int(m / 2)):
            grid[2][i] = 0
            grid[-3][i] = 0

    else:
        for i in range(1, n - 1):
            grid[i][int(col_block / 2)] = 0
            grid[i][col_block] = 0

    return start, end, grid

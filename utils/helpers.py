import random
from queue import LifoQueue

from .search import get_neighbors, manhattan_distance
from .types import Grid_t, Position_t

"""
4 primary methods of maze / grid generation

- Maze
- Grid -> viable path created (random walk with stack), blocks outside that path
randomly filled
- Grid -> randomly fill blocks, ensure viable path in each iteration
- Fixed -> Some fixed grid variants to explore.

In all cases, 1 denotes valid unit, 0 denotes blocked unit.
"""


def is_path_valid(start: Position_t, target: Position_t, matrix: Grid_t):
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


def generate_maze(rows, cols):
    """There's a bug in here, sometimes generates in invalid maze"""
    # Initialize the grid with walls (1) and spaces (0)
    maze = [[0 for _ in range(cols)] for _ in range(rows)]

    # Define the directions for moving in the maze (right, down, left, up)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def is_valid_move(r, c):
        # Check if the move is within the maze boundaries
        return 0 <= r < rows and 0 <= c < cols

    def carve_passages_from(cr, cc):
        # Randomly shuffle the directions to create a more random maze
        random.shuffle(directions)

        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            nr2, nc2 = cr + 2*dr, cc + 2*dc

            if is_valid_move(nr2, nc2) and maze[nr2][nc2] == 0:
                # Carve through the blockages to create passages
                maze[nr][nc] = 1
                maze[nr2][nc2] = 1
                carve_passages_from(nr2, nc2)

    # Ensure the start and end positions are within bounds
    start_r, start_c = 0, 1 if cols > 1 else 0
    end_r, end_c = rows - 1, cols - 2 if cols > 1 else cols - 1

    # Start carving from a central location
    start_cr, start_cc = 1 if rows > 1 else 0, 1 if cols > 1 else 0
    maze[start_cr][start_cc] = 1
    carve_passages_from(start_cr, start_cc)

    # Ensure the start (top-left) and end (bottom-right) are open
    maze[start_r][start_c] = 1
    maze[end_r][end_c] = 1

    return (start_r, start_c), (end_r, end_c), maze


def generate_grid(rows: int, cols: int, percent_blocked: float = 0.3):
    def generate_viable_path(rows, cols):
        start = (random.choice(range(rows)), random.choice(range(cols)))

        # Worst case scenario, we start in the center, this is the
        # maximum distance we can be.
        min_allowable_distance = int(rows / 2) + int(cols / 2)
        max_size = rows * cols

        cur_distance = 0

        stack = LifoQueue(maxsize=max_size)
        stack.put(start)
        visited = set([start])

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while True:
            cur_node = stack.queue[-1]
            cur_distance = manhattan_distance(start, cur_node)
            is_available = False
            random.shuffle(directions)

            for d in directions:
                pos = (cur_node[0] + d[0], cur_node[1] + d[1])
                is_valid = (0 <= pos[0] <= rows - 1) and (0 <= pos[1] <= cols - 1)
                if is_valid:
                    if pos not in visited:
                        stack.put(pos)
                        visited.add(pos)
                        cur_distance = manhattan_distance(start, pos)
                        is_available = True
                        break
            if not is_available:
                # preven backtracking if desired distance is already achieved
                # (here due to the condition below for possible extensions)
                if cur_distance >= min_allowable_distance:
                    break
                stack.get()

            if cur_distance >= min_allowable_distance:
                # allow for a chance of extending past min_allowable_distance
                if random.random() < 0.75:
                    break

        return stack.queue

    if rows < 2:
        raise ValueError("N is too small")
    if cols < 2:
        raise ValueError("M is too small")
    percent_blocked = max(0, min(1, percent_blocked))

    grid = [[1 for _ in range(cols)] for _ in range(rows)]
    path = generate_viable_path(rows, cols)

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in path:
                if random.random() < percent_blocked:
                    grid[r][c] = 0

    return path[0], path[-1], grid


def generate_random_grid(
    rows: int, cols: int, percent_blocked: float = 0.3
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

    if rows < 2:
        raise ValueError("N is too small")
    if cols < 2:
        raise ValueError("M is too small")
    percent_blocked = max(0, min(1, percent_blocked))

    available_positions = [(r, c) for c in range(cols) for r in range(rows)]

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

    grid = [[1 for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
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

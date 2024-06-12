import random
from queue import LifoQueue

from .graph import Graph
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


def generate_maze(graph: Graph):
    """There's a bug in here, sometimes generates in invalid maze"""
    # Initialize the grid with walls (1) and spaces (0)
    # Define the directions for moving in the maze (right, down, left, up)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def carve_passages_from(cr, cc):
        # Randomly shuffle the directions to create a more random maze
        random.shuffle(directions)

        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            nr2, nc2 = cr + 2 * dr, cc + 2 * dc

            if graph.is_in_grid((nr2, nc2)) and graph.is_blocked((nr2, nc2)):
                # Carve through the blockages to create passages
                graph.unblock_node((nr, nc))
                graph.unblock_node((nr2, nc2))
                carve_passages_from(nr2, nc2)

    # Ensure the start and end positions are within bounds
    start_r, start_c = 0, 1 if graph.cols > 1 else 0
    end_r, end_c = graph.rows - 1, graph.cols - 2 if graph.cols > 1 else graph.cols - 1

    # Start carving from a central location
    start_cr, start_cc = 1 if graph.rows > 1 else 0, 1 if graph.cols > 1 else 0
    graph.unblock_node((start_cr, start_cc))
    carve_passages_from(start_cr, start_cc)

    # Ensure the start (top-left) and end (bottom-right) are open
    graph.unblock_node((start_r, start_c))
    graph.unblock_node((end_r, end_c))

    return (start_r, start_c), (end_r, end_c)


def generate_grid(graph: Graph, percent_blocked: float = 0.3):
    """
    Generates a random grid by random walking a path until a certain manhattan distance
    is met. Then we fill blocks around it.
    """

    def generate_viable_path():
        start = (random.choice(range(graph.rows)), random.choice(range(graph.cols)))
        # Worst case scenario, we start in the center, this is the
        # maximum distance we can be.
        min_allowable_distance = int(graph.rows / 2) + int(graph.cols / 2)
        max_size = graph.rows * graph.cols

        cur_distance = 0

        stack = LifoQueue(maxsize=max_size)
        stack.put(start)
        visited = set([start])

        while True:
            cur_node = stack.queue[-1]
            cur_distance = Graph.distance(start, cur_node)

            if cur_distance >= min_allowable_distance:
                # allow for a chance of extending past min_allowable_distance
                if random.random() < 0.75:
                    break

            is_available = False
            neighbors = graph.get_neighbors(cur_node)
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in visited:
                    if not graph.crosses_path(cur_node, neighbor, visited):
                        stack.put(neighbor)
                        visited.add(neighbor)
                        is_available = True
                        break
            if not is_available:
                # preven backtracking if desired distance is already achieved
                # (here due to the condition below for possible extensions)
                if cur_distance >= min_allowable_distance:
                    break
                stack.get()

        return stack.queue

    if graph.rows < 2:
        raise ValueError("Rows is too small")
    if graph.cols < 2:
        raise ValueError("Cols is too small")
    percent_blocked = max(0, min(1, percent_blocked))

    path = generate_viable_path()

    for r in range(graph.rows):
        for c in range(graph.cols):
            if (r, c) not in path:
                does_bisect = False
                for d in graph.get_all_neighbors((r, c)):
                    # similar to checking if diagonal paths cross,
                    # we can check if 2 blocked diagonal nodes would prevent
                    # a path from passing through
                    if graph.crosses_path((r, c), d, path):
                        does_bisect = True
                        break
                if not does_bisect:
                    if random.random() < percent_blocked:
                        graph.block_node((r, c))

    return path[0], path[-1]


def generate_random_grid(
    graph: Graph, percent_blocked: float = 0.3
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

    if graph.rows < 2:
        raise ValueError("Rows is too small")
    if graph.cols < 2:
        raise ValueError("Cols is too small")
    percent_blocked = max(0, min(1, percent_blocked))

    available_r = list(range(graph.rows))
    available_c = list(range(graph.cols))

    start_r = random.choice(available_r)
    start_c = random.choice(available_c)

    available_r.remove(start_r)
    available_c.remove(start_c)

    end_r = random.choice(available_r)
    end_c = random.choice(available_c)

    start = (start_r, start_c)
    end = (end_r, end_c)

    for r in range(graph.rows):
        for c in range(graph.cols):
            pos = (r, c)
            if pos == start:
                continue
            if pos == end:
                continue
            if random.random() < percent_blocked:
                graph.block_node((r, c))
                valid_path = graph.path_exists(start, end)
                if not valid_path:
                    graph.unblock_node((r, c))  # unblock if needed

    return start, end


def generate_fixed_grid(graph: Graph, variant: int):

    if graph.rows < 4:
        raise ValueError("Rows is too small")
    if graph.cols < 4:
        raise ValueError("Cols is too small")

    variant = max(0, min(variant, 2))

    row = int(graph.rows / 2)
    col_block = 2 * int(graph.cols / 3)

    start = (row, 0)
    end = (row, col_block + 1)

    if variant == 0:
        for r in range(1, graph.rows - 1):
            graph.block_node((r, col_block))
    elif variant == 1:
        for r in range(2, graph.rows - 2):
            graph.block_node((r, int(graph.cols / 2)))
        for c in range(2, int(graph.cols / 2)):
            graph.block_node((2, c))
            graph.block_node((-3, c))
    else:
        for r in range(1, graph.rows - 1):
            graph.block_node((r, int(col_block / 2)))
            graph.block_node((r, col_block))

    return start, end

from collections.abc import Iterator
from queue import LifoQueue, PriorityQueue, Queue

from .types import Grid_t, Position_t


def recreate_path(method: str, **kwargs):
    if method.lower() == "dfs":
        if "stack" not in kwargs:
            raise KeyError("must include stack for DFS")
        return kwargs["stack"].queue[::-1]

    if "parents" not in kwargs:
        raise KeyError("must include parents for BFS")
    if "target" not in kwargs:
        raise KeyError("must include target for BFS")

    path = []
    node = kwargs["target"]
    while node is not None:
        path.insert(0, node)
        node = kwargs["parents"][node]
    return path


def manhattan_distance(point_a, point_b):
    return abs(point_a[0] - point_b[0]) + abs(point_a[1] - point_b[1])


def get_neighbors(pos: Position_t, matrix: Grid_t) -> Iterator[Position_t]:
    """
    whichever path algorithm is used, it'll add neighbors in the following order:
    UP, LEFT, RIGHT, DOWN
    """

    r_d = len(matrix)
    c_d = len(matrix[0])
    r, c = pos

    for r_i in [-1, 0, 1]:
        for c_i in [-1, 0, 1]:
            if abs(r_i + c_i) != 1:
                continue  # no diagonal moves allowed. don't return starting position
            if r + r_i >= r_d:
                continue
            if c + c_i >= c_d:
                continue
            if r + r_i < 0:
                continue
            if c + c_i < 0:
                continue
            if matrix[r + r_i][c + c_i] == 0:
                continue
            yield (r + r_i, c + c_i)


def bfs(start: Position_t, target: Position_t, matrix: Grid_t):
    """
    BFS for an adjacency matrix.
    Streaming enabled, yields the current explored node, OR path, OR None (no path).

    Args:
    - start (int, int)
    - end (int, int)
    - matrix: adjacency matrix

    Returns:
    - valid_path boolean
    - parents dict: adjacency list of parent relations, for rebuilding path.
    - complexity: dict
        - time (s)
        - iterations int
    """
    visited = set([start])
    queue = Queue()
    queue.put(start)
    parents = {start: None}
    is_valid_path = False

    while queue.qsize():
        cur_node = queue.get()
        yield ("search", cur_node)
        if cur_node == target:
            is_valid_path = True
            break

        neighbors = get_neighbors(cur_node, matrix)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.put(neighbor)
                parents[neighbor] = cur_node

    if is_valid_path:
        path = recreate_path(method="bfs", target=target, parents=parents)
        yield ("path", path)
    else:
        yield ("error", None)


def dfs(start: Position_t, target: Position_t, matrix: Grid_t):
    """
    DFS for an adjacency matrix.
    Streaming enabled, yields the current explored node, OR path, OR None (no path).

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
        yield ("search", cur_node)
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

    if is_valid_path:
        path = recreate_path(method="dfs", stack=stack)
        yield ("path", path)
    else:
        yield ("error", None)


def dfs_heuristic(start: Position_t, target: Position_t, matrix: Grid_t):
    """
    DFS for an adjacency matrix. Includes some simple heuristic
    Streaming enabled, yields the current explored node, OR path, OR None (no path).

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
        yield ("search", cur_node)
        if cur_node == target:
            is_valid_path = True
            break

        is_available = False
        neighbors = get_neighbors(cur_node, matrix)
        neighbors = sorted(
            map(lambda x: (manhattan_distance(x, target), x), neighbors),
            key=lambda x : x[0]
        )
        for _, neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.put(neighbor)
                is_available = True
                break
        if not is_available:
            stack.get()

    if is_valid_path:
        path = recreate_path(method="dfs", stack=stack)
        yield ("path", path)
    else:
        yield ("error", None)


def greedy_bfs(start: Position_t, target: Position_t, matrix: Grid_t):
    """
    Greedy BFS for an adjacency matrix, using basic heuristics.
    Streaming enabled, yields the current explored node, OR path, OR None (no path).

    Args:
    - start (int, int)
    - end (int, int)
    - matrix: adjacency matrix

    Returns:
    - valid_path boolean
    - parents dict: adjacency list of parent relations, for rebuilding path.
    - complexity: dict
        - time (s)
        - iterations int
    """
    visited = set([start])
    queue = PriorityQueue()
    queue.put((0, start))
    parents = {start: None}
    is_valid_path = False

    while not queue.empty():
        cur_node = queue.get()[1]
        yield ("search", cur_node)
        if cur_node == target:
            is_valid_path = True
            break

        neighbors = get_neighbors(cur_node, matrix)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                distance = manhattan_distance(neighbor, target)
                queue.put((distance, neighbor))
                parents[neighbor] = cur_node

    if is_valid_path:
        path = recreate_path(method="bfs", target=target, parents=parents)
        yield ("path", path)
    else:
        yield ("error", None)


def a_star(start: Position_t, target: Position_t, matrix: Grid_t):
    """
    A* for an adjacency matrix.
    Our costs are consistent, since we assume equal weights of edges.

    F = G + H
    F -> total cost of node
    G -> distance between node and start
    H -> distance between node and target

    Streaming enabled, yields the current explored node, OR path, OR None (no path).

    Args:
    - start (int, int)
    - end (int, int)
    - matrix: adjacency matrix

    Returns:
    - valid_path boolean
    - parents dict: adjacency list of parent relations, for rebuilding path.
    - complexity: dict
        - time (s)
        - iterations int
    """
    visited = set(
        [start]
    )  # becomes repetitive since i have "g_costs", but I'll keep it.
    queue = PriorityQueue()
    # priority should be the cost, but it's popped immediately so it doesn't matter.
    queue.put((0, start))
    parents = {start: None}
    g_costs = {start: 0}
    is_valid_path = False

    while not queue.empty():
        cur_node = queue.get()[1]
        yield ("search", cur_node)
        if cur_node == target:
            is_valid_path = True
            break

        g_cost = g_costs[cur_node] + 1

        neighbors = get_neighbors(cur_node, matrix)
        for neighbor in neighbors:
            # ALSO, if the current g_cost is less than neighbor's, allow moving to that
            # (essentially an overwrite).
            if neighbor not in visited or g_cost < g_costs[neighbor]:
                visited.add(neighbor)
                h = manhattan_distance(neighbor, target)
                f_cost = g_cost + h
                queue.put((f_cost, neighbor))
                parents[neighbor] = cur_node
                g_costs[neighbor] = g_cost

    if is_valid_path:
        path = recreate_path(method="bfs", target=target, parents=parents)
        yield ("path", path)
    else:
        yield ("error", None)
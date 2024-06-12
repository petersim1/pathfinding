from queue import LifoQueue, PriorityQueue, Queue

from .graph import Graph
from .types import Position_t


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


def bfs(start: Position_t, target: Position_t, graph: Graph, **kwargs):
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

        neighbors = graph.get_neighbors(cur_node)
        for neighbor in neighbors:
            if neighbor not in visited:
                if not graph.crosses_path(cur_node, neighbor, visited):
                    visited.add(neighbor)
                    queue.put(neighbor)
                    parents[neighbor] = cur_node

    if is_valid_path:
        path = recreate_path(method="bfs", target=target, parents=parents)
        yield ("path", path)
    else:
        yield ("error", None)


def dfs(start: Position_t, target: Position_t, graph: Graph, **kwargs):
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

    max_size = graph.rows * graph.cols

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
        neighbors = graph.get_neighbors(cur_node)
        for neighbor in neighbors:
            if neighbor not in visited:
                if not graph.crosses_path(cur_node, neighbor, visited):
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


def dfs_heuristic(
    start: Position_t, target: Position_t, graph: Graph, method: str = "manhattan"
):
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
    max_size = graph.rows * graph.cols

    visited = set([start])
    stack = LifoQueue(maxsize=max_size)
    stack.put(start)
    values = {start: Graph.distance(start, target, method=method)}
    is_valid_path = False

    while stack.qsize():
        cur_node = stack.queue[-1]
        yield ("search", cur_node)
        if cur_node == target:
            is_valid_path = True
            break

        is_available = False

        to_search = []
        for neighbor in graph.get_neighbors(cur_node):
            if neighbor not in values:
                values[neighbor] = Graph.distance(neighbor, target, method=method)
            value = values[neighbor]
            to_search.append((value, neighbor))
        to_search = sorted(to_search, key=lambda x: x[0])
        for _, neighbor in to_search:
            if neighbor not in visited:
                if not graph.crosses_path(cur_node, neighbor, visited):
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


def greedy_bfs(
    start: Position_t, target: Position_t, graph: Graph, method: str = "manhattan"
):
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

        for neighbor in graph.get_neighbors(cur_node):
            if neighbor not in visited:
                if not graph.crosses_path(cur_node, neighbor, visited):
                    visited.add(neighbor)
                    distance = Graph.distance(neighbor, target, method=method)
                    queue.put((distance, neighbor))
                    parents[neighbor] = cur_node

    if is_valid_path:
        path = recreate_path(method="bfs", target=target, parents=parents)
        yield ("path", path)
    else:
        yield ("error", None)


def a_star(
    start: Position_t, target: Position_t, graph: Graph, method: str = "manhattan"
):
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

        for neighbor in graph.get_neighbors(cur_node):
            # ALSO, if the current g_cost is less than neighbor's, allow moving to that
            # (essentially an overwrite).
            if neighbor not in visited or g_cost < g_costs[neighbor]:
                visited.add(neighbor)
                h = Graph.distance(neighbor, target, method=method)
                f_cost = g_cost + h
                queue.put((f_cost, neighbor))
                parents[neighbor] = cur_node
                g_costs[neighbor] = g_cost

    if is_valid_path:
        path = recreate_path(method="bfs", target=target, parents=parents)
        yield ("path", path)
    else:
        yield ("error", None)


def jps(start: Position_t, target: Position_t, graph: Graph, method: str = "manhattan"):

    def can_enter(pos):
        return graph.is_in_grid(pos) and not graph.is_blocked(pos)

    def jump(node_from, node, scanned=[]):
        r, c = node
        dr, dc = r - node_from[0], c - node_from[1]

        if not can_enter(node):
            return None, scanned

        if node == target:
            return node, scanned

        if dr != 0:  # vertical move
            if (can_enter((r, c - 1)) and not can_enter((r - dr, c - 1))) or (
                can_enter((r, c + 1)) and not can_enter((r - dr, c + 1))
            ):
                return node, scanned
            vert_0, vert_0_s = jump((r, c + 1), node, scanned)
            vert_1, vert_1_s = jump((r, c - 1), node, scanned)
            if (vert_0 is not None) or (vert_1 is not None):
                return node, list(set(vert_1_s + vert_0_s))
        if dc != 0:  # horizontal move
            if (can_enter((r - 1, c)) and not can_enter((r - 1, c - dc))) or (
                can_enter((r + 1, c)) and not can_enter((r + 1, c - dc))
            ):
                return node, scanned

        return jump(node, (r + dr, c + dc), scanned + [node])

    open_list = PriorityQueue()
    open_list.put((0, start))
    parents = {start: None}
    g_score = {start: 0}

    is_valid_path = False

    while not open_list.empty():
        _, current = open_list.get()
        yield ("search", current)

        if current == target:
            is_valid_path = True
            break

        tentative_g_score = g_score[current] + 1

        for neighbor in graph.get_neighbors(current):
            jp, scanned = jump(current, neighbor)
            for s in scanned:
                yield ("scan", s)
            if jp is None:
                continue
            if jp not in g_score or tentative_g_score < g_score[jp]:
                parents[jp] = current
                g_score[jp] = tentative_g_score
                f_score = tentative_g_score + Graph.distance(jp, target, method=method)
                open_list.put((f_score, jp))

    if is_valid_path:
        path = recreate_path(method="bfs", target=target, parents=parents)
        yield ("path", path)
    else:
        yield ("error", None)

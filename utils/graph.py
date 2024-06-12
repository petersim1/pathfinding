from queue import LifoQueue


class Graph:
    """
    Represents the grid (graph), with quick access to neighbors,
    distances, and ability to alter available nodes.
    """

    def __init__(self, rows: int, cols: int, allow_diagonal: bool = False):
        self._rows = rows
        self._cols = cols
        self._allow_diagonal = allow_diagonal

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def allow_diagonal(self):
        return self._allow_diagonal

    @allow_diagonal.setter
    def allow_diagonal(self, value: bool):
        self._allow_diagonal = value

    @property
    def grid(self):
        return self._grid

    @classmethod
    def from_grid(cls, grid, **kwargs):
        rows = len(grid)
        cols = len(grid[0])
        module = cls(rows, cols, **kwargs)
        module._grid = grid
        return module

    def initialize_grid(self, value: int = 1):
        if value not in [0, 1]:
            raise ValueError("grid can only contain 0 or 1")
        grid = [[value for _ in range(self.cols)] for _ in range(self.rows)]
        self._grid = grid

    def block_node(self, node):
        r, c = node
        if r < 0:
            r = self.rows + r
        if c < 0:
            c = self.cols + c
        if not (0 <= r < self.rows):
            raise ValueError("Invalid position for row")
        if not (0 <= c < self.cols):
            raise ValueError("Invalid position for col")
        self._grid[r][c] = 0

    def unblock_node(self, node):
        r, c = node
        if not (0 <= r < self.rows):
            raise ValueError("Invalid position for row")
        if not (0 <= c < self.cols):
            raise ValueError("Invalid position for col")
        self._grid[r][c] = 1

    def is_blocked(self, node):
        r, c = node
        if not self.is_in_grid(node):
            raise ValueError("node is not within grid")
        return self._grid[r][c] == 0

    def is_in_grid(self, node):
        r, c = node
        if 0 <= r < self.rows:
            if 0 <= c < self.cols:
                return True
        return False

    def get_all_neighbors(self, node):
        r, c = node
        neighbors = []
        for r_i in [-1, 0, 1]:
            for c_i in [-1, 0, 1]:
                if r_i == c_i == 0:
                    continue
                if not self._allow_diagonal:
                    if abs(r_i + c_i) != 1:
                        continue
                if not (0 <= r + r_i < self.rows):
                    continue
                if not (0 <= c + c_i < self.cols):
                    continue
                neighbors.append((r + r_i, c + c_i))
        return neighbors

    def get_neighbors(self, node):
        r, c = node
        neighbors = []
        for r_i in [-1, 0, 1]:
            for c_i in [-1, 0, 1]:
                if r_i == c_i == 0:
                    continue
                if not self._allow_diagonal:
                    if abs(r_i + c_i) != 1:
                        continue
                if not (0 <= r + r_i < self.rows):
                    continue
                if not (0 <= c + c_i < self.cols):
                    continue
                if not self.grid[r + r_i][c + c_i]:
                    continue
                if self._allow_diagonal:
                    if abs(r_i) + abs(c_i) == 2:
                        if self.is_blocked((r, c + c_i)) and self.is_blocked(
                            (r + r_i, c)
                        ):
                            continue
                neighbors.append((r + r_i, c + c_i))
        return neighbors

    def crosses_path(self, node_from, node_to, visited):
        if not self.allow_diagonal:
            # will never make it into the candidate neighbor set
            return False

        r_f, c_f = node_from
        r_t, c_t = node_to

        dir_r = r_t - r_f
        dir_c = c_t - c_f

        if (r_f + dir_r, c_f) in visited:
            if (r_f, c_f + dir_c) in visited:
                return True
        return False

    @staticmethod
    def distance(node_a, node_b, method: str = "manhattan"):
        r_a, c_a = node_a
        r_b, c_b = node_b

        d_r = abs(r_a - r_b)
        d_c = abs(c_a - c_b)
        match method:
            case "euclidean":
                return (d_r**2 + d_c**2) ** 0.5
            case "chebyshev":
                return max(d_r, d_c)
            case _:
                return d_r + d_c

    def get_adjacency_list(self):
        adjacency = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.is_blocked((r, c)):
                    adjacency[(r, c)] = self.get_neighbors((r, c))
        return adjacency

    def path_exists(self, start, target):
        max_size = self.rows * self.cols

        visited = set([start])
        stack = LifoQueue(maxsize=max_size)
        stack.put(start)

        while stack.qsize():
            cur_node = stack.queue[-1]
            if cur_node == target:
                return True

            is_available = False
            for neighbor in self.get_neighbors(cur_node):
                if neighbor not in visited:
                    if not self.crosses_path(cur_node, neighbor, visited):
                        visited.add(neighbor)
                        stack.put(neighbor)
                        is_available = True
                        break
            if not is_available:
                stack.get()

        return False

from collections.abc import Iterator
from functools import wraps
from time import time
from typing import List, Tuple

from .graph import Graph
from .helpers import (
    generate_fixed_grid,
    generate_grid,
    generate_maze,
    generate_random_grid,
)


class Search:
    """
    Creates the graph class to manage the grid, start position, and end positions.

    Most variables are private, as updating directly one would put the graph in
    an "unstable" state with many mismatches.
    """

    def __init__(self, rows: int, cols: int):
        self.__rows = rows
        self.__cols = cols
        self.__grid: Graph = None
        self.__p_start = None
        self.__p_end = None

    def _state_check(f):
        def inner(self, *args, **kwargs):
            if self.grid is None:
                raise ValueError("grid isn't set yet, call generate_grid()")
            if self.p_start is None:
                raise ValueError("p_start isn't set yet, call generate_grid()")
            if self.p_end is None:
                raise ValueError("p_end isn't set yet, call generate_grid()")
            return f(self, *args, **kwargs)

        return inner

    def _complexity(f):
        @wraps(f)
        def inner(self, *args, **kwargs):
            now = time()
            complexity = {"iterations": 0, "time": None, "length": None}
            result = f(self, *args, **kwargs)
            path = []
            for r in result:
                if r[0] == "search":
                    complexity["iterations"] += 1
            if r[0] == "path":
                path = r[1]
                complexity["time"] = time() - now
                complexity["length"] = len(path) - 1
            return path, complexity

        return inner

    @property
    def p_start(self) -> Tuple[int, int]:
        return self.__p_start

    @property
    def p_end(self) -> Tuple[int, int]:
        return self.__p_end

    @property
    def dim(self) -> Tuple[int, int]:
        return (self.__rows, self.__cols)

    @property
    def grid(self) -> List[List[0 | 1]]:
        return self.__grid

    @classmethod
    def from_grid(cls, start, end, graph, **kwargs):
        # Initialize from a pre-defined grid, start position, and end position
        grid = Graph.from_grid(grid=graph, **kwargs)
        g = cls(rows=grid.rows, cols=grid.cols)
        g.__p_start = start
        g.__p_end = end
        g.__grid = grid
        return g

    def generate_grid(
        self, grid_type: str = "random", allow_diagonal: bool = False, **kwargs
    ) -> None:

        graph = Graph(rows=self.__rows, cols=self.__cols, allow_diagonal=allow_diagonal)
        match grid_type:
            case "random":
                graph.initialize_grid()
                s, t = generate_random_grid(graph, **kwargs)
            case "random-1":
                graph.initialize_grid()
                s, t = generate_grid(graph, **kwargs)
            case "maze":
                graph.initialize_grid(value=0)
                s, t = generate_maze(graph)
            case "fixed":
                graph.initialize_grid()
                s, t = generate_fixed_grid(graph, **kwargs)
            case _:
                graph.initialize_grid()
                s, t = generate_grid(graph, **kwargs)

        self.__p_start = s
        self.__p_end = t
        self.__grid = graph

    @_state_check
    @_complexity
    def search(self, fct, **kwargs) -> Tuple[List[Tuple[int, int]], object]:
        return fct(self.p_start, self.p_end, self.grid, **kwargs)

    @_state_check
    def search_generator(self, fct, **kwargs) -> Iterator[Tuple[str, any]]:
        return fct(self.p_start, self.p_end, self.grid, **kwargs)

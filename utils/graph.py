from collections.abc import Iterator
from functools import wraps
from time import time
from typing import List, Tuple

from .helpers import (generate_fixed_grid, generate_grid, generate_maze,
                      generate_random_grid)


class Graph:
    """
    Creates the graph class to manage the grid, start position, and end positions.

    Most variables are private, as updating directly one would put the graph in
    an "unstable" state with many mismatches.
    """
    def __init__(self, rows: int, cols: int):
        self.__rows = rows
        self.__cols = cols
        self.__grid = None
        self.__p_start = None
        self.__p_end = None

    def _state_check(f):
        def inner(self, *args, **kwargs):
            if (self.grid is None):
                raise ValueError("grid isn't set yet, call generate_grid()")
            if (self.p_start is None):
                raise ValueError("p_start isn't set yet, call generate_grid()")
            if (self.p_end is None):
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
    def from_grid(cls, grid, start, end):
        # Initialize from a pre-defined grid, start position, and end position
        rows = len(grid)
        cols = len(grid[0])
        g = cls(rows, cols)
        g.__p_start = start
        g.__p_end = end
        g.__grid = grid
        return g

    def generate_grid(self, grid_type: str = "random", **kwargs) -> None:
        match grid_type:
            case "random":
                s, t, g = generate_random_grid(self.__rows, self.__cols, **kwargs)
            case "random-1":
                s, t, g = generate_grid(self.__rows, self.__cols, **kwargs)
            case "maze":
                s, t, g = generate_maze(self.__rows, self.__cols, **kwargs)
            case "fixed":
                s, t, g = generate_fixed_grid(self.__rows, self.__cols, **kwargs)
            case _:
                s, t, g = generate_grid(self.__rows, self.__cols, **kwargs)

        self.__p_start = s
        self.__p_end = t
        self.__grid = g

    @_state_check
    @_complexity
    def search(self, fct) -> Tuple[List[Tuple[int, int]], object]:
        return fct(self.p_start, self.p_end, self.grid)

    @_state_check
    def search_generator(self, fct) -> Iterator[Tuple[str, any]]:
        return fct(self.p_start, self.p_end, self.grid)

import matplotlib.pyplot as plt

from .types import Grid_t, Position_t


def plot_grid(start: Position_t, target: Position_t, grid: Grid_t, axes: None = None):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(6, 3))
        axes = [axes]

    for ax in axes:
        ax.imshow(grid, cmap="gray", interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.scatter(x=start[1], y=start[0], s=100, c="yellow", marker="o")
        ax.text(
            x=start[1],
            y=start[0],
            s="s",
            c="black",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
        )
        ax.scatter(x=target[1], y=target[0], s=100, c="green", marker="o")
        ax.text(
            x=target[1],
            y=target[0],
            s="f",
            c="white",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
        )


def plot_path(ax, path, **kwargs):

    label = None
    if "label" in kwargs:
        label = kwargs["label"]
        del kwargs["label"]

    ax.set_title(label)
    for i in range(len(path) - 1):
        ax.plot([path[i][1], path[i + 1][1]], [path[i][0], path[i + 1][0]], **kwargs)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pandas import DataFrame, Series

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


def bar_chart_performance(partition: DataFrame | Series, colors: object, **kwargs):

    fig, axes = plt.subplots(nrows=1, ncols=3, **kwargs)
    fig.suptitle("random grids, varying % blockages")

    color_order_keys = []

    for i, label in enumerate(["time", "iterations", "length"]):

        subbed = partition[label]

        if i == 0:
            color_order_keys = subbed.columns

        n_cols = len(subbed.columns)
        n_ind = len(subbed.index)

        x_labels = []
        color = []
        y = []
        x = 0
        xs = []
        for ind in subbed.index:
            x_labels.append(ind)
            for col in subbed.columns:
                y.append(subbed.loc[ind][col])
                color.append(colors[col])
                xs.append(x)
                x += 1
            x += 1

        starting_label = (n_cols - 1) / 2
        xs_labels = [i * (n_cols + 1) + starting_label for i in range(n_ind)]

        axes[i].bar(xs, y, color=color)
        axes[i].set_ylabel(label)
        axes[i].set_xticks(xs_labels, x_labels)

    color_order_values = [colors[c] for c in color_order_keys]

    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in color_order_values]
    fig.legend(custom_lines, color_order_keys)
    return fig

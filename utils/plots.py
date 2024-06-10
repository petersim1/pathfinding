from typing import List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


def bar_chart_performance(
        partition: List[object],
        colormap: object,
        items: List[str],
        key: str,
        **kwargs):

    fig, axes = plt.subplots(nrows=1, ncols=3, **kwargs)

    n_groups = len(partition)
    n_per_group = len(items)

    for i, label in enumerate(["time", "iterations", "length"]):

        x_labels = []
        color = []
        y = []
        err = []
        x = 0
        xs = []

        for d in partition:
            x_labels.append(d[key])
            for k in items:
                y.append(d[k][label]["avg"])
                xs.append(x)
                err.append(d[k][label]["std"])
                color.append(colormap[k])
                x += 1
            x += 1  # create a gap between groupings

        starting_label_i = (n_per_group - 1) / 2
        x_labels_i = [i * (n_per_group + 1) + starting_label_i for i in range(n_groups)]

        axes[i].bar(xs, y, yerr=err, color=color)
        axes[i].set_ylabel(label)
        axes[i].set_xticks(x_labels_i, x_labels)

    color_order_values = [colormap[c] for c in items]

    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in color_order_values]
    fig.legend(custom_lines, items)
    return fig

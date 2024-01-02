"""Draw plots.
"""
from typing import Any
from typing import List
from typing import Tuple

import matplotlib.axes as Axes
import matplotlib.pyplot as plt


def charts(t, ax: Axes):
    return {
        "line": ax.plot,
        "scatter": ax.scatter,
        "bar": ax.bar,
    }[t]


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def draw_chart(
    ys: List[List],
    lbls: List[str],
    y_colors: List[str] = None,
    xs: List[Any] = None,
    fill_between: List = None,
    fill_colors: List[str] = None,
    fill_alpha: List[float] or float = 0.5,
    figsize: Tuple = (24, 3),
    xmode: str = "s",
    # pylint: disable=unused-argument
    ymode: str = "s",
    title: str = None,
    set_xticks: bool = False,
    xticks: List[Any] = None,
    tick_labels: List[Any] = None,
    x_rotation: float = 45,
    boxes: List[List[int]] = None,
    box_colors: List[str] = None,
    box_alphas: List[float] = None,
    ylim: (float, float) = None,
    xlim: (float, float) = None,
    linewidth: List[int] = None,
    types: List[str] or str = "line",
    markersize: float = 1.25,
    save_path: str = None,
    bar_width: float = 0.25,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: Tuple[float] = (1, 0.5),
) -> None:
    """Draw charts.

    Args:
        ys (List[List]): value lists of the y axis.
        lbls (List[str]): legend labels of the value lists.
        y_colors (List[str], optional): colors of the the ys. Defaults to None.
        xs (List[Any], optional): value list of the x axis. \
            If None, idx will be used. Defaults to None.
        fill_between (List, optional): fill the range of \
            `[ys[idx][i] - fill_between[idx][i], ys[idx][i] + fill_between[idx][i]]`.\
                  Defaults to None.
        fill_colors (List[str], optional): colors of the fill range. Defaults to None.
        fill_alpha (List[float] or float, optional): color alphas of the fill range. \
            Defaults to 0.5.
        figsize (Tuple, optional): size of the output figure. Defaults to (24, 3).
        xmode (str, optional): all ys with the same x ('s') or with different xs ('d'). \
            Defaults to "s".
        ymode (str, optional): all ys with the same tick ('s') or with different ticks ('d'). \
            Defaults to "s".
        title (str, optional): figure title. Defaults to None.
        set_xticks (bool, optional): set the x ticks. Defaults to True.
        xticks (List[Any], optional): ticks for the x axis. Defaults to None.
        tick_labels (List[Any], optional): tick labels for the x axis. Defaults to None.
        x_rotation (float, optional): rotation of the x ticks. Defaults to 45.
        boxes (List[List[int]], optional): draw boxes. Defaults to None.
        box_colors (List[str], optional): colors of the boxes to draw. Defaults to None.
        box_alpha (List[float], optional): color alphas of the boxes to draw. Defaults to 0.5.
        ylim (float, optional): maximum value of the y axis. Defaults to None.
        xlim (float, optional): maximum value of the x axis. Defaults to None.
        linewidth (List[int], optional): line width of the line charts to draw. Defaults to None.
        types (List[str] or str, optional): types of charts of the ys. Defaults to "line".
        markersize (float, optional): dot size of the scatter charts. Defaults to 1.25.
        save_path (str, optional): If not None, save image to the path. Defaults to None.
        bar_width (float, optional): width of the bars. Defaults to 0.25.
        legend_loc (str, optional): location of the legend. Defaults to "center left".
        legend_bbox_to_anchor (Tuple[float], optional): bbox_to_anchor of the legend. \
            Defaults to (1, 0.5).

    Raises:
        ValueError: "The arg 'types:'{types} has values not supported."
    """

    plt.clf()
    _, ax = plt.subplots(figsize=figsize)

    # drawing func list
    funcs = []
    optional_args = []
    if isinstance(types, List):
        for t in types:
            funcs.append(charts(t, ax))
            if t == "bar":
                optional_args.append({
                    "width": bar_width,
                })
            else:
                optional_args.append({})
    elif isinstance(types, str):
        for _ in range(len(ys)):
            funcs.append(charts(types, ax))
            if t == "bar":
                optional_args.append({
                    "width": bar_width,
                })
            else:
                optional_args.append({})
    else:
        raise ValueError(f"The arg 'types:'{types} has values not supported.")

    plt.rcParams["lines.markersize"] = markersize
    # ys with the same x
    if xmode == "s":
        for idx, y in enumerate(ys):
            funcs[idx](
                xs,
                y,
                label=lbls[idx],
                color=y_colors[idx] if y_colors is not None and y_colors[idx] is not None else None,
                linewidth=1 if linewidth is None else linewidth[idx],
                **optional_args[idx],
            )

            # fill the range of
            # [ys[idx][i] - fill_between[idx][i], ys[idx][i] + fill_between[idx][i]]
            if fill_between is not None and fill_between[idx] is not None:
                ax.fill_between(
                    xs,
                    [y + fill_between[idx][i] for i, y in enumerate(y)],
                    [y - fill_between[idx][i] for i, y in enumerate(y)],
                    facecolor=fill_colors[idx]
                    if fill_colors is not None and fill_colors[idx] is not None else None,
                    alpha=fill_alpha[idx] if isinstance(fill_alpha, List) and
                    fill_colors[idx] is not None else fill_alpha,
                )

        # set the ticks of the x axis
        if set_xticks:
            ax.set_xticks(
                ticks=range(0, len(xs)) if xticks is None else xticks,
                labels=xs if tick_labels is None else tick_labels,
                rotation=x_rotation,
                ha="right",
            )

    # ys with different xs
    elif xmode == "d":
        for idx, y in enumerate(ys):
            funcs[idx](
                xs[idx],
                y,
                label=lbls[idx],
                color=y_colors[idx] if y_colors is not None and y_colors[idx] is not None else None,
                linewidth=1 if linewidth is None else linewidth[idx],
                **optional_args[idx],
            )

            # fill the range of
            # [ys[idx][i] - fill_between[idx][i], ys[idx][i] + fill_between[idx][i]]
            if fill_between is not None and fill_between[idx] is not None:
                ax.fill_between(
                    xs,
                    [y + fill_between[idx][i] for i, y in enumerate(y)],
                    [y - fill_between[idx][i] for i, y in enumerate(y)],
                    facecolor=fill_colors[idx]
                    if fill_colors is not None and fill_colors[idx] is not None else None,
                    alpha=fill_alpha[idx] if isinstance(fill_alpha, List) and
                    fill_colors[idx] is not None else fill_alpha,
                )

        # set the ticks of the x axis
        if set_xticks:
            ax.set_xticks(
                ticks=range(0, len(xs[0])) if xticks is None else xticks,
                labels=xs[0] if tick_labels is None else tick_labels,
                rotation=x_rotation,
                ha="right",
            )

    # set xs with numbers when no xs provided
    else:
        for idx, y in enumerate(ys):
            xs = list(range(len(y)))
            funcs[idx](
                xs,
                y,
                label=lbls[idx],
                color=y_colors[idx] if y_colors is not None and y_colors[idx] is not None else None,
                linewidth=1 if linewidth is None else linewidth[idx],
                **optional_args[idx],
            )

            # fill the range of
            # [ys[idx][i] - fill_between[idx][i], ys[idx][i] + fill_between[idx][i]]
            if fill_between is not None and fill_between[idx] is not None:
                ax.fill_between(
                    xs,
                    [y + fill_between[idx][i] for i, y in enumerate(y)],
                    [y - fill_between[idx][i] for i, y in enumerate(y)],
                    facecolor=fill_colors[idx]
                    if fill_colors is not None and fill_colors[idx] is not None else None,
                    alpha=fill_alpha[idx] if isinstance(fill_alpha, List) and
                    fill_colors[idx] is not None else fill_alpha,
                )

        # set the ticks of the x axis
        if set_xticks:
            ax.set_xticks(
                ticks=range(0, len(xs)) if xticks is None else xticks,
                labels=list(range(len(ys[0]))) if tick_labels is None else tick_labels,
                rotation=45,
                ha="right",
            )

    # draw a box on the figure
    if boxes is not None:
        for idx, box in enumerate(boxes):
            ax.add_patch(
                plt.Rectangle(
                    box[0],
                    box[1],
                    box[2],
                    transform=ax.transAxes,
                    color=box_colors[idx] if box_colors is not None else "darkgrey",
                    alpha=box_alphas[idx] if box_alphas is not None else 0.5,
                )
            )

    ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)
    if title:
        plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

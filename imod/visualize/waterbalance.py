import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _draw_bars(ax, x, df, labels, barwidth, colors):
    ndates, _ = df.shape
    bottoms = np.hstack([np.zeros((ndates, 1)), df.cumsum(axis=1).values]).T[:-1]
    heights = df.values.T
    if colors is None:
        for label, bottom, height in zip(labels, bottoms, heights):
            ax.bar(
                x,
                bottom=bottom,
                height=height,
                width=barwidth,
                edgecolor="k",
                label=label,
            )
    else:
        for label, bottom, height, color in zip(labels, bottoms, heights, colors):
            ax.bar(
                x,
                bottom=bottom,
                height=height,
                width=barwidth,
                edgecolor="k",
                label=label,
                color=color,
            )


def waterbalance_barchart(
    df,
    inflows,
    outflows,
    datecolumn=None,
    format="%Y-%m-%d",
    ax=None,
    unit=None,
    colors=None,
):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the water balance data.
    inflows : listlike of str
    outflows : listlike of str
    datecolumn : str, optional
    format : str, optional,
    ax : matplotlib.Axes, optional
    unit : str, optional
    colors : listlike of strings or tuples

    Returns
    -------
    ax : matplotlib.Axes

    Examples
    --------

    >>> fig, ax = plt.subplots()
    >>> imod.visualize.waterbalance_barchart(
    >>>    ax=ax,
    >>>    df=df,
    >>>    inflows=["Rainfall", "River upstream"],
    >>>    outflows=["Evapotranspiration", "Discharge to Sea"],
    >>>    datecolumn="Time",
    >>>    format="%Y-%m-%d",
    >>>    unit="m3/d",
    >>>    colors=["#ca0020", "#f4a582", "#92c5de", "#0571b0"],
    >>>    )
    >>> fig.savefig("Waterbalance.png", dpi=300, bbox_inches="tight")

    """
    # Do some checks
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df should be a pandas.DataFrame")
    if datecolumn is not None:
        if datecolumn not in df.columns:
            raise ValueError(f"datecolumn {datecolumn} not in df")
    for column in itertools.chain(inflows, outflows):
        if column not in df:
            raise ValueError(f"{column} not in df")
    if colors is not None:
        ncolors = len(colors)
        nflows = len(inflows + outflows)
        if ncolors < nflows:
            raise ValueError(
                f"Not enough colors: Number of flows is {nflows}, while number of colors is {ncolors}"
            )
    # Deal with colors, takes both dict and list
    if isinstance(colors, dict):
        incolors = [colors[k] for k in inflows]
        outcolors = [colors[k] for k in outflows]
    elif isinstance(colors, (tuple, list)):
        incolors = colors[: len(inflows)]
        outcolors = colors[len(inflows) :]
    else:
        incolors = None
        outcolors = None

    # Determine x position
    ndates, _ = df.shape
    barwidth = 1.0
    r1 = np.arange(0.0, ndates * barwidth * 3, barwidth * 3)
    r2 = np.array([x + barwidth for x in r1])
    r_between = 0.5 * (r1 + r2)

    # Grab ax if not provided directly
    if ax is None:
        ax = plt.gca()

    # Draw inflows
    _draw_bars(
        ax=ax, x=r1, df=df[inflows], labels=inflows, barwidth=barwidth, colors=incolors
    )
    # Draw outflows
    _draw_bars(
        ax=ax,
        x=r2,
        df=df[outflows],
        labels=outflows,
        barwidth=barwidth,
        colors=outcolors,
    )

    # Place xticks
    xticks_location = list(itertools.chain(*zip(r1, r_between, r2)))
    # Collect the labels, and format them as desired
    # TODO: might not work for all dateformats?
    xticks_labels = []
    if datecolumn is None:
        dates = df.index
    else:
        dates = df[datecolumn]
    for date in dates:
        # Place the date labels two lines (two \n) below the minor labels ("in", "out")
        xticks_labels.extend(["in", f"\n\n{date.strftime(format)}", "out"])

    # Adjust the ticks. Lengthen the major ticks, so they extend down to the dates
    ax.tick_params(axis="x", which="major", bottom=False, top=False, labelbottom=True)
    ax.tick_params(
        axis="x",
        which="minor",
        bottom=True,
        top=False,
        labelbottom=False,
        length=barwidth * 45,
    )
    ax.xaxis.set_ticks(xticks_location)
    ax.xaxis.set_ticklabels(xticks_labels)
    xticks_location_minor = r1[1:] - barwidth
    ax.xaxis.set_ticks(xticks_location_minor, minor=True)

    # Create a legend on the right side of the chart
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.03, 1.0),
        ncol=2,
        borderaxespad=0,
        frameon=True,
    )

    # Set a unit on the y-axis
    if unit is not None:
        ax.yaxis.set_label(unit)

    return ax

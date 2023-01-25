import matplotlib
import numpy as np


def _cmapnorm_from_colorslevels(colors, levels):
    """
    Create ListedColormap and BoundaryNorm from colors (either a list of colors
    or a named colormap) and a list of levels. Number of levels must be at least
    two, and number of colors therefore least three.

    In the case of a list of colors, the resulting colorbar looks like:
    < color 0 | color 1 | color 2 ... n-1 | color n >
              ^         ^                 ^
           level 0   level 1           level n-1

    Parameters
    ----------

    colors : list of str, list of RGBA/RGBA tuples, colormap name (str), or matplotlib.colors.Colormap
        If list, it should be a Matplotlib acceptable list of colors. Length N.
        Accepts both tuples of (R, G, B) and hexidecimal (e.g. `#7ec0ee`).
                If str, use an existing Matplotlib colormap. This function will
                autmatically add distinctive colors for pixels lower or high than the given
                min respectively max level.
                If LinearSegmentedColormap, you can use something like
                `matplotlib.cm.get_cmap('jet')` as input. This function will not alter
                the colormap, so add under- and over-colors yourself.

        Looking for good colormaps? Try: http://colorbrewer2.org/
        Choose a colormap, and use the HEX JS array.
    levels : listlike of floats or integers
        Boundaries between the legend colors/classes. Length: N - 1.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
    norm : matplotlib.colors.BoundaryNorm
    """
    # check number of levels
    if len(levels) < 2:
        raise ValueError(f"Number of levels {levels} should exceed 1.")

    # check monotonic increasing levels
    if not (np.diff(levels) > 0).all():
        raise ValueError(f"Levels {levels} are not monotonic increasing.")

    if isinstance(colors, matplotlib.colors.Colormap):
        # use given cmap
        cmap = colors
    else:
        nlevels = len(levels)
        if isinstance(colors, str):
            # Use given cmap, but fix the under and over colors
            # The colormap (probably) does not have a nice under and over color.
            # So we cant use `cmap = matplotlib.cm.get_cmap(colors)`
            cmap = matplotlib.cm.get_cmap(colors)
            colors = cmap(np.linspace(0, 1, nlevels + 1))

        # Validate number of colors vs number of levels
        ncolors = len(colors)
        if not nlevels == ncolors - 1:
            raise ValueError(
                f"Incorrect number of levels. Number of colors is {ncolors},"
                f" expected {ncolors - 1} levels, got {nlevels} levels instead."
            )
        # Create cmap from given list of colors
        cmap = matplotlib.colors.ListedColormap(colors[1:-1])
        cmap.set_under(
            colors[0]
        )  # this is the color for values smaller than raster.min()
        cmap.set_over(
            colors[-1]
        )  # this is the color for values larger than raster.max()
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
    return cmap, norm

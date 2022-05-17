import matplotlib
import pytest

import imod


def test_cmapnorm_from_colorslevels_cmap():
    levels = [0.0, 1.0]
    cmap, norm = imod.visualize.common._cmapnorm_from_colorslevels("viridis", levels)

    assert isinstance(cmap, matplotlib.colors.ListedColormap)
    assert isinstance(norm, matplotlib.colors.BoundaryNorm)
    assert norm.vmin == levels[0]
    assert norm.vmax == levels[1]
    assert len(cmap.colors) == len(levels) - 1


def test_cmapnorm_from_colorslevels_colorslevels():
    colors = ["#ffffff", "#808080", "#000000"]
    levels = [0.25, 0.75]
    cmap, norm = imod.visualize.common._cmapnorm_from_colorslevels(colors, levels)

    assert isinstance(cmap, matplotlib.colors.ListedColormap)
    assert isinstance(norm, matplotlib.colors.BoundaryNorm)
    assert norm.vmin == levels[0]
    assert norm.vmax == levels[1]
    assert len(cmap.colors) == len(levels) - 1
    assert cmap._rgba_over == (0.0, 0.0, 0.0, 1.0)
    assert cmap._rgba_under == (1.0, 1.0, 1.0, 1.0)


def test_cmapnorm_from_colorslevels_colornameslevels():
    colors = ["w", "gray", "black"]
    levels = [0.25, 0.75]
    cmap, norm = imod.visualize.common._cmapnorm_from_colorslevels(colors, levels)

    assert isinstance(cmap, matplotlib.colors.ListedColormap)
    assert isinstance(norm, matplotlib.colors.BoundaryNorm)
    assert norm.vmin == levels[0]
    assert norm.vmax == levels[1]
    assert len(cmap.colors) == len(levels) - 1
    assert cmap._rgba_over == (0.0, 0.0, 0.0, 1.0)
    assert cmap._rgba_under == (1.0, 1.0, 1.0, 1.0)


def test_cmapnorm_from_colorslevels_justonelevel():
    levels = [0.5]
    colors = "viridis"
    with pytest.raises(ValueError, match="Number of levels"):
        cmap, norm = imod.visualize.common._cmapnorm_from_colorslevels(colors, levels)


def test_cmapnorm_from_colorslevels_unorderedlevels():
    levels = [10.0, -1, 5.0]
    colors = "viridis"
    with pytest.raises(ValueError, match="monotonic increasing"):
        cmap, norm = imod.visualize.common._cmapnorm_from_colorslevels(colors, levels)


def test_cmapnorm_from_colorslevels_nooflevels():
    levels = [-1, 0, 5.0]
    colors = ["k", "b", "w"]
    with pytest.raises(ValueError, match="Incorrect number of levels"):
        cmap, norm = imod.visualize.common._cmapnorm_from_colorslevels(colors, levels)

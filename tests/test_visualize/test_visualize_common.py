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
    colors = ["#ffffff", "#000000"]
    levels = [0.5]
    cmap, norm = imod.visualize.common._cmapnorm_from_colorslevels(colors, levels)

    assert isinstance(cmap, matplotlib.colors.ListedColormap)
    assert isinstance(norm, matplotlib.colors.BoundaryNorm)
    assert norm.vmin == levels[0]
    assert norm.vmax == levels[0]
    assert len(cmap.colors) == len(levels) - 1
    assert cmap._rgba_over == (0.0, 0.0, 0.0, 1.0)
    assert cmap._rgba_under == (1.0, 1.0, 1.0, 1.0)


def test_cmapnorm_from_colorslevels_colornameslevels():
    colors = ["w", "black"]
    levels = [0.5]
    cmap, norm = imod.visualize.common._cmapnorm_from_colorslevels(colors, levels)

    assert isinstance(cmap, matplotlib.colors.ListedColormap)
    assert isinstance(norm, matplotlib.colors.BoundaryNorm)
    assert norm.vmin == levels[0]
    assert norm.vmax == levels[0]
    assert len(cmap.colors) == len(levels) - 1
    assert cmap._rgba_over == (0.0, 0.0, 0.0, 1.0)
    assert cmap._rgba_under == (1.0, 1.0, 1.0, 1.0)


def test_cmapnorm_from_colorslevels_unorderedlevels():
    levels = [10.0, -1, 5.0]
    colors = "viridis"
    with pytest.raises(ValueError):
        cmap, norm = imod.visualize.common._cmapnorm_from_colorslevels(colors, levels)


def test_cmapnorm_from_colorslevels_nooflevels():
    levels = [-1, 0, 5.0]
    colors = ["k", "b", "w"]
    with pytest.raises(ValueError):
        cmap, norm = imod.visualize.common._cmapnorm_from_colorslevels(colors, levels)

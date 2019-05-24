import os

import numpy as np
import pytest
import xarray as xr
import matplotlib.pyplot as plt

import imod


@pytest.fixture(scope="module")
def example_legend(request):
    legend_content = """
   17    1    1    1    1    1    1    1
 UPPER BND LOWER BND      IRED    IGREEN     IBLUE     DOMAIN
 200.0      10.00             75          0          0 "> 10 m"
 10.00      6.000            115          0          0 "6.0 - 10.0 m"
 6.000      4.000            166          0          0 "4.0 - 6.0 m"
 4.000      3.000            191          0          0 "3.0 - 4.0 m"
 3.000      2.500            217          0          0 "2.5 - 3.0 m"
 2.500      2.000            237          0          0 "2.0 - 2.5 m"
 2.000      1.800            255         85          0 "1.8 - 2.0 m"
 1.800      1.600            254        140          0 "1.6 - 1.8 m"
 1.600      1.400            254        191         10 "1.4 - 1.6 m"
 1.400      1.200            254        221         51 "1.2 - 1.4 m"
 1.200      1.000            254        255        115 "1.0 - 1.2 m"
 1.000     0.8000            255        255        190 "0.8 - 1.0 m"
0.8000     0.6000            136        255         72 "0.6 - 0.8 m"
0.6000     0.4000              0        168          0 "0.4 - 0.6 m"
0.4000     0.2000              0        206        206 "0.2 - 0.4 m"
0.2000      0.000             17         17        255 "0 - 0.2 m"
 0.000     -200.0              0          0        125 "< 0 m"
    """

    with open("example_legend.leg", "w") as f:
        f.write(legend_content)

    def teardown():
        try:
            os.remove("example_legend.leg")
        except FileNotFoundError:
            pass

    request.addfinalizer(teardown)


def test_read_legend(example_legend):

    colors, levels = imod.visualize.spatial.read_imod_legend(path="example_legend.leg")

    assert colors == [
        "#00007d",
        "#1111ff",
        "#00cece",
        "#00a800",
        "#88ff48",
        "#ffffbe",
        "#feff73",
        "#fedd33",
        "#febf0a",
        "#fe8c00",
        "#ff5500",
        "#ed0000",
        "#d90000",
        "#bf0000",
        "#a60000",
        "#730000",
        "#4b0000",
    ]
    assert levels == [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
        1.6,
        1.8,
        2.0,
        2.5,
        3.0,
        4.0,
        6.0,
        10.0,
    ]


def test_plot_map():
    fig, ax = imod.visualize.spatial.plot_map(
        raster=xr.DataArray(np.random.randn(2, 3), dims=("x", "y")),
        legend_colors=["#ff0000", "#00ff00", "#0000ff"],
        legend_levels=[0.2, 0.8],
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

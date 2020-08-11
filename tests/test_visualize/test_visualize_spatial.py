import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import imod


@pytest.fixture(scope="function")
def write_legend():
    def _write_legend(delim, path):
        legend_content = (
            "17{delim}1{delim}1{delim}1{delim}1{delim}1{delim}1{delim}1\n"
            "UPPERBND{delim}LOWERBND{delim}IRED{delim}IGREEN{delim}IBLUE{delim}DOMAIN\n"
            '200.0{delim}10.00{delim}75{delim}0{delim}0{delim}"> 10 m"\n'
            '10.00{delim}6.000{delim}115{delim}0{delim}0{delim}"6.0 - 10.0 m"\n'
            '6.000{delim}4.000{delim}166{delim}0{delim}0{delim}"4.0 - 6.0 m"\n'
            '4.000{delim}3.000{delim}191{delim}0{delim}0{delim}"3.0 - 4.0 m"\n'
            '3.000{delim}2.500{delim}217{delim}0{delim}0{delim}"2.5 - 3.0 m"\n'
            '2.500{delim}2.000{delim}237{delim}0{delim}0{delim}"2.0 - 2.5 m"\n'
            '2.000{delim}1.800{delim}255{delim}85{delim}0{delim}"1.8 - 2.0 m"\n'
            '1.800{delim}1.600{delim}254{delim}140{delim}0{delim}"1.6 - 1.8 m"\n'
            '1.600{delim}1.400{delim}254{delim}191{delim}10{delim}"1.4 - 1.6 m"\n'
            '1.400{delim}1.200{delim}254{delim}221{delim}51{delim}"1.2 - 1.4 m"\n'
            '1.200{delim}1.000{delim}254{delim}255{delim}115{delim}"1.0 - 1.2 m"\n'
            '1.000{delim}0.800{delim}255{delim}255{delim}190{delim}"0.8 - 1.0 m"\n'
            '0.800{delim}0.600{delim}136{delim}255{delim}72{delim}"0.6 - 0.8 m"\n'
            '0.600{delim}0.400{delim}0{delim}168{delim}0{delim}"0.4 - 0.6 m"\n'
            '0.400{delim}0.200{delim}0{delim}206{delim}206{delim}"0.2 - 0.4 m"\n'
            '0.200{delim}0.000{delim}17{delim}17{delim}255{delim}"0 - 0.2 m"\n'
            '0.000{delim}200.0{delim}0{delim}0{delim}125{delim}"< 0 m"\n'
        )

        with open(path, "w") as f:
            f.write(legend_content.format(delim=delim))

    return _write_legend


@pytest.mark.parametrize("delim", [",", " ", "\t"])
def test_read_legend(write_legend, delim, tmp_path):
    leg_path = tmp_path / "example_legend.leg"
    write_legend(delim=delim, path=leg_path)
    colors, levels = imod.visualize.spatial.read_imod_legend(path=leg_path)

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
        raster=xr.DataArray(
            np.random.randn(2, 3),
            coords={"x": [0.5, 1.5], "y": [1.5, 0.5, -0.5]},
            dims=("x", "y"),
        ),
        colors=["#ff0000", "#00ff00", "#0000ff"],
        levels=[0.2, 0.8],
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_map_basemap():
    fig, ax = imod.visualize.spatial.plot_map(
        raster=xr.DataArray(
            np.random.randn(2, 3),
            coords={"x": [0.5, 1.5], "y": [1.5, 0.5, -0.5]},
            dims=("x", "y"),
        ),
        colors=["#ff0000", "#00ff00", "#0000ff"],
        levels=[0.2, 0.8],
        basemap=True,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

import os

import numpy as np
import pytest
import xarray as xr
import matplotlib.pyplot as plt

import imod


@pytest.fixture(scope="module")
def example_legend(request):
    legend_content = """

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

    colors, levels = imod.visualize.spatial.read_imod_legend("example_legend.leg")

    assert colors == ["#0000000", "#000"]
    assert levels == [1.0, 2.0]


def test_plot_map():
    figure = imod.visualize.spatial.plot_map()

    assert isinstance(figure, plt.figure)

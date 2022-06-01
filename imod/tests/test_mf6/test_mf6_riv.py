import pathlib
import textwrap

import numpy as np
import pytest

import imod


def test_render():
    riv = imod.mf6.River(stage=1.0, conductance=10.0, bottom_elevation=-1.0)
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = riv.render(directory, "river", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 1
        end dimensions

        begin period 1
          open/close mymodel/river/riv.bin (binary)
        end period
        """
    )
    assert actual == expected


@pytest.mark.usefixtures("concentration_fc")
def test_render_concentration(concentration_fc):
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    riv = imod.mf6.River(
        stage=1.0,
        conductance=10.0,
        bottom_elevation=-1.0,
        concentration=concentration_fc.sel(time= np.datetime64("2000-01-01")).reset_coords(drop=True),
        concentration_boundary_type="AUX",
    )
    actual = riv.render(directory, "riv", globaltimes, False)

    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity temperature
        end options

        begin dimensions
          maxbound 1
        end dimensions

        begin period 1
          open/close mymodel/riv/riv.dat
        end period
        """
    )
    assert actual == expected


def test_wrong_dtype():
    with pytest.raises(TypeError):
        imod.mf6.River(stage=1, conductance=10.0, bottom_elevation=-1.0)

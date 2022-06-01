import pathlib
import tempfile
import textwrap

import numpy as np
import pytest
import xarray as xr

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
        concentration=concentration_fc.sel(
            time=np.datetime64("2000-01-01")
        ).reset_coords(drop=True),
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


pytest.mark.usefixtures("concentration_fc")


def test_write_concentration_period_data(concentration_fc):
    globaltimes = [np.datetime64("2000-01-01")]
    concentration_fc[:] = 2
    stage = xr.full_like(concentration_fc.sel({"species": "salinity"}), 13)
    conductance = xr.full_like(stage, 13)
    bottom_elevation = xr.full_like(stage, 13)
    riv = imod.mf6.River(
        stage=stage,
        conductance=conductance,
        bottom_elevation=bottom_elevation,
        concentration=concentration_fc,
        concentration_boundary_type="AUX",
    )
    with tempfile.TemporaryDirectory() as output_dir:
        riv.write(output_dir, "riv", globaltimes, False)
        with open(output_dir + "\\riv\\riv-0.dat", "r") as f:
            data = f.read()
            assert (
                data.count("2") == 1755
            )  # the number 2 is in the concentration data, and in the cell indices.

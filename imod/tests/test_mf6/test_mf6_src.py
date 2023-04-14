import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod


@pytest.fixture(scope="function")
def src_rate():
    nlayer = 2
    nrow = 3
    ncol = 4
    data_rate = np.ones((nlayer, nrow, ncol))
    data_rate[0, 0, 0] = np.nan
    coords = {"layer": [1, 2], "y": [3.5, 2.5, 1.5], "x": [0.5, 1.5, 2.5, 3.5]}
    dims = ("layer", "y", "x")
    rate = xr.DataArray(data_rate, coords, dims)
    return rate


def test_render(src_rate):
    src = imod.mf6.MassSourceLoading(
        rate=src_rate,
        print_input=False,
        print_flows=False,
        save_flows=False,
    )
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")
    actual = src.render(directory, "src", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 23
        end dimensions

        begin period 1
          open/close mymodel/src/src.bin (binary)
        end period
        """
    )
    assert actual == expected


def test_render_mass_source_transient(src_rate):
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-02-01",
            "2000-03-01",
        ],
        dtype="datetime64[ns]",
    )
    rate = xr.concat(
        [
            src_rate.assign_coords(time=globaltimes[0]),
            src_rate.assign_coords(time=globaltimes[1]),
        ],
        dim="time",
    )

    src = imod.mf6.MassSourceLoading(
        rate=rate,
        print_input=False,
        print_flows=False,
        save_flows=False,
    )
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-02-01",
            "2000-03-01",
        ],
        dtype="datetime64[ns]",
    )
    actual = src.render(directory, "src", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 23
        end dimensions

        begin period 1
          open/close mymodel/src/src-0.bin (binary)
        end period
        begin period 2
          open/close mymodel/src/src-1.bin (binary)
        end period
        """
    )
    assert actual == expected

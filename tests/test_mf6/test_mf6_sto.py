import pathlib
import textwrap

import numpy as np
import xarray as xr

import imod


def test_render():
    nlay = 3
    nrow = 15
    ncol = 15
    shape = (nlay, nrow, ncol)

    dx = 5000.0
    dy = -5000.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.array([1, 2, 3])
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    # Discretization data
    idomain = xr.DataArray(np.ones(shape), coords=coords, dims=dims)

    # for better coverage, use full (conv) layered (sy) and constant (ss)
    conv = xr.full_like(idomain, 0, dtype=int)
    sy_layered = xr.DataArray([0.16, 0.15, 0.14], {"layer": layer}, ("layer",))
    sto = imod.mf6.Storage(
        specific_storage=0.0003,
        specific_yield=sy_layered,
        transient=True,
        convertible=conv,
    )

    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = sto.render(directory, "sto", globaltimes)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin griddata
          iconvert
            open/close mymodel/sto/iconvert.bin (binary)
          ss
            constant 0.0003
          sy layered
            constant 0.16
            constant 0.15
            constant 0.14
        end griddata

        begin period 1
          transient
        end period
        """
    )
    print(actual)
    assert actual == expected

    # again but starting with two steady-state periods, followed by a transient stress period
    times = [np.datetime64("2000-01-01"), np.datetime64("2000-01-03")]
    transient = xr.DataArray([False, True], {"time": times}, ("time",))
    sto = imod.mf6.Storage(
        specific_storage=0.0003,
        specific_yield=sy_layered,
        transient=transient,
        convertible=conv,
    )
    globaltimes = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-02"),
        np.datetime64("2000-01-03"),
    ]
    actual = sto.render(directory, "sto", globaltimes)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin griddata
          iconvert
            open/close mymodel/sto/iconvert.bin (binary)
          ss
            constant 0.0003
          sy layered
            constant 0.16
            constant 0.15
            constant 0.14
        end griddata

        begin period 1
          steady-state
        end period
        begin period 3
          transient
        end period
        """
    )
    assert actual == expected

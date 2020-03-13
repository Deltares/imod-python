import numpy as np
import xarray as xr

import imod


def test_convert_pointwaterhead_freshwaterhead_scalar():
    # fresh water
    assert (
        round(imod.evaluate.convert_pointwaterhead_freshwaterhead(4.0, 1000.0, 1.0), 5)
        == 4.0
    )

    # saline
    assert (
        round(imod.evaluate.convert_pointwaterhead_freshwaterhead(4.0, 1025.0, 1.0), 5)
        == 4.075
    )


def test_convert_pointwaterhead_freshwaterhead_da():
    data = np.ones((3, 2, 1))
    coords = {"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    dims = ("layer", "y", "x")

    da = xr.DataArray(data, coords, dims)
    pwh = xr.full_like(da, 4.0)
    dens = xr.full_like(da, 1025.0)
    z = xr.full_like(da, 1.0)
    fwh = xr.full_like(da, 4.075)
    fwh2 = imod.evaluate.convert_pointwaterhead_freshwaterhead(pwh, dens, z)
    assert fwh.equals(fwh2.round(5))

import os
import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod

exdir = pathlib.Path("d:/repo/imod/mf6-dist/examples/ex02-tidal-sprint")

# from ex02-tidal
xorigin = 0.0
yorigin = 0.0
angrot = 0.0
nlay = 3
nrow = 15
ncol = 10

dx = 500.0
dy = -500.0
top = 50.0

# setup ibound
idomain = xr.DataArray(
    data=np.full((nlay, nrow, ncol), 1.0),
    coords={
        "y": np.arange(yorigin + 0.5 * dy, yorigin + nrow * dy, dy),
        # "x": np.arange(xorigin + 0.5 * dx, xorigin + ncol * dx, dx),
        # test nonequidistant
        "x": np.cumsum(np.arange(ncol)),
        "dx": ("x", np.arange(ncol)),
        "layer": np.arange(nlay, dtype=np.int),
    },
    dims=("layer", "y", "x"),
)

bottom = xr.DataArray(
    data=np.array([5.0, -10.0, -100.0]),
    coords={"layer": np.arange(nlay, dtype=np.int)},
    dims=("layer"),
)

dis = imod.mf6.StructuredDiscretization(idomain=idomain, top=top, bottom=bottom)

print(dis.render("mymodel", "mydis"))

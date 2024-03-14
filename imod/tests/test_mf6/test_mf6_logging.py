import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod
from imod.mf6.utilities.package import get_repeat_stress
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError
from imod.logging import LoggerType, LogLevel

@pytest.fixture(scope="function")
def drainage():
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    elevation = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )
    conductance = elevation.copy()

    drn = dict(elevation=elevation, conductance=conductance)
    return drn

def test_write(drainage, tmp_path):

    imod.logging.configure(LoggerType.PYTHON, log_level = LogLevel.DEBUG  ,add_default_file_handler=True, add_default_stream_handler = True)

    drn = imod.mf6.Drainage(**drainage)
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=True)
    drn.write("mydrn", [1], write_context)


import numpy as np
import pytest
import xarray as xr

import imod
from imod.logging import LoggerType, LogLevel
from imod.mf6.write_context import WriteContext

from pathlib import Path

import os
from io import StringIO
import sys

out = StringIO()
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

def test_write_package_is_logged(drainage, tmp_path):
    # arrange
    logfile_path  = tmp_path/'logfile.txt'
    drn = imod.mf6.Drainage(**drainage)
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=True)

    # act
    with open(logfile_path, 'w') as sys.stdout:
        imod.logging.configure(LoggerType.PYTHON, log_level = LogLevel.DEBUG  ,add_default_file_handler=False, add_default_stream_handler = True)
        drn.write("mydrn", [1], write_context)
   
    # assert
    with open(logfile_path, "r") as log_file:
        unknown = log_file.read()

        assert "beginning execution of imod.mf6.package.write for object <class 'imod.mf6.drn.Drainage'>" in unknown
        assert "finished execution of imod.mf6.package.write  for object <class 'imod.mf6.drn.Drainage'>" in unknown


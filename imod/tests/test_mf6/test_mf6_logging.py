import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import imod
from imod.logging import LoggerType, LogLevel
from imod.mf6.write_context import WriteContext

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
    logfile_path = tmp_path / "logfile.txt"

    # act
    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )
        drn = imod.mf6.Drainage(**drainage)
        write_context = WriteContext(simulation_directory=tmp_path, use_binary=True)
        drn.write("mydrn", [1], write_context)

    # assert
    with open(logfile_path, "r") as log_file:
        log = log_file.read()
        assert "Initializing the Drainage package..." in log
        assert "Successfully initialized the Drainage..." in log
        assert (
            "Beginning execution of imod.mf6.package.write for object Drainage..."
            in log
        )
        assert (
            "Finished execution of imod.mf6.package.write  for object Drainage..."
            in log
        )


def test_write_model_is_logged(
    flow_transport_simulation: imod.mf6.Modflow6Simulation, tmp_path: Path
):
    # arrange
    logfile_path = tmp_path / "logfile.txt"
    transport_model = flow_transport_simulation["tpt_c"]
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=True)
    globaltimes = np.array(
        [
            "2000-01-01",
        ],
        dtype="datetime64[ns]",
    )
    # act
    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )
        transport_model.write("model.txt", globaltimes, True, write_context)

    # assert
    with open(logfile_path, "r") as log_file:
        log = log_file.read()

        assert (
            "Beginning execution of imod.mf6.model.write for object GroundwaterTransportModel"
            in log
        )
        assert (
            "Finished execution of imod.mf6.model.write  for object GroundwaterTransportModel"
            in log
        )


def test_write_simulation_is_logged(
    flow_transport_simulation: imod.mf6.Modflow6Simulation, tmp_path: Path
):
    # arrange
    logfile_path = tmp_path / "logfile.txt"

    # act
    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )
        flow_transport_simulation.write(tmp_path, True, True, True)

    # assert
    with open(logfile_path, "r") as log_file:
        log = log_file.read()

        assert (
            "Beginning execution of imod.mf6.simulation.write for object Modflow6Simulation"
            in log
        )
        assert (
            "Finished execution of imod.mf6.simulation.write  for object Modflow6Simulation"
            in log
        )

import re
import sys
from io import StringIO
from pathlib import Path
from time import sleep

import numpy as np
import pytest
import xarray as xr

import imod
from imod.logging import LoggerType, LogLevel, standard_log_decorator
from imod.mf6.validation_settings import ValidationSettings
from imod.mf6.write_context import WriteContext

out = StringIO()
simple_real_number_regexp = (
    r"[0-9]*\.?[0-9]*"  # regexp for a real number without a sign and without exponents
)


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

    drn = {"elevation": elevation, "conductance": conductance}
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
        drn._write("mydrn", [1], write_context)

    # assert
    with open(logfile_path, "r") as log_file:
        log = log_file.read()
        assert "Initializing the Drainage package..." in log
        assert (
            re.search(
                f"Successfully initialized the Drainage in {simple_real_number_regexp} seconds...",
                log,
            )
            is not None
        )
        assert (
            "Beginning execution of imod.mf6.package._write for object Drainage..."
            in log
        )
        assert (
            re.search(
                f"Finished execution of imod.mf6.package._write  for object Drainage in {simple_real_number_regexp} seconds...",
                log,
            )
            is not None
        )


def test_write_model_is_logged(
    flow_transport_simulation: imod.mf6.Modflow6Simulation, tmp_path: Path
):
    # arrange
    logfile_path = tmp_path / "logfile.txt"
    transport_model = flow_transport_simulation["tpt_c"]
    validation_context = ValidationSettings()
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
        transport_model._write(
            "model.txt", globaltimes, write_context, validation_context
        )

    # assert
    with open(logfile_path, "r") as log_file:
        log = log_file.read()

        assert (
            "Beginning execution of imod.mf6.model._write for object GroundwaterTransportModel"
            in log
        )
        assert (
            "Finished execution of imod.mf6.model._write  for object GroundwaterTransportModel"
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


def test_runtime_is_logged(drainage, tmp_path):
    # arrange
    @standard_log_decorator()
    def sleep_half_a_second(_):
        sleep(0.5)

    logfile_path = tmp_path / "logfile.txt"

    # act
    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )

        sleep_half_a_second(
            drainage
        )  # the logger needs an imod-python object as the first function argument

    # assert
    with open(logfile_path, "r") as log_file:
        log = log_file.read()

        assert re.search("in [0-9]*.[0-9]* seconds", log) is not None

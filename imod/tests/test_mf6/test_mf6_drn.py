import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod
from imod.schemata import ValidationError


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
    drn = imod.mf6.Drainage(**drainage)
    drn.write(tmp_path, "mydrn", [1], True)
    dir_for_render = tmp_path.stem
    block_expected = textwrap.dedent(
        f"""\
        begin options
        end options

        begin dimensions
          maxbound 75
        end dimensions

        begin period 1
          open/close {dir_for_render}/mydrn/drn.bin (binary)
        end period
        """
    )

    with open(tmp_path / "mydrn.drn") as f:
        block = f.read()

    assert block == block_expected


def test_wrong_dtype(drainage):
    drainage["elevation"] = drainage["elevation"].astype(np.int32)

    with pytest.raises(ValidationError):
        imod.mf6.Drainage(**drainage)


def test_check_conductance_zero(drainage):
    drainage["conductance"] = drainage["conductance"] * 0.0

    idomain = drainage["elevation"].astype(np.int16)
    top = 1.0
    bottom = top - idomain.coords["layer"]

    dis = imod.mf6.StructuredDiscretization(top=1.0, bottom=bottom, idomain=idomain)

    drn = imod.mf6.Drainage(**drainage)

    errors = drn._validate(drn._write_schemata, **dis.dataset)

    assert len(errors) == 1

    for var, error in errors.items():
        assert var == "conductance"


def test_discontinuous_layer(drainage):
    drn = imod.mf6.Drainage(**drainage)
    drn["layer"] = [1, 3, 5]
    bin_ds = drn[list(drn._period_data)]
    layer = bin_ds["layer"].values
    arrdict = drn._ds_to_arrdict(bin_ds)
    sparse_data = drn.to_sparse(arrdict, layer)
    assert np.array_equal(np.unique(sparse_data["layer"]), [1, 3, 5])


def test_3d_singelayer():
    # Introduced because of Issue #224
    layer = [1]
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    elevation = xr.DataArray(
        np.full((1, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )
    conductance = elevation.copy()
    drn = imod.mf6.Drainage(elevation=elevation, conductance=conductance)

    bin_ds = drn[list(drn._period_data)]
    layer = bin_ds["layer"].values
    arrdict = drn._ds_to_arrdict(bin_ds)
    sparse_data = drn.to_sparse(arrdict, layer)
    assert isinstance(sparse_data, np.ndarray)


@pytest.mark.usefixtures("concentration_fc", "elevation_fc", "conductance_fc")
def test_render_concentration(
    concentration_fc,
    elevation_fc,
    conductance_fc,
):
    directory = pathlib.Path("mymodel")
    globaltimes = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-02"),
        np.datetime64("2000-01-03"),
    ]

    drn = imod.mf6.Drainage(
        elevation=elevation_fc,
        conductance=conductance_fc,
        concentration=concentration_fc,
        concentration_boundary_type="AUX",
    )

    actual = drn.render(directory, "drn", globaltimes, False)

    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity temperature
        end options

        begin dimensions
          maxbound 0
        end dimensions

        begin period 1
          open/close mymodel/drn/drn-0.dat
        end period
        begin period 2
          open/close mymodel/drn/drn-1.dat
        end period
        begin period 3
          open/close mymodel/drn/drn-2.dat
        end period
        """
    )
    assert actual == expected

import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError


@pytest.fixture(scope="function")
def idomain_and_bottom():
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
    idomain = xr.DataArray(np.ones(shape, dtype=np.int8), coords=coords, dims=dims)
    bottom = xr.DataArray([-200.0, -350.0, -450.0], {"layer": layer}, ("layer",))

    return idomain, bottom


def test_render(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom

    directory = pathlib.Path("mymodel")
    dis = imod.mf6.StructuredDiscretization(top=200.0, bottom=bottom, idomain=idomain)
    actual = dis.render(directory, "dis", None, True)
    expected = textwrap.dedent(
        """\
        begin options
          xorigin 0.0
          yorigin 0.0
        end options

        begin dimensions
          nlay 3
          nrow 15
          ncol 15
        end dimensions

        begin griddata
          delr
            constant 5000.0
          delc
            constant 5000.0
          top
            constant 200.0
          botm layered
            constant -200.0
            constant -350.0
            constant -450.0
          idomain
            open/close mymodel/dis/idomain.bin (binary)
        end griddata
        """
    )
    assert actual == expected


def test_wrong_dtype(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom

    with pytest.raises(ValidationError):
        imod.mf6.StructuredDiscretization(
            top=200.0, bottom=bottom, idomain=idomain.astype(np.float64)
        )


def test_bottom_exceeding_itself(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom

    bottom[2] = 100

    dis = imod.mf6.StructuredDiscretization(top=200.0, bottom=bottom, idomain=idomain)

    errors = dis._validate(dis._write_schemata, idomain=idomain)
    assert len(errors) == 1
    assert isinstance(errors["bottom"][0], ValidationError)


def test_top_exceeding_bottom(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom

    dis = imod.mf6.StructuredDiscretization(top=-400.0, bottom=bottom, idomain=idomain)
    errors = dis._validate(dis._write_schemata, idomain=idomain)
    assert len(errors) == 1
    for var, var_errors in errors.items():
        assert var == "top"


def test_overlaying_bottom_inactive(idomain_and_bottom):
    """
    To compute thicknesses properly, Modflow 6 requires bottom data in the
    layer above the active cell in question.
    """
    idomain, bottom = idomain_and_bottom

    # Broadcast bottom to full 3D grid.
    bottom = bottom.where(idomain == 1)

    # Deactivate cell in layer 1.
    idomain[0, 7, 7] = 0

    # No error: bottom in layer 1 required for thickness layer 2
    dis = imod.mf6.StructuredDiscretization(top=200.0, bottom=bottom, idomain=idomain)
    errors = dis._validate(dis._write_schemata, idomain=idomain)
    assert len(errors) == 0

    # Now deactivate bottom in 0,7,7 as well
    bottom[0, 7, 7] = np.nan

    # Error: Thickness in layer 2 cannot be computed.
    dis = imod.mf6.StructuredDiscretization(top=200.0, bottom=bottom, idomain=idomain)
    errors = dis._validate(dis._write_schemata, idomain=idomain)
    assert len(errors) == 1
    for var, var_errors in errors.items():
        assert var == "bottom"


def test_disconnected_idomain(idomain_and_bottom):
    """
    Test if no disconnected arrays occur
    """

    idomain, bottom = idomain_and_bottom

    # Inactive edge, no error
    idomain[:, 0, :] = 0
    dis = imod.mf6.StructuredDiscretization(top=200.0, bottom=bottom, idomain=idomain)
    errors = dis._validate(dis._write_schemata, idomain=idomain)
    assert len(errors) == 0

    # Split in the middle, should throw error
    idomain[:, 7, :] = 0
    dis = imod.mf6.StructuredDiscretization(top=200.0, bottom=bottom, idomain=idomain)
    errors = dis._validate(dis._write_schemata, idomain=idomain)
    assert len(errors) == 1
    for var, var_errors in errors.items():
        assert var == "idomain"


def test_write_ascii_griddata_2d_3d(idomain_and_bottom, tmp_path):
    idomain, bottom = idomain_and_bottom
    top = xr.full_like(idomain.isel(layer=0), 200.0, dtype=float)
    bottom = bottom * xr.ones_like(idomain, dtype=float)

    dis = imod.mf6.StructuredDiscretization(top=top, bottom=bottom, idomain=idomain)
    # 2D data should be rows and colums; 3D should be a single row.
    # https://gitlab.com/deltares/imod/imod-python/-/issues/270
    directory = tmp_path / "dis_griddata"
    directory.mkdir()
    write_context = WriteContext(simulation_directory=directory)
    write_context.current_write_directory = directory
    dis.write(pkgname="dis", globaltimes=[], write_context=write_context)

    with open(directory / "dis/top.dat") as f:
        top_content = f.readlines()
    assert len(top_content) == 15

    with open(directory / "dis/botm.dat") as f:
        bottom_content = f.readlines()
    assert len(bottom_content) == 1

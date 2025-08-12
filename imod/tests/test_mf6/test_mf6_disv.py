import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr
import xugrid as xu

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
    xmin = 10_000.0
    xmax = dx * ncol + xmin
    ymin = 10_000.0
    ymax = abs(dy) * nrow + ymin
    dims = ("layer", "y", "x")

    layer = np.array([1, 2, 3])
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}
    idomain = xr.DataArray(np.ones(shape, dtype=np.int8), coords=coords, dims=dims)
    bottom = xr.DataArray([-200.0, -350.0, -450.0], {"layer": layer}, ("layer",))
    idomain = xu.UgridDataArray.from_structured2d(idomain)

    return idomain, bottom


def test_render(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom

    directory = pathlib.Path("mymodel")
    dis = imod.mf6.VerticesDiscretization(top=200.0, bottom=bottom, idomain=idomain)
    actual = dis._render(directory, "dis", None, True)
    expected = textwrap.dedent(
        """\
        begin options
          xorigin 0.0
          yorigin 0.0
        end options

        begin dimensions
          nlay 3
          ncpl 225
          nvert 256
        end dimensions

        begin griddata
          top
            constant 200.0
          botm layered
            constant -200.0
            constant -350.0
            constant -450.0
          idomain
            open/close mymodel/dis/idomain.bin (binary)
        end griddata"""
    )
    assert actual == expected


def test_wrong_dtype(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom

    with pytest.raises(ValidationError):
        imod.mf6.VerticesDiscretization(
            top=200.0, bottom=bottom, idomain=idomain.astype(np.float64)
        )


def test_copy(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom
    disv = imod.mf6.VerticesDiscretization(top=200.0, bottom=bottom, idomain=idomain)
    disv2 = disv.copy()
    assert isinstance(disv2, imod.mf6.VerticesDiscretization)
    assert disv2.dataset.equals(disv.dataset)


def test_zero_thickness_validation(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom
    # Create a bottom array that has constant value of -1 across all layers, so
    # that layers 2 and 3 have thickness 0.
    bottom = (bottom * 0.0) - 1.0
    disv = imod.mf6.VerticesDiscretization(top=0.0, bottom=bottom, idomain=idomain)

    errors = disv._validate(disv._write_schemata, idomain=idomain)
    assert len(errors) == 1
    error = errors["bottom"][0]
    assert isinstance(error, ValidationError)
    assert error.args[0] == "found thickness <= 0.0"


def test_wrong_layer_coord(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom

    with pytest.raises(ValidationError):
        imod.mf6.VerticesDiscretization(
            top=0.0,
            bottom=bottom.assign_coords(layer=[0, 1, 2]),
            idomain=idomain.assign_coords(layer=[0, 1, 2]),
        )


def test_bottom_exceeding_itself(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom

    bottom[2] = 100

    dis = imod.mf6.VerticesDiscretization(top=200.0, bottom=bottom, idomain=idomain)

    errors = dis._validate(dis._write_schemata, idomain=idomain)
    assert len(errors) == 1
    assert isinstance(errors["bottom"][0], ValidationError)


def test_top_exceeding_bottom(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom

    dis = imod.mf6.VerticesDiscretization(top=-400.0, bottom=bottom, idomain=idomain)
    errors = dis._validate(dis._write_schemata, idomain=idomain)
    assert len(errors) == 1
    for var, var_errors in errors.items():
        assert var == "top"

    # No error should be thrown if zero to negative thickness in vertical
    # passthrough
    idomain[0:2, :] = -1
    dis = imod.mf6.VerticesDiscretization(top=-400.0, bottom=bottom, idomain=idomain)
    errors = dis._validate(dis._write_schemata, idomain=idomain)
    assert len(errors) == 0

    # Or inactive
    idomain[0:2, :] = 0
    dis = imod.mf6.VerticesDiscretization(top=-400.0, bottom=bottom, idomain=idomain)
    errors = dis._validate(dis._write_schemata, idomain=idomain)
    assert len(errors) == 0


def test_overlaying_bottom_inactive(idomain_and_bottom):
    """
    To compute thicknesses properly, Modflow 6 requires bottom data in the
    layer above the active cell in question.
    """
    idomain, bottom = idomain_and_bottom

    # Broadcast bottom to full 3D grid.
    bottom = bottom.where(idomain == 1)

    # Deactivate cell in layer 1.
    idomain[0, 7] = 0

    # No error: bottom in layer 1 required for thickness layer 2
    dis = imod.mf6.VerticesDiscretization(top=200.0, bottom=bottom, idomain=idomain)
    errors = dis._validate(dis._write_schemata, idomain=idomain)
    assert len(errors) == 0

    # Now deactivate bottom in 0,7,7 as well
    bottom[0, 7] = np.nan

    # Error: Thickness in layer 2 cannot be computed.
    dis = imod.mf6.VerticesDiscretization(top=200.0, bottom=bottom, idomain=idomain)
    errors = dis._validate(dis._write_schemata, idomain=idomain)
    assert len(errors) == 1
    for var, _ in errors.items():
        assert var == "bottom"


def test_write_ascii_griddata_2d_3d(idomain_and_bottom, tmp_path):
    idomain, bottom = idomain_and_bottom
    top = xu.full_like(idomain.isel(layer=0), 200.0, dtype=float)
    bottom = bottom * xu.ones_like(idomain, dtype=float)

    dis = imod.mf6.VerticesDiscretization(top=top, bottom=bottom, idomain=idomain)
    # 2D data should be rows and colums; 3D should be a single row.
    # https://github.com/Deltares/imod-python/issues/270
    directory = tmp_path / "disv_griddata"
    directory.mkdir()
    write_context = WriteContext(simulation_directory=directory)

    dis._write(pkgname="disv", globaltimes=[], write_context=write_context)

    with open(directory / "disv/top.dat") as f:
        top_content = f.readlines()
    assert len(top_content) == 225

    with open(directory / "disv/botm.dat") as f:
        bottom_content = f.readlines()
    assert len(bottom_content) == 3

import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr
import xugrid as xu
from pytest_cases import parametrize_with_cases

import imod
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError
from imod.tests.fixtures.backward_compatibility_fixture import (
    _load_imod5_data_in_memory,
)


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
    coords = {"layer": layer, "y": y, "x": x, "dy": dy, "dx": dx}
    idomain = xr.DataArray(np.ones(shape, dtype=np.int32), coords=coords, dims=dims)
    bottom = xr.DataArray([-200.0, -350.0, -450.0], {"layer": layer}, ("layer",))

    return idomain, bottom


@pytest.fixture(scope="function")
def idomain_extended(idomain_and_bottom):
    idomain, _ = idomain_and_bottom
    idomain = idomain.copy()
    # Create grid with larger x dimension
    coords = idomain.coords.copy()
    x_max = idomain.coords["x"].max().values
    dx = idomain.coords["dx"].values
    n_x = 10
    extension_x = dx * np.arange(1, n_x + 1) + x_max
    coords["x"] = extension_x
    extension = xr.DataArray(
        np.ones((3, 15, n_x), dtype=np.int8), dims=("layer", "y", "x"), coords=coords
    )
    idomain_extended = xr.concat([idomain, extension], dim="x")
    return idomain_extended


def test_render(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom

    directory = pathlib.Path("mymodel")
    dis = imod.mf6.StructuredDiscretization(top=200.0, bottom=bottom, idomain=idomain)
    actual = dis._render(directory, "dis", None, True)
    expected = textwrap.dedent(
        """\
        begin options
          xorigin 10000.0
          yorigin 10000.0
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


def test_copy(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom

    dis = imod.mf6.StructuredDiscretization(top=200.0, bottom=bottom, idomain=idomain)
    dis2 = dis.copy()
    assert isinstance(dis2, imod.mf6.StructuredDiscretization)
    assert dis2.dataset.equals(dis.dataset)


def test_mask(idomain_and_bottom):
    # Arrange
    idomain, bottom = idomain_and_bottom
    dis = imod.mf6.StructuredDiscretization(top=200.0, bottom=bottom, idomain=idomain)
    mask = idomain.copy()
    mask[0, 7, 7] = 0  # Deactivate cell in layer 1.
    # Act
    dis_masked = dis.mask(mask)
    # Assert
    is_positive = dis_masked.dataset["idomain"] >= 0
    is_zero = dis_masked.dataset["idomain"][0, 7, 7] == 0
    assert is_positive.all()
    assert is_zero


class LargerGridCases:
    def case_structured(self, idomain_extended):
        target_grid = idomain_extended.sel(layer=1)
        return target_grid, imod.mf6.StructuredDiscretization

    def case_unstructured(self, idomain_extended):
        target_grid = xu.UgridDataArray.from_structured2d(idomain_extended.sel(layer=1))
        return target_grid, imod.mf6.VerticesDiscretization


@parametrize_with_cases("target_grid, expected_pkg_type", cases=LargerGridCases)
def test_regrid_to_larger_grid(
    idomain_and_bottom, idomain_extended, target_grid, expected_pkg_type
):
    """
    Test in which we regrid to a larger grid. The larger regrid
    results in nans by xugrid, these should be filled properly for idomain. If
    this is done incorrectly, the regridded idomain will contain very large
    negative integer values.
    """
    idomain, bottom = idomain_and_bottom
    bottom_3d = bottom * idomain
    extension_size = idomain_extended.size - idomain.size

    dis = imod.mf6.StructuredDiscretization(
        top=200.0, bottom=bottom_3d, idomain=idomain
    )
    regrid_cache = imod.util.RegridderWeightsCache()
    # Act
    dis_regridded = dis.regrid_like(target_grid, regrid_cache=regrid_cache)
    # Assert
    assert isinstance(dis_regridded, expected_pkg_type)
    is_positive = dis_regridded["idomain"] >= 0
    assert is_positive.all()
    contains_zeros = dis_regridded["idomain"] == 0
    assert contains_zeros.any()
    assert np.count_nonzero(contains_zeros.values) == extension_size
    contains_nans = np.isnan(dis_regridded["bottom"])
    assert contains_nans.any()
    assert np.count_nonzero(contains_nans.values) == extension_size


def test_wrong_dtype(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom

    with pytest.raises(ValidationError):
        imod.mf6.StructuredDiscretization(
            top=200.0, bottom=bottom, idomain=idomain.astype(np.float64)
        )


def test_wrong_layer_coord_value(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom

    with pytest.raises(ValidationError):
        imod.mf6.StructuredDiscretization(
            top=200.0,
            bottom=bottom.assign_coords(layer=[0, 1, 2]),
            idomain=idomain.assign_coords(layer=[0, 1, 2]),
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

    # No error should be thrown if zero to negative thickness in vertical
    # passthrough
    idomain[0:2, :, :] = -1
    dis = imod.mf6.StructuredDiscretization(top=-400.0, bottom=bottom, idomain=idomain)
    errors = dis._validate(dis._write_schemata, idomain=idomain)
    assert len(errors) == 0

    # Or inactive
    idomain[0:2, :, :] = 0
    dis = imod.mf6.StructuredDiscretization(top=-400.0, bottom=bottom, idomain=idomain)
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

    # Middle layer vertical passthrough, no error
    idomain[1, :, :] = -1
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


def test_misalignment(idomain_and_bottom):
    """ "Test misalignment of top and idomain throws error."""
    idomain, bottom = idomain_and_bottom

    nrow = 15
    ncol = 15
    shape = (nrow, ncol)

    dx = 5000.0
    dy = -5000.0
    offset = dx * ncol + dx
    xmin = offset
    xmax = offset + dx * ncol
    ymin = offset
    ymax = offset + abs(dy) * nrow
    dims = ("y", "x")

    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"y": y, "x": x}
    top = xr.DataArray(np.ones(shape, dtype=np.int8), coords=coords, dims=dims)

    with pytest.raises(ValueError, match="align"):
        imod.mf6.StructuredDiscretization(top=top, bottom=bottom, idomain=idomain)


def test_write_ascii_griddata_2d_3d(idomain_and_bottom, tmp_path):
    idomain, bottom = idomain_and_bottom
    top = xr.full_like(idomain.isel(layer=0), 200.0, dtype=float)
    bottom = bottom * xr.ones_like(idomain, dtype=float)

    dis = imod.mf6.StructuredDiscretization(top=top, bottom=bottom, idomain=idomain)
    # 2D data should be rows and colums; 3D should be a single row.
    # https://github.com/Deltares/imod-python/issues/270
    directory = tmp_path / "dis_griddata"
    directory.mkdir()
    write_context = WriteContext(simulation_directory=directory)

    dis._write(pkgname="dis", globaltimes=[], write_context=write_context)

    with open(directory / "dis/top.dat") as f:
        top_content = f.readlines()
    assert len(top_content) == 15

    with open(directory / "dis/botm.dat") as f:
        bottom_content = f.readlines()
    assert len(bottom_content) == 1


def test_from_imod5_data__idomain_values(imod5_dataset):
    imod5_data = imod5_dataset[0]

    dis = imod.mf6.StructuredDiscretization.from_imod5_data(imod5_data)

    # Test if idomain has appropriate count
    assert (dis["idomain"] == -1).sum() == 371824
    assert (dis["idomain"] == 0).sum() == 176912
    # IBOUND -1 was converted to 1, therefore not all active cells are 2
    assert (dis["idomain"] == 1).sum() == 15607
    assert (dis["idomain"] == 2).sum() == 688329


def test_from_imod5_data__grid_extent(imod5_dataset):
    imod5_data = imod5_dataset[0]

    dis = imod.mf6.StructuredDiscretization.from_imod5_data(imod5_data)

    # Test if regridded to smallest grid resolution
    assert dis["top"].dx == 25.0
    assert dis["top"].dy == -25.0
    assert (dis.dataset.coords["x"][1] - dis.dataset.coords["x"][0]) == 25.0
    assert (dis.dataset.coords["y"][1] - dis.dataset.coords["y"][0]) == -25.0

    # Test extent
    assert dis.dataset.coords["y"].min() == 360712.5
    assert dis.dataset.coords["y"].max() == 365287.5
    assert dis.dataset.coords["x"].min() == 194712.5
    assert dis.dataset.coords["x"].max() == 199287.5


def test_from_imod5_data__write(imod5_dataset, tmp_path):
    directory = tmp_path / "dis_griddata"
    directory.mkdir()
    write_context = WriteContext(simulation_directory=directory)
    imod5_data = imod5_dataset[0]

    dis = imod.mf6.StructuredDiscretization.from_imod5_data(imod5_data)

    # Test if package written without ValidationError
    dis._write(pkgname="dis", globaltimes=[], write_context=write_context)

    # Assert if files written
    assert (directory / "dis/top.dat").exists()
    assert (directory / "dis/botm.dat").exists()


def test_from_imod5_data__validation_error(tmp_path):
    # don't use the fixture "imod5_dataset" for this test, because we don't want the
    # ibound cleanup. Without this cleanup we get a validation error,
    # which is what we want to test here.

    tmp_path = imod.util.temporary_directory()
    data = imod.data.imod5_projectfile_data(tmp_path)
    data = data[0]
    # Set bottom above top, to create "gap" intebetween interfaces.
    data["bot"]["bottom"][20, 6, 6] = data["top"]["top"][21, 6, 6] - 1.0

    _load_imod5_data_in_memory(data)
    with pytest.raises(ValidationError):
        imod.mf6.StructuredDiscretization.from_imod5_data(data)

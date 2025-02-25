import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis.strategies import floats
from numpy import nan
from numpy.testing import assert_almost_equal, assert_equal
from pytest_cases import case, parametrize_with_cases

from imod.mf6.dis import StructuredDiscretization
from imod.mf6.utilities.regrid import RegridderWeightsCache
from imod.msw import GridData
from imod.msw.fixed_format import format_fixed_width
from imod.util.spatial import get_total_grid_area


@given(
    floats(
        GridData._metadata_dict["area"].min_value,
        GridData._metadata_dict["area"].max_value,
    ),
    floats(
        GridData._metadata_dict["landuse"].min_value,
        GridData._metadata_dict["landuse"].max_value,
    ),
    floats(
        GridData._metadata_dict["rootzone_depth"].min_value,
        GridData._metadata_dict["rootzone_depth"].max_value,
    ),
    floats(
        GridData._metadata_dict["surface_elevation"].min_value,
        GridData._metadata_dict["surface_elevation"].max_value,
    ),
    floats(
        GridData._metadata_dict["soil_physical_unit"].min_value,
        GridData._metadata_dict["soil_physical_unit"].max_value,
    ),
)
@settings(deadline=400)
def test_write(
    fixed_format_parser,
    area,
    landuse,
    rootzone_depth,
    surface_elevation,
    soil_physical_unit,
):
    # An error will be thrown if dx * dy exceeds the total area specified.
    # Therefore we have to ensure that dx * dy exceeds
    # GridData._metadata_dict["area"].max_value, which is 1e6. So a dx and dy of
    # 1000.0 should do the job.
    coords = {"x": [500.0, 1500.0], "y": [1500.0, 500.0]}
    like = xr.DataArray(np.ones((2, 2)), coords=coords, dims=("y", "x"))
    grid_data = GridData(
        (like * area).expand_dims(subunit=[0]),
        xr.full_like(like, landuse, dtype=int).expand_dims(subunit=[0]),
        (like * rootzone_depth).expand_dims(subunit=[0]),
        (like * surface_elevation),
        (like * soil_physical_unit),
        xr.full_like(like, True, dtype=bool),
    )

    index, svat = grid_data.generate_index_array()

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        grid_data.write(output_dir, index, svat, None, None)

        results = fixed_format_parser(
            output_dir / GridData._file_name, GridData._metadata_dict
        )

    assert_almost_equal(
        results["area"],
        float(
            format_fixed_width(
                area,
                GridData._metadata_dict["area"],
            )
        ),
    )
    assert_almost_equal(
        results["rootzone_depth"],
        float(
            format_fixed_width(
                rootzone_depth,
                GridData._metadata_dict["rootzone_depth"],
            )
        ),
    )
    assert_almost_equal(
        results["surface_elevation"],
        float(
            format_fixed_width(
                surface_elevation,
                GridData._metadata_dict["surface_elevation"],
            )
        ),
    )

    assert_equal(results["landuse"][0], int(landuse))
    assert_equal(results["soil_physical_unit"][0], int(soil_physical_unit))


@pytest.fixture(scope="function")
def coords_planar() -> dict:
    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
    dx = 1.0
    dy = 1.0
    return {"y": y, "x": x, "dx": dx, "dy": dy}


@pytest.fixture(scope="function")
def coords_one_subunit(coords_planar: dict) -> dict:
    coords_subunit = deepcopy(coords_planar)
    coords_subunit["subunit"] = [0]
    return coords_subunit


@pytest.fixture(scope="function")
def coords_two_subunit(coords_planar: dict) -> dict:
    coords_subunit = deepcopy(coords_planar)
    coords_subunit["subunit"] = [0, 1]
    return coords_subunit


@case(tags="one_subunit")
def case_grid_data_one_subunits(
    coords_one_subunit: dict, coords_planar: dict
) -> dict[str, xr.DataArray]:
    data = {}
    # fmt: off
    data["area"] = xr.DataArray(
        np.array(
            [
                [[0.5, 0.5, 0.5],
                 [0.7, 0.7, 0.7],
                 [1.0, 1.0, 1.0]],

            ]
        ),
        dims=("subunit", "y", "x"),
        coords=coords_one_subunit
    )
    data["landuse"] = xr.DataArray(
        np.array(
            [
                [[1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords=coords_one_subunit
    )
    data["rootzone_depth"] = xr.DataArray(
        np.array(
            [
                [[1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords=coords_one_subunit
    )

    data["surface_elevation"] = xr.DataArray(
        np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]),
        dims=("y", "x"),
        coords=coords_planar
    )

    data["soil_physical_unit"] = xr.DataArray(
        np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]),
        dims=("y", "x"),
        coords=coords_planar
    )

    data["active"] = xr.DataArray(
        np.array(
            [[False, True, False],
             [False, True, False],
             [False, True, False]]),
        dims=("y", "x"),
        coords=coords_planar
    )
    # fmt: on
    return data


@case(tags="two_subunit")
def case_grid_data_two_subunits(
    coords_two_subunit: dict, coords_planar: dict
) -> dict[str, xr.DataArray]:
    data = {}
    # fmt: off
    data["area"] = xr.DataArray(
        np.array(
            [
                [[0.5, 0.5, 0.5],
                 [nan, nan, nan],
                 [1.0, 1.0, 1.0]],

                [[0.5, 0.5, 0.5],
                 [1.0, 1.0, 1.0],
                 [nan, nan, nan]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords=coords_two_subunit
    )
    data["landuse"] = xr.DataArray(
        np.array(
            [
                [[1.0, 1.0, 1.0],
                 [nan, nan, nan],
                 [1.0, 1.0, 1.0]],

                [[2.0, 2.0, 2.0],
                 [2.0, 2.0, 2.0],
                 [nan, nan, nan]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords=coords_two_subunit
    )
    data["rootzone_depth"] = xr.DataArray(
        np.array(
            [
                [[1.0, 1.0, 1.0],
                 [nan, nan, nan],
                 [1.0, 1.0, 1.0]],

                [[1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [nan, nan, nan]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords=coords_two_subunit
    )

    data["surface_elevation"] = xr.DataArray(
        np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]),
        dims=("y", "x"),
        coords=coords_planar
    )

    data["soil_physical_unit"] = xr.DataArray(
        np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]),
        dims=("y", "x"),
        coords=coords_planar
    )

    data["active"] = xr.DataArray(
        np.array(
            [[False, True, False],
             [False, True, False],
             [False, True, False]]),
        dims=("y", "x"),
        coords=coords_planar
    )
    # fmt: on
    return data


@case(tags="two_subunit")
def case_grid_data_two_subunits__dask(
    coords_two_subunit: dict, coords_planar: dict
) -> dict[str, xr.DataArray]:
    data = case_grid_data_two_subunits(coords_two_subunit, coords_planar)
    for key, values in data.items():
        data[key] = values.chunk({"x": 3, "y": 1})
    return data


@parametrize_with_cases("grid_data_dict", cases=".", has_tag="two_subunit")
def test_generate_index_array(
    grid_data_dict: dict[str, xr.DataArray], coords_two_subunit: dict
):
    grid_data = GridData(**grid_data_dict)

    index, svat = grid_data.generate_index_array()

    index_expected = [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]

    # fmt: off
    svat_expected = xr.DataArray(
        np.array(
            [
                [[0, 1, 0],
                 [0, 0, 0],
                 [0, 2, 0]],

                [[0, 3, 0],
                 [0, 4, 0],
                 [0, 0, 0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords=coords_two_subunit,
    )
    # fmt: on
    assert_equal(index, np.array(index_expected))
    assert_equal(svat.values, svat_expected.values)


@parametrize_with_cases("grid_data_dict", cases=".", has_tag="two_subunit")
def test_simple_model(fixed_format_parser, grid_data_dict: dict[str, xr.DataArray]):
    grid_data = GridData(**grid_data_dict)

    index, svat = grid_data.generate_index_array()

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        grid_data.write(output_dir, index, svat, None, None)

        results = fixed_format_parser(
            output_dir / GridData._file_name, GridData._metadata_dict
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_almost_equal(results["area"], np.array([0.5, 1.0, 0.5, 1.0]))
    assert_almost_equal(results["surface_elevation"], np.array([2.0, 8.0, 2.0, 5.0]))
    assert_equal(results["soil_physical_unit"], np.array([2, 8, 2, 5]))
    assert_equal(results["landuse"], np.array([1, 1, 2, 2]))
    assert_almost_equal(results["rootzone_depth"], np.array([1.0, 1.0, 1.0, 1.0]))


# Only use two subunit case, as this one sums to total area whereas the one
# subunit case doesn't sum to 1 across subunit.
@parametrize_with_cases("grid_data_dict", cases=".", has_tag="two_subunit")
def test_simple_model_regrid(
    simple_2d_grid_with_subunits, grid_data_dict: dict[str, xr.DataArray]
):
    grid_data = GridData(**grid_data_dict)
    new_grid = simple_2d_grid_with_subunits

    regrid_context = RegridderWeightsCache()

    regridded_griddata = grid_data.regrid_like(new_grid, regrid_context)

    regridded_area = regridded_griddata.dataset["area"].sum(dim="subunit")
    regridded_total_area = get_total_grid_area(new_grid.sel({"subunit": 0}, drop=True))
    assert np.sum(regridded_area.values) == regridded_total_area


@parametrize_with_cases("grid_data_dict", cases=".", has_tag="one_subunit")
def test_simple_model_1_subunit(
    fixed_format_parser, grid_data_dict: dict[str, xr.DataArray]
):
    grid_data = GridData(**grid_data_dict)

    index, svat = grid_data.generate_index_array()

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        grid_data.write(output_dir, index, svat, None, None)

        results = fixed_format_parser(
            output_dir / GridData._file_name, GridData._metadata_dict
        )

    assert_equal(results["svat"], np.array([1, 2, 3]))
    assert_almost_equal(results["area"], np.array([0.5, 0.7, 1.0]))
    assert_almost_equal(results["surface_elevation"], np.array([2.0, 5.0, 8.0]))
    assert_equal(results["soil_physical_unit"], np.array([2, 5, 8]))
    assert_equal(results["landuse"], np.array([1, 1, 1]))
    assert_almost_equal(results["rootzone_depth"], np.array([1.0, 1.0, 1.0]))


@parametrize_with_cases("grid_data_dict", cases=".", has_tag="two_subunit")
def test_area_grid_exceeds_cell_area(
    grid_data_dict: dict[str, xr.DataArray], coords_two_subunit: dict
):
    """
    Test where provided area grid exceeds total cell area, should throw error.
    """
    # fmt: off
    grid_data_dict["area"] = xr.DataArray(
        np.array(
            [
                [[0.5, 0.5, 0.5],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0]],

                [[0.5, 0.5, 0.5],
                 [1.0, 1.0, 1.0],
                 [nan, nan, nan]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords=coords_two_subunit
    )
    # fmt: on
    with pytest.raises(ValueError):
        GridData(**grid_data_dict)


@parametrize_with_cases("grid_data_dict", cases=".")
def test_non_equidistant(grid_data_dict: dict[str, xr.DataArray]):
    """
    Test where provided grid is non-equidistant, should throw error.
    """
    dx = [1.0, 1.0, 5.0]
    dy = [1.0, 1.0, 5.0]

    for key, value in grid_data_dict.items():
        grid_data_dict[key] = value.assign_coords(dx=("x", dx), dy=("y", dy))

    with pytest.raises(ValueError):
        GridData(**grid_data_dict)


@parametrize_with_cases("grid_data_dict", cases=".")
def test_from_imod5_data(grid_data_dict: dict[str, xr.DataArray]):
    cap_data = {}
    cap_data["wetted_area"] = 1 - grid_data_dict["area"].sum(dim="subunit")
    like = grid_data_dict["area"].sel(subunit=0, drop=True)
    top = xr.zeros_like(like, dtype=float)
    cap_data["boundary"] = xr.ones_like(like, dtype=int)
    cap_data["urban_area"] = xr.zeros_like(like, dtype=float) + 0.1
    cap_data["landuse"] = xr.ones_like(like, dtype=int)
    cap_data["rootzone_thickness"] = xr.ones_like(like, dtype=int)
    cap_data["surface_elevation"] = top
    cap_data["soil_physical_unit"] = xr.ones_like(like, dtype=int)
    cap_data["active"] = xr.ones_like(like, dtype=bool)

    imod5_data = {"cap": cap_data}

    layer = xr.DataArray([1, 1], coords={"layer": [1, 2]}, dims=("layer",))
    idomain = layer * xr.ones_like(like, dtype=int)

    # Dis only needed for idomain
    dis = StructuredDiscretization(top, top - 0.1, idomain, validate=False)

    griddata, _ = GridData.from_imod5_data(imod5_data, target_dis=dis)
    expected_rootzone_depth = cap_data["rootzone_thickness"] * 0.01
    xr.testing.assert_allclose(
        expected_rootzone_depth, griddata["rootzone_depth"].sel(subunit=0, drop=True)
    )
    assert (griddata["landuse"].sel(subunit=1, drop=True) == 18).all()

import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from hypothesis import given, settings
from hypothesis.strategies import floats
from numpy import nan
from numpy.testing import assert_almost_equal, assert_equal

from imod.fixed_format import format_fixed_width
from imod.msw import GridData


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
    grid_data = GridData(
        xr.DataArray(area).expand_dims(subunit=[0]),
        xr.DataArray(landuse).expand_dims(subunit=[0]),
        xr.DataArray(rootzone_depth).expand_dims(subunit=[0]),
        xr.DataArray(surface_elevation),
        xr.DataArray(soil_physical_unit),
        xr.DataArray(True),
    )

    index, svat = grid_data.generate_index_array()

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        grid_data.write(output_dir, index, svat)

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


def test_simple_model(fixed_format_parser):

    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    subunit = [0, 1]
    dx = 1.0
    dy = 1.0
    # fmt: off
    area = xr.DataArray(
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
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )
    landuse = xr.DataArray(
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
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )
    rootzone_depth = xr.DataArray(
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
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )

    surface_elevation = xr.DataArray(
        np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]),
        dims=("y", "x"),
        coords={"y": y, "x": x, "dx": dx, "dy": dy}
    )

    soil_physical_unit = xr.DataArray(
        np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]),
        dims=("y", "x"),
        coords={"y": y, "x": x, "dx": dx, "dy": dy}
    )

    active = xr.DataArray(
        np.array(
            [[False, True, False],
             [False, True, False],
             [False, True, False]]),
        dims=("y", "x"),
        coords={"y": y, "x": x}
    )
    # fmt: on

    grid_data = GridData(
        area,
        landuse,
        rootzone_depth,
        surface_elevation,
        soil_physical_unit,
        active,
    )

    index, svat = grid_data.generate_index_array()

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        grid_data.write(output_dir, index, svat)

        results = fixed_format_parser(
            output_dir / GridData._file_name, GridData._metadata_dict
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_almost_equal(results["area"], np.array([0.5, 1.0, 0.5, 1.0]))
    assert_almost_equal(results["surface_elevation"], np.array([2.0, 8.0, 2.0, 5.0]))
    assert_equal(results["soil_physical_unit"], np.array([2, 8, 2, 5]))
    assert_equal(results["landuse"], np.array([1, 1, 2, 2]))
    assert_almost_equal(results["rootzone_depth"], np.array([1.0, 1.0, 1.0, 1.0]))

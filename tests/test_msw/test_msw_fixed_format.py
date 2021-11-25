import tempfile
from pathlib import Path

import xarray as xr
from hypothesis import given
from hypothesis.strategies import floats
from numpy.testing import assert_almost_equal

from imod.msw import GridData, Infiltration


def fixed_format_parser(file, metadata_dict):
    results = {}
    with open(file) as f:
        for varname, metadata in metadata_dict.items():
            results[varname] = f.read(metadata.column_width)

    return results


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
def test_grid_data_write(
    area, landuse, rootzone_depth, surface_elevation, soil_physical_unit
):
    grid_data = GridData(
        xr.DataArray(area).expand_dims(subunit=[0]),
        xr.DataArray(landuse).expand_dims(subunit=[0]),
        xr.DataArray(rootzone_depth).expand_dims(subunit=[0]),
        xr.DataArray(surface_elevation),
        xr.DataArray(soil_physical_unit),
        xr.DataArray(True),
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        grid_data.write(output_dir)

        results = fixed_format_parser(
            output_dir / GridData._file_name, GridData._metadata_dict
        )

    # (original) integers are compared differentlz than floats
    assert_almost_equal(float(results["area"]), area, decimal=2)
    assert int(results["landuse"]) == int(landuse)
    assert_almost_equal(float(results["rootzone_depth"]), rootzone_depth, decimal=2)
    assert_almost_equal(
        float(results["surface_elevation"]), surface_elevation, decimal=2
    )
    assert int(results["soil_physical_unit"]) == int(soil_physical_unit)


@given(
    floats(
        Infiltration._metadata_dict["infiltration_capacity"].min_value,
        Infiltration._metadata_dict["infiltration_capacity"].max_value,
    ),
    floats(
        Infiltration._metadata_dict["downward_resistance"].min_value,
        Infiltration._metadata_dict["downward_resistance"].max_value,
    ),
    floats(
        Infiltration._metadata_dict["upward_resistance"].min_value,
        Infiltration._metadata_dict["upward_resistance"].max_value,
    ),
    floats(
        Infiltration._metadata_dict["bottom_resistance"].min_value,
        Infiltration._metadata_dict["bottom_resistance"].max_value,
    ),
    floats(
        Infiltration._metadata_dict["extra_storage_coefficient"].min_value,
        Infiltration._metadata_dict["extra_storage_coefficient"].max_value,
    ),
)
def test_infiltration_write(
    infiltration_capacity,
    downward_resistance,
    upward_resistance,
    bottom_resistance,
    extra_storage_coefficient,
):
    grid_data = Infiltration(
        xr.DataArray(infiltration_capacity).expand_dims(subunit=[0]),
        xr.DataArray(downward_resistance),
        xr.DataArray(upward_resistance),
        xr.DataArray(bottom_resistance),
        xr.DataArray(extra_storage_coefficient),
        xr.DataArray(True),
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        grid_data.write(output_dir)

        results = fixed_format_parser(
            output_dir / Infiltration._file_name, Infiltration._metadata_dict
        )

    assert_almost_equal(
        float(results["infiltration_capacity"]), infiltration_capacity, decimal=2
    )
    assert_almost_equal(
        float(results["downward_resistance"]), downward_resistance, decimal=2
    )
    assert_almost_equal(
        float(results["upward_resistance"]), upward_resistance, decimal=2
    )
    assert_almost_equal(
        float(results["bottom_resistance"]), bottom_resistance, decimal=2
    )
    assert_almost_equal(
        float(results["extra_storage_coefficient"]),
        extra_storage_coefficient,
        decimal=2,
    )

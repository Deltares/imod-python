import tempfile
from pathlib import Path

import xarray as xr
from hypothesis import given
from hypothesis.strategies import floats
from numpy.testing import assert_almost_equal

from imod.msw import GridData, Infiltration
from imod.msw.pkgbase import Package


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

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        grid_data.write(output_dir)

        results = fixed_format_parser(
            output_dir / GridData._file_name, GridData._metadata_dict
        )

    assert_almost_equal(
        float(results["area"]),
        float(
            Package.format_fixed_width(
                area,
                GridData._metadata_dict["area"],
            )
        ),
    )
    assert_almost_equal(
        float(results["rootzone_depth"]),
        float(
            Package.format_fixed_width(
                rootzone_depth,
                GridData._metadata_dict["rootzone_depth"],
            )
        ),
    )
    assert_almost_equal(
        float(results["surface_elevation"]),
        float(
            Package.format_fixed_width(
                surface_elevation,
                GridData._metadata_dict["surface_elevation"],
            )
        ),
    )

    # integers are compared differently than floats
    assert int(results["landuse"]) == int(landuse)
    assert int(results["soil_physical_unit"]) == int(soil_physical_unit)

import tempfile
from pathlib import Path

import xarray as xr
from hypothesis import given
from hypothesis.strategies import floats, integers

from imod.msw import GridData


@given(
    floats(
        GridData._metadata_dict["area"].min_value,
        GridData._metadata_dict["area"].max_value,
    ),
    integers(
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
    integers(
        GridData._metadata_dict["soil_physical_unit"].min_value,
        GridData._metadata_dict["soil_physical_unit"].max_value,
    ),
)
def test_decode_inverts_encode(
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

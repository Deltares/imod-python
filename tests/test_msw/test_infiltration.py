import tempfile
from pathlib import Path

import xarray as xr
from hypothesis import given
from hypothesis.strategies import floats
from numpy.testing import assert_almost_equal

from imod.msw import Infiltration
from imod.msw.pkgbase import Package


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
def test_write(
    fixed_format_parser,
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
        results["infiltration_capacity"],
        float(
            Package.format_fixed_width(
                infiltration_capacity,
                Infiltration._metadata_dict["infiltration_capacity"],
            )
        ),
    )
    assert_almost_equal(
        results["downward_resistance"],
        float(
            Package.format_fixed_width(
                downward_resistance,
                Infiltration._metadata_dict["downward_resistance"],
            )
        ),
    )

    assert_almost_equal(
        results["upward_resistance"],
        float(
            Package.format_fixed_width(
                upward_resistance,
                Infiltration._metadata_dict["upward_resistance"],
            )
        ),
    )

    assert_almost_equal(
        results["bottom_resistance"],
        float(
            Package.format_fixed_width(
                bottom_resistance,
                Infiltration._metadata_dict["bottom_resistance"],
            )
        ),
    )

    assert_almost_equal(
        results["extra_storage_coefficient"],
        float(
            Package.format_fixed_width(
                extra_storage_coefficient,
                Infiltration._metadata_dict["extra_storage_coefficient"],
            )
        ),
    )

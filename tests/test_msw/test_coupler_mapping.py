import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy import nan
from numpy.testing import assert_equal

from imod import mf6, msw


def test_simple_model(fixed_format_parser):

    x = [1, 2, 3]
    y = [1, 2, 3]
    subunit = [0, 1]
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
        coords={"subunit": subunit, "y": y, "x": x}
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

    # Well
    well_layer = [3, 2, 1]
    well_row = [2, 1, 2]
    well_column = [1, 2, 2]
    well_rate = [-5.0] * 3
    well = mf6.Well(
        layer=well_layer,
        row=well_row,
        column=well_column,
        rate=well_rate,
    )

    grid_data = msw.CouplerMapping(area, active, well)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        grid_data.write(output_dir)

        results = fixed_format_parser(
            output_dir / msw.CouplerMapping._file_name,
            msw.CouplerMapping._metadata_dict,
        )

    # TODO: extend arrays to add mappings coming from sprinkling
    assert_equal(results["mod_id"], np.array([2, 8, 2, 5]))
    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_equal(results["layer"], np.array([1, 1, 1, 1]))

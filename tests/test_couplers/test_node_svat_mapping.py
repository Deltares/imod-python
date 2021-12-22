import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy import nan
from numpy.testing import assert_equal

from imod.couplers import metamod


def test_simple_model(fixed_format_parser):

    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
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

    grid_data = metamod.node_svat_mapping.NodeSvatMapping(
        area,
        active,
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        grid_data.write(output_dir)

        results = fixed_format_parser(
            output_dir / metamod.node_svat_mapping.NodeSvatMapping._file_name,
            metamod.node_svat_mapping.NodeSvatMapping._metadata_dict,
        )

    assert_equal(results["mod_id"], np.array([2, 8, 2, 5]))
    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_equal(results["layer"], np.array([1, 1, 1, 1]))

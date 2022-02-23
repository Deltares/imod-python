import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy import nan
from numpy.testing import assert_almost_equal, assert_equal

from imod.fixed_format import format_fixed_width
from imod.msw import Ponding


def test_simple_model(fixed_format_parser):

    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    subunit = [0, 1]
    dx = 1.0
    dy = 1.0

    # fmt: off

    ponding_depth = xr.DataArray(
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

    runoff_resistance = xr.DataArray(
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

    active = xr.DataArray(
        np.array(
            [[False, True, False],
             [False, True, False],
             [False, True, False]]),
        dims=("y", "x"),
        coords={"y": y, "x": x, "dx": dx, "dy": dy}
    )

    # fmt: on

    ponding = Ponding(
        ponding_depth=ponding_depth,
        runoff_resistance=runoff_resistance,
        runon_resistance=runoff_resistance,
        active=active,
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ponding.write(output_dir)

        results = fixed_format_parser(
            output_dir / Ponding._file_name, Ponding._metadata_dict
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_equal(results["swnr"], np.array([0, 0, 0, 0]))
    assert_almost_equal(results["ponding_depth"], np.array([0.5, 1.0, 0.5, 1.0]))
    assert_almost_equal(results["runoff_resistance"], np.array([0.5, 1.0, 0.5, 1.0]))
    assert_almost_equal(results["runoff_resistance"], np.array([0.5, 1.0, 0.5, 1.0]))

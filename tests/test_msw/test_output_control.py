import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy import nan
from numpy.testing import assert_almost_equal, assert_equal

from imod.msw import IdfOutputControl


def grid():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 3.0]
    subunit = [0, 1]
    dx = 1.0
    dy = 1.0
    # fmt: off
    area = xr.DataArray(
        np.array(
            [
                [[0.5, 0.5, 0.5, 0.5],
                 [nan, nan, nan, nan],
                 [1.0, 1.0, 1.0, 1.0]],

                [[0.5, 0.5, 0.5, 0.5],
                 [1.0, 1.0, 1.0, 1.0],
                 [nan, nan, nan, nan]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )
    active = xr.DataArray(
        np.array(
            [[False, True, False, False],
             [False, True, False, True],
             [False, True, False, False]]),
        dims=("y", "x"),
        coords={"y": y, "x": x}
    )
    # fmt: on
    return area, active


def test_idf_oc_write_simple_model(fixed_format_parser):
    area, active = grid()
    nodata = -9999.0

    idf_output_control = IdfOutputControl(area, active, nodata)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        idf_output_control.write(output_dir)

        results = fixed_format_parser(
            output_dir / IdfOutputControl._file_name, IdfOutputControl._metadata_dict
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4, 5]))
    assert_equal(results["rows"], np.array([1, 3, 1, 2, 2]))
    assert_equal(results["columns"], np.array([2, 2, 2, 2, 4]))
    assert_almost_equal(results["y_coords"], np.array([1.0, 3.0, 1.0, 2.0, 2.0]))
    assert_almost_equal(results["x_coords"], np.array([2.0, 2.0, 2.0, 2.0, 4.0]))


def test_idf_oc_settings_simple_model():
    area, active = grid()
    nodata = -9999.0

    idf_output_control = IdfOutputControl(area, active, nodata)

    expected = dict(
        simgro_opt=-1,
        idf_per=1,
        idf_dx=1.0,
        idf_dy=1.0,
        idf_ncol=4,
        idf_nrow=3,
        idf_xmin=0.5,
        idf_ymin=0.5,
        idf_nodata=nodata,
    )

    settings = idf_output_control.get_settings()

    assert expected == settings

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numpy import nan
from numpy.testing import assert_almost_equal, assert_equal

from imod.msw import IdfOutputControl, VariableOutputControl, TimeOutputControl


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


def test_var_oc(fixed_format_parser):
    var_output_control = VariableOutputControl()

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        var_output_control.write(output_dir)

        results = fixed_format_parser(
            output_dir / VariableOutputControl._file_name,
            VariableOutputControl._metadata_dict,
        )

    variable_names = ["Pm", "Psgw", "Pssw", "qrun", "qdr", "qspgw", "qmodf", "ETact"]
    expected_names = np.array(["{:10}".format(v) for v in variable_names])

    assert_equal(results["variable"], expected_names)
    assert_equal(results["option"], np.array([1, 1, 1, 1, 1, 1, 1, 1]))


def test_time_oc(fixed_format_parser):
    freq = "D"
    times = pd.date_range(start="1/1/1971", end="1/3/1971", freq=freq)

    time_output_control = TimeOutputControl(time=times)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        time_output_control.write(output_dir)

        results = fixed_format_parser(
            output_dir / TimeOutputControl._file_name, TimeOutputControl._metadata_dict
        )

    assert_almost_equal(results["time_since_start_year"], np.array([1.0, 2.0, 3.0]))
    assert_equal(results["year"], [1971, 1971, 1971])
    assert_equal(results["option"], [7, 7, 7])


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

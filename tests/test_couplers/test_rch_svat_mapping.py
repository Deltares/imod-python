import tempfile
from pathlib import Path
import pytest

import numpy as np
import xarray as xr
from numpy import nan
from numpy.testing import assert_equal
from imod import mf6

from imod.couplers.metamod.rch_svat_mapping import RechargeSvatMapping


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

    rate = xr.DataArray(
        np.array(
            [[0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ),
        dims=("y", "x"),
        coords={"y": y, "x": x}
    )

    # fmt: on
    recharge = mf6.Recharge(rate)

    grid_data = RechargeSvatMapping(area, active, recharge)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        grid_data.write(output_dir)

        results = fixed_format_parser(
            output_dir / RechargeSvatMapping._file_name,
            RechargeSvatMapping._metadata_dict,
        )

    assert_equal(results["rch_id"], np.array([2, 8, 2, 5]))
    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_equal(results["layer"], np.array([1, 1, 1, 1]))


def test_simple_model_inactive_rch(fixed_format_parser):
    """
    Test simple model where recharge is inactive where metaswap is inactive as
    well. This changes the numbering of rch_id.
    """
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

    rate = xr.DataArray(
        np.array(
            [[nan, 0.0, 0.0],
             [nan, 0.0, 0.0],
             [nan, 0.0, 0.0]],
        ),
        dims=("y", "x"),
        coords={"y": y, "x": x}
    )

    # fmt: on
    recharge = mf6.Recharge(rate)

    grid_data = RechargeSvatMapping(area, active, recharge)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        grid_data.write(output_dir)

        results = fixed_format_parser(
            output_dir / RechargeSvatMapping._file_name,
            RechargeSvatMapping._metadata_dict,
        )

    assert_equal(results["rch_id"], np.array([1, 5, 1, 3]))
    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_equal(results["layer"], np.array([1, 1, 1, 1]))


def test_simple_model_inactive_rch_error():
    """
    Test simple model where recharge is inactive where metaswap is active. This
    should raise an error.
    """
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

    rate = xr.DataArray(
        np.array(
            [[nan, nan, 0.0],
             [nan, nan, 0.0],
             [nan, nan, 0.0]],
        ),
        dims=("y", "x"),
        coords={"y": y, "x": x}
    )

    # fmt: on
    recharge = mf6.Recharge(rate)

    with pytest.raises(ValueError):
        RechargeSvatMapping(area, active, recharge)

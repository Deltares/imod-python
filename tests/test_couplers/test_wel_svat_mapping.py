import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.testing import assert_equal

from imod import mf6
from imod.couplers.metamod.wel_svat_mapping import WellSvatMapping


def test_simple_model(fixed_format_parser):

    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    subunit = [0, 1]
    dx = 1.0
    dy = 1.0
    # fmt: off
    svat = xr.DataArray(
        np.array(
            [
                [[0, 1, 0],
                 [0, 0, 0],
                 [0, 2, 0]],

                [[0, 3, 0],
                 [0, 4, 0],
                 [0, 0, 0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )
    # fmt: on
    index = (svat != 0).values.ravel()

    # Well
    well_layer = [3, 2, 1]
    well_row = [1, 2, 3]
    well_column = [2, 2, 2]
    well_rate = [-5.0] * 3
    well = mf6.WellDisStructured(
        layer=well_layer,
        row=well_row,
        column=well_column,
        rate=well_rate,
    )

    coupler_mapping = WellSvatMapping(svat, well)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        coupler_mapping.write(output_dir, index, svat)

        results = fixed_format_parser(
            output_dir / WellSvatMapping._file_name,
            WellSvatMapping._metadata_dict,
        )

    assert_equal(results["wel_id"], np.array([1, 3, 1, 2]))
    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_equal(results["layer"], np.array([3, 1, 3, 2]))


def test_simple_model_1_subunit(fixed_format_parser):

    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    subunit = [0]
    dx = 1.0
    dy = 1.0
    # fmt: off
    svat = xr.DataArray(
        np.array(
            [
                [[0, 1, 0],
                 [0, 0, 0],
                 [0, 2, 0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )
    # fmt: on
    index = (svat != 0).values.ravel()

    # Well
    well_layer = [3, 2]
    well_row = [1, 3]
    well_column = [2, 2]
    well_rate = [-5.0] * 2
    well = mf6.WellDisStructured(
        layer=well_layer,
        row=well_row,
        column=well_column,
        rate=well_rate,
    )

    coupler_mapping = WellSvatMapping(svat, well)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        coupler_mapping.write(output_dir, index, svat)

        results = fixed_format_parser(
            output_dir / WellSvatMapping._file_name,
            WellSvatMapping._metadata_dict,
        )

    assert_equal(results["wel_id"], np.array([1, 2]))
    assert_equal(results["svat"], np.array([1, 2]))
    assert_equal(results["layer"], np.array([3, 2]))


def test_simple_model_inactive(fixed_format_parser):
    """
    Test with first well in inactive metaswap cell. This should increase the
    wel_id number, as the first modflow 6 well is not coupled to.
    """

    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    subunit = [0, 1]
    dx = 1.0
    dy = 1.0
    # fmt: off
    svat = xr.DataArray(
        np.array(
            [
                [[0, 1, 0],
                 [0, 0, 0],
                 [0, 2, 0]],

                [[0, 3, 0],
                 [0, 4, 0],
                 [0, 0, 0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )
    # fmt: on
    index = (svat != 0).values.ravel()

    # Well
    well_layer = [1, 3, 2, 1]
    well_row = [1, 1, 2, 3]
    well_column = [1, 2, 2, 2]
    well_rate = [-5.0] * 4
    well = mf6.WellDisStructured(
        layer=well_layer,
        row=well_row,
        column=well_column,
        rate=well_rate,
    )

    coupler_mapping = WellSvatMapping(svat, well)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        coupler_mapping.write(output_dir, index, svat)

        results = fixed_format_parser(
            output_dir / WellSvatMapping._file_name,
            WellSvatMapping._metadata_dict,
        )

    assert_equal(results["wel_id"], np.array([2, 4, 2, 3]))
    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_equal(results["layer"], np.array([3, 1, 3, 2]))

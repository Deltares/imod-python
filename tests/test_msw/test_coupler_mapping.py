import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.testing import assert_equal

from imod import mf6, msw


def test_simple_model_with_sprinkling(fixed_format_parser):

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

    like = xr.full_like(svat.sel(subunit=1, drop=True), 1.0).expand_dims(
        layer=[1, 2, 3]
    )

    dis = mf6.StructuredDiscretization(
        top=xr.full_like(like, 1.0),
        bottom=xr.full_like(like, 0.0),
        idomain=xr.full_like(like, 1),
    )

    coupler_mapping = msw.CouplerMapping(dis, well)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        coupler_mapping.write(output_dir, index, svat)

        results = fixed_format_parser(
            output_dir / msw.CouplerMapping._file_name,
            msw.CouplerMapping._metadata_dict,
        )

    assert_equal(results["mod_id"], np.array([20, 8, 20, 14, 2, 8, 2, 5]))
    assert_equal(results["svat"], np.array([1, 2, 3, 4, 1, 2, 3, 4]))
    assert_equal(results["layer"], np.array([3, 1, 3, 2, 1, 1, 1, 1]))


def test_simple_model_with_sprinkling_1_subunit(fixed_format_parser):

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

    like = xr.full_like(svat.sel(subunit=0, drop=True), 1.0).expand_dims(
        layer=[1, 2, 3]
    )

    dis = mf6.StructuredDiscretization(
        top=xr.full_like(like, 1.0),
        bottom=xr.full_like(like, 0.0),
        idomain=xr.full_like(like, 1),
    )

    coupler_mapping = msw.CouplerMapping(dis, well)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        coupler_mapping.write(output_dir, index, svat)

        results = fixed_format_parser(
            output_dir / msw.CouplerMapping._file_name,
            msw.CouplerMapping._metadata_dict,
        )

    assert_equal(results["mod_id"], np.array([20, 17, 2, 8]))
    assert_equal(results["svat"], np.array([1, 2, 1, 2]))
    assert_equal(results["layer"], np.array([3, 2, 1, 1]))


def test_simple_model_without_sprinkling(fixed_format_parser):

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

    like = xr.full_like(svat.sel(subunit=1, drop=True), 1.0).expand_dims(
        layer=[1, 2, 3]
    )

    dis = mf6.StructuredDiscretization(
        top=xr.full_like(like, 1.0),
        bottom=xr.full_like(like, 0.0),
        idomain=xr.full_like(like, 1),
    )

    coupler_mapping = msw.CouplerMapping(dis)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        coupler_mapping.write(output_dir, index, svat)

        results = fixed_format_parser(
            output_dir / msw.CouplerMapping._file_name,
            msw.CouplerMapping._metadata_dict,
        )

    assert_equal(results["mod_id"], np.array([2, 8, 2, 5]))
    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_equal(results["layer"], np.array([1, 1, 1, 1]))

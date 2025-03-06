import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.testing import assert_equal
from pytest_cases import parametrize_with_cases

from imod import mf6, msw
from imod.mf6.mf6_wel_adapter import Mf6Wel
from imod.mf6.wel import derive_cellid_from_points


def case_svat_data():
    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
    subunit = [0, 1]
    dx = 1.0
    dy = -1.0
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
    return svat


def case_svat_data__dask():
    return case_svat_data().chunk({"x": 3, "y": 3, "subunit": 1})


def get_mf6_wel(svat_data):
    well_layer = [3, 2, 1]
    well_y = [3.0, 2.0, 1.0]
    well_x = [2.0, 2.0, 2.0]
    well_rate = [-5.0] * 3
    cellids = derive_cellid_from_points(svat_data, well_x, well_y, well_layer)
    return Mf6Wel(cellids, well_rate)


def get_mf6_dis(svat_data):
    like = xr.full_like(
        svat_data.isel(subunit=0, drop=True), 1.0, dtype=float
    ).expand_dims(layer=[1, 2, 3])
    return mf6.StructuredDiscretization(
        top=1.0,
        bottom=xr.full_like(like, 0.0),
        idomain=xr.full_like(like, 1, dtype=np.int32),
    )


def get_index(svat_data):
    return (svat_data != 0).data.ravel()


def test_clip_box():
    # Act
    copyfiles = msw.CouplerMapping()
    clipped = copyfiles.clip_box(x_min=1.0, x_max=2.5, y_min=1.0, y_max=2.5)
    # Arrange
    assert copyfiles.dataset.identical(clipped.dataset)


@parametrize_with_cases("svat_data", cases=[case_svat_data, case_svat_data__dask])
def test_simple_model_with_sprinkling(fixed_format_parser, svat_data):
    index = get_index(svat_data)
    mf6_dis = get_mf6_dis(svat_data)
    mf6_wel = get_mf6_wel(svat_data)

    coupler_mapping = msw.CouplerMapping()

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        coupler_mapping.write(output_dir, index, svat_data, mf6_dis, mf6_wel)

        results = fixed_format_parser(
            output_dir / msw.CouplerMapping._file_name,
            msw.CouplerMapping._metadata_dict,
        )

    assert_equal(results["mod_id"], np.array([20, 8, 20, 14, 2, 2, 5]))
    assert_equal(results["svat"], np.array([1, 2, 3, 4, 1, 3, 4]))
    assert_equal(results["layer"], np.array([3, 1, 3, 2, 1, 1, 1]))


@parametrize_with_cases("svat_data", cases=[case_svat_data, case_svat_data__dask])
def test_simple_model_with_sprinkling_1_subunit(fixed_format_parser, svat_data):
    svat_data = svat_data.sel(subunit=[0])
    index = get_index(svat_data)
    mf6_dis = get_mf6_dis(svat_data)
    mf6_wel = get_mf6_wel(svat_data)

    coupler_mapping = msw.CouplerMapping()

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        coupler_mapping.write(output_dir, index, svat_data, mf6_dis, mf6_wel)

        results = fixed_format_parser(
            output_dir / msw.CouplerMapping._file_name,
            msw.CouplerMapping._metadata_dict,
        )

    assert_equal(results["mod_id"], np.array([20, 8, 2]))
    assert_equal(results["svat"], np.array([1, 2, 1]))
    assert_equal(results["layer"], np.array([3, 1, 1]))


@parametrize_with_cases("svat_data", cases=[case_svat_data, case_svat_data__dask])
def test_simple_model_without_sprinkling(fixed_format_parser, svat_data):
    index = get_index(svat_data)
    mf6_dis = get_mf6_dis(svat_data)
    coupler_mapping = msw.CouplerMapping()

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        coupler_mapping.write(output_dir, index, svat_data, mf6_dis, None)

        results = fixed_format_parser(
            output_dir / msw.CouplerMapping._file_name,
            msw.CouplerMapping._metadata_dict,
        )

    assert_equal(results["mod_id"], np.array([2, 8, 2, 5]))
    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_equal(results["layer"], np.array([1, 1, 1, 1]))

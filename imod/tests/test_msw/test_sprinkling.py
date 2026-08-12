import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pytest
import xarray as xr
from numpy import nan
from numpy.testing import assert_almost_equal, assert_equal
from pytest_cases import parametrize_with_cases

from imod import msw
from imod.mf6.mf6_wel_adapter import Mf6Wel
from imod.mf6.wel import derive_cellid_from_points


@pytest.fixture(scope="function")
def sprinkling_svat_index():
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
    return svat, index


@dataclass
class AbstractionCaseData:
    max_abstraction_groundwater: Optional[xr.DataArray] = None
    max_abstraction_surfacewater: Optional[xr.DataArray] = None
    expected_abs_gw: Optional[np.ndarray] = None
    expected_abs_sw: Optional[np.ndarray] = None
    expected_layer: Optional[np.ndarray] = None
    expected_svat_gw: Optional[np.ndarray] = None


class AbstractionGrids:
    def case_all_svats(self, sprinkling_svat_index) -> AbstractionCaseData:
        svat, _ = sprinkling_svat_index
        case_data = AbstractionCaseData()
        case_data.max_abstraction_groundwater = xr.full_like(svat, 0.0)
        case_data.max_abstraction_surfacewater = xr.full_like(svat, 0.0)
        # fmt: off
        case_data.max_abstraction_groundwater.data = np.array(
            [
                [[nan, 100.0, nan],
                [nan, 200.0, nan],
                [nan, 300.0, nan]],
                [[nan, 100.0, nan],
                [nan, 200.0, nan],
                [nan, 300.0, nan]]
            ]
        )
        case_data.max_abstraction_surfacewater.data = np.array(
            [
                [[nan, 100.0, nan],
                [nan, 200.0, nan],
                [nan, 300.0, nan]],
                [[nan, 100.0, nan],
                [nan, 200.0, nan],
                [nan, 300.0, nan]]
            ]
        )
        # fmt: on
        case_data.expected_abs_gw = np.array([100.0, 300.0, 100.0, 200.0])
        case_data.expected_abs_sw = np.array([100.0, 300.0, 100.0, 200.0])
        case_data.expected_layer = np.array([3, 1, 3, 2])
        case_data.expected_svat_gw = np.array([1, 2, 3, 4])
        return case_data

    def case_some_svats(self, sprinkling_svat_index):
        svat, _ = sprinkling_svat_index
        case_data = AbstractionCaseData()
        case_data.max_abstraction_groundwater = xr.full_like(svat, 0.0)
        case_data.max_abstraction_surfacewater = xr.full_like(svat, 0.0)
        # fmt: off
        case_data.max_abstraction_groundwater.data = np.array(
            [
                [[nan, 100.0, nan],
                [nan, 200.0, nan],
                [nan, 300.0, nan]],
                [[nan, nan, nan],
                [nan, 200.0, nan],
                [nan, nan, nan]]
            ]
        )
        case_data.max_abstraction_surfacewater.data = np.array(
            [
                [[nan, 100.0, nan],
                [nan, 200.0, nan],
                [nan, 300.0, nan]],
                [[nan, nan, nan],
                [nan, 200.0, nan],
                [nan, nan, nan]]
            ]
        )
        # fmt: on
        case_data.expected_abs_gw = np.array([100.0, 300.0, 200.0])
        case_data.expected_abs_sw = np.array([100.0, 300.0, 200.0])
        case_data.expected_layer = np.array([3, 1, 2])
        case_data.expected_svat_gw = np.array([1, 2, 4])

        return case_data

    def case_inconsistent_active_capacity(self, sprinkling_svat_index):
        svat, _ = sprinkling_svat_index
        case_data = AbstractionCaseData()
        case_data.max_abstraction_groundwater = xr.full_like(svat, 0.0)
        case_data.max_abstraction_surfacewater = xr.full_like(svat, 0.0)
        # fmt: off
        case_data.max_abstraction_groundwater.data = np.array(
            [
                [[nan, 100.0, nan],
                [nan, 0.0, nan],
                [nan, 0.0, nan]],
                [[nan, nan, nan],
                [nan, 200.0, nan],
                [nan, nan, nan]]
            ]
        )
        case_data.max_abstraction_surfacewater.data = np.array(
            [
                [[nan, 0.0, nan],
                [nan, 200.0, nan],
                [nan, 300.0, nan]],
                [[nan, nan, nan],
                [nan, 200.0, nan],
                [nan, nan, nan]]
            ]
        )
        # fmt: on
        case_data.expected_abs_gw = np.array([100.0, 0.0, 200.0])
        case_data.expected_abs_sw = np.array([0.0, 300.0, 200.0])
        case_data.expected_layer = np.array([3, 1, 2])
        case_data.expected_svat_gw = np.array([1, 2, 4])

        return case_data


@parametrize_with_cases("case_data", cases=AbstractionGrids)
def test_simple_model(
    fixed_format_parser: Callable,
    sprinkling_svat_index: tuple[xr.DataArray, np.ndarray],
    case_data: AbstractionCaseData,
):
    svat, index = sprinkling_svat_index

    # Well
    well_layer = [3, 2, 1]
    well_y = [1.0, 2.0, 3.0]
    well_x = [2.0, 2.0, 2.0]
    well_rate = [-5.0] * 3
    well_id = ["a", "b", "c"]
    cellids = derive_cellid_from_points(svat, well_x, well_y, well_layer)
    well = Mf6Wel(cellids, well_rate, well_id)

    sprinkling = msw.Sprinkling(
        case_data.max_abstraction_groundwater,
        case_data.max_abstraction_surfacewater,
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        sprinkling.write(output_dir, index, svat, None, well)

        results = fixed_format_parser(
            output_dir / msw.Sprinkling._file_name,
            msw.Sprinkling._metadata_dict,
        )

    assert_equal(results["svat"], case_data.expected_svat_gw)
    assert_almost_equal(
        results["max_abstraction_groundwater"],
        case_data.expected_abs_gw,
    )
    assert_almost_equal(
        results["max_abstraction_surfacewater"],
        case_data.expected_abs_sw,
    )
    assert_equal(results["layer"], case_data.expected_layer)
    assert_equal(results["svat_groundwater"], case_data.expected_svat_gw)


def test_simple_model_1_subunit(fixed_format_parser):
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    subunit = [0]
    dx = 1.0
    dy = 1.0
    # fmt: off
    max_abstraction_groundwater = xr.DataArray(
        np.array(
            [
                [[nan, 100.0, nan],
                 [nan, 200.0, nan],
                 [nan, 300.0, nan]]
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )

    max_abstraction_surfacewater = xr.DataArray(
        np.array(
            [
                [[nan, 100.0, nan],
                 [nan, 200.0, nan],
                 [nan, 300.0, nan]]
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )

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
    well_y = [1.0, 3.0]
    well_x = [2.0, 2.0]
    well_rate = [-5.0] * 2
    well_id = ["a", "c"]
    cellids = derive_cellid_from_points(svat, well_x, well_y, well_layer)
    well = Mf6Wel(cellids, well_rate, well_id)

    sprinkling = msw.Sprinkling(
        max_abstraction_groundwater,
        max_abstraction_surfacewater,
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        sprinkling.write(output_dir, index, svat, None, well)

        results = fixed_format_parser(
            output_dir / msw.Sprinkling._file_name,
            msw.Sprinkling._metadata_dict,
        )

    assert_equal(results["svat"], np.array([1, 2]))
    assert_almost_equal(
        results["max_abstraction_groundwater"],
        np.array([100.0, 300.0]),
    )
    assert_almost_equal(
        results["max_abstraction_surfacewater"],
        np.array([100.0, 300.0]),
    )
    assert_equal(results["layer"], np.array([3, 2]))
    assert_equal(results["svat_groundwater"], np.array([1, 2]))


@pytest.mark.unittest_jit
def test_sprinkling_from_imod5_data__points(cap_data_sprinkling_points):
    with pytest.raises(NotImplementedError):
        msw.Sprinkling.from_imod5_data(cap_data_sprinkling_points)


@pytest.mark.unittest_jit
def test_sprinkling_from_imod5_data__grid(cap_data_sprinkling_grid):
    # Arrange
    # Convert unit from m3/d to mm/d
    rate = 25.0 * 0.25 * 1.0e-3
    # fmt: off
    expected_gw_abstraction = np.array(
    [[nan, rate, 0.],
     [nan, rate, 0.],
     [nan, rate, 0.]]
    )
    expected_sw_abstraction = np.array(
    [[nan, 0., rate],
     [nan, 0., rate],
     [nan, 0., rate]]
    )
    # fmt: on

    # Act
    sprinkling = msw.Sprinkling.from_imod5_data(cap_data_sprinkling_grid)

    # Assert
    assert isinstance(sprinkling, msw.Sprinkling)
    ds = sprinkling.dataset
    assert (ds.sel(subunit=1) == 0).all()
    rural_ds = ds.sel(subunit=0)
    np.testing.assert_array_equal(
        rural_ds["max_abstraction_groundwater"].to_numpy(), expected_gw_abstraction
    )
    np.testing.assert_array_equal(
        rural_ds["max_abstraction_surfacewater"].to_numpy(), expected_sw_abstraction
    )


@pytest.mark.unittest_jit
def test_sprinklingpoints_from_imod5_data__points(cap_data_sprinkling_points):
    # Arrange
    expected_vars = {"id2grid_p", "capacity_p", "layer_p", "y_p", "x_p", "id_msw"}

    # Act
    sprinkling = msw.SprinklingPoints.from_imod5_data(cap_data_sprinkling_points)

    # Assert
    assert sprinkling.dataset.sizes == {"id": 2, "x": 3, "y": 3}
    assert set(sprinkling.dataset.keys()) == expected_vars
    # No unit conversion is done in SprinklingPoints, as the capacity is already
    # in m3/d
    np.testing.assert_almost_equal(sprinkling.dataset["capacity_p"].max(), 100.0)


# @pytest.mark.unittest_jit
# def test_sprinklingpoints_write__points(cap_data_sprinkling_points, tmp_path):
#    well_x = cap_data_sprinkling_points["cap"]["x"].values
#    well_y = cap_data_sprinkling_points["cap"]["y"].values
#    well_layer = cap_data_sprinkling_points["cap"]["layer"].values
#
#    # Arrange
#    # cellids = derive_cellid_from_points(svat, well_x, well_y, well_layer)
#    # well = Mf6Wel(cellids, well_rate)
#
#    # Act
#    sprinkling = msw.SprinklingPoints.from_imod5_data(cap_data_sprinkling_points)
#
#    # sprinkling.write()
#

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
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy},
        name="svat",
    )
    # fmt: on
    index = (svat != 0).values.ravel()
    return svat, index

@dataclass
class ExpectedCaseData:
    abs_gw: Optional[np.ndarray] = None
    abs_sw: Optional[np.ndarray] = None
    layer: Optional[np.ndarray] = None
    svat: Optional[np.ndarray] = None
    svat_gw: Optional[np.ndarray] = None


@dataclass
class SprinklingGridCaseData:
    max_abstraction_groundwater: Optional[xr.DataArray] = None
    max_abstraction_surfacewater: Optional[xr.DataArray] = None

@dataclass
class SprinklingPointsCaseData:
    art_grid: Optional[xr.DataArray] = None
    x_p: Optional[xr.DataArray] = None
    y_p: Optional[xr.DataArray] = None
    layer_p: Optional[xr.DataArray] = None
    id2grid_p: Optional[xr.DataArray] = None
    capacity_p: Optional[xr.DataArray] = None


class SprinklingGridCases:
    def case_all_svats(self, sprinkling_svat_index) -> tuple[SprinklingGridCaseData, ExpectedCaseData]:
        svat, _ = sprinkling_svat_index
        case_data = SprinklingGridCaseData()
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
        expected_data = ExpectedCaseData()
        expected_data.abs_gw = np.array([100.0, 300.0, 100.0, 200.0])
        expected_data.abs_sw = np.array([100.0, 300.0, 100.0, 200.0])
        expected_data.layer = np.array([3, 1, 3, 2])
        expected_data.svat = np.array([1, 2, 3, 4])
        expected_data.svat_gw = np.array([1, 2, 3, 4])
        case_data.expected_data = expected_data
        return case_data, expected_data

    def case_some_svats(self, sprinkling_svat_index) -> tuple[SprinklingGridCaseData, ExpectedCaseData]:
        svat, _ = sprinkling_svat_index
        case_data = SprinklingGridCaseData()
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
        expected_data = ExpectedCaseData()
        expected_data.abs_gw = np.array([100.0, 300.0, 200.0])
        expected_data.abs_sw = np.array([100.0, 300.0, 200.0])
        expected_data.layer = np.array([3, 1, 2])
        expected_data.svat = np.array([1, 2, 4])
        expected_data.svat_gw = np.array([1, 2, 4])

        return case_data, expected_data

    def case_inconsistent_active_capacity(self, sprinkling_svat_index) -> tuple[SprinklingGridCaseData, ExpectedCaseData]:
        svat, _ = sprinkling_svat_index
        case_data = SprinklingGridCaseData()
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
        expected_data = ExpectedCaseData()
        expected_data.abs_gw = np.array([100.0, 0.0, 200.0])
        expected_data.abs_sw = np.array([0.0, 300.0, 200.0])
        expected_data.layer = np.array([3, 1, 2])
        expected_data.svat = np.array([1, 2, 4])
        expected_data.svat_gw = np.array([1, 2, 4])

        return case_data, expected_data


class SprinklingPointsCases:
    def case_simple(self, sprinkling_svat_index):
        svat, _ = sprinkling_svat_index
        case_data = SprinklingPointsCaseData()



@parametrize_with_cases("case_data, expected_data", cases=SprinklingGridCases)
def test_simple_model(
    fixed_format_parser: Callable,
    sprinkling_svat_index: tuple[xr.DataArray, np.ndarray],
    case_data: SprinklingGridCaseData,
    expected_data: ExpectedCaseData,
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

    assert_equal(results["svat"], expected_data.svat)
    assert_almost_equal(
        results["max_abstraction_groundwater"],
        expected_data.abs_gw,
    )
    assert_almost_equal(
        results["max_abstraction_surfacewater"],
        expected_data.abs_sw,
    )
    assert_equal(results["layer"], expected_data.layer)
    assert_equal(results["svat_groundwater"], expected_data.svat_gw)


@parametrize_with_cases("case_data, expected_data", cases=SprinklingGridCases)
def test_simple_model_1_subunit(
    fixed_format_parser: Callable,
    sprinkling_svat_index: tuple[xr.DataArray, np.ndarray],
    case_data: SprinklingGridCaseData,
    expected_data: ExpectedCaseData,
):
    svat, index = sprinkling_svat_index

    svat = svat.isel(subunit=[0])
    index = index[:9]  # Only the first subunit

    # Well
    well_layer = [3, 1]
    well_y = [1.0, 3.0]
    well_x = [2.0, 2.0]
    well_rate = [-5.0] * 2
    well_id = ["a", "c"]
    cellids = derive_cellid_from_points(svat, well_x, well_y, well_layer)
    well = Mf6Wel(cellids, well_rate, well_id)

    sprinkling = msw.Sprinkling(
        case_data.max_abstraction_groundwater.isel(subunit=[0]),
        case_data.max_abstraction_surfacewater.isel(subunit=[0]),
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        sprinkling.write(output_dir, index, svat, None, well)

        results = fixed_format_parser(
            output_dir / msw.Sprinkling._file_name,
            msw.Sprinkling._metadata_dict,
        )

    assert_equal(results["svat"], expected_data.svat[:2])
    assert_almost_equal(
        results["max_abstraction_groundwater"],
        expected_data.abs_gw[:2],
    )
    assert_almost_equal(
        results["max_abstraction_surfacewater"],
        expected_data.abs_sw[:2],
    )
    assert_equal(results["layer"], expected_data.layer[:2])
    assert_equal(results["svat_groundwater"], expected_data.svat_gw[:2])


@pytest.mark.unittest_jit
def test_sprinkling_from_imod5_data__points(cap_data_sprinkling_points):
    with pytest.raises(TypeError):
        msw.Sprinkling.from_imod5_data(cap_data_sprinkling_points)


@pytest.mark.unittest_jit
def test_sprinklingpoints_from_imod5_data__grid(cap_data_sprinkling_grid):
    with pytest.raises(TypeError):
        msw.SprinklingPoints.from_imod5_data(cap_data_sprinkling_grid)


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


# TODO: Create more test cases for SprinklingPoints.write() to test edge cases,
#   such as wells in inactive SVATs, wells outside the model domain, etc.



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
    np.testing.assert_almost_equal(sprinkling.dataset["capacity_p"], [15.0, 30.0])


@pytest.mark.unittest_jit
def test_sprinklingpoints_from_imod5_data_write__points(
    sprinkling_svat_index,
    fixed_format_parser,
    cap_data_sprinkling_points,
    cap_coupled_dis_grid,
    tmp_path,
):
    """
    Test with two wells: one inside the active SVAT area, and one outside the
    active SVAT area but still in the model domain. Well nr 2. is not assigned
    to anything. Well nr 1. is assigned and is located in the centre cell. In
    subunit 1 this cell is inactive and the svats coupled to this well are
    assigned to surface water, in subunit 2 this cell is active and this well is
    coupled to the groundwater svat.
    """
    # Arrange
    svat, index = sprinkling_svat_index
    df = cap_data_sprinkling_points["cap"]["artificial_recharge_layer"]
    well_x = df["x"].to_numpy()
    well_y = df["y"].to_numpy()
    well_layer = df["layer"].to_numpy()
    well_rate = xr.DataArray([0.0, 0.0], dims=("ncellid",))
    well_id = xr.DataArray(["0", "1"], dims=("ncellid",))
    cellids = derive_cellid_from_points(svat, well_x, well_y, well_layer)
    mf6_well = Mf6Wel(cellids, well_rate, well_id)
    mf6_dis = cap_coupled_dis_grid
    directory = tmp_path / "sprinkling_points"
    directory.mkdir(parents=True, exist_ok=True)

    # Act
    sprinkling = msw.SprinklingPoints.from_imod5_data(cap_data_sprinkling_points)
    sprinkling.write(directory, index, svat, mf6_dis, mf6_well)

    results = fixed_format_parser(
        directory / msw.Sprinkling._file_name,
        msw.Sprinkling._metadata_dict,
    )

    # Assert
    # TODO: Check with Hendrik whether this is the appropriate behaviour.
    np.testing.assert_equal(results["svat"], [1, 2, 3, 4])
    np.testing.assert_equal(results["svat_groundwater"], [1, 2, 4, 4])
    np.testing.assert_equal(results["layer"], [2, 2, 2, 2])
    np.testing.assert_equal(
        results["max_abstraction_surfacewater"], [15.0, 15.0, 0.0, 0.0]
    )
    np.testing.assert_equal(
        results["max_abstraction_groundwater"], [0.0, 0.0, 15.0, 15.0]
    )

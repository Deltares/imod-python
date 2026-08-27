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
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.mf6_wel_adapter import Mf6Wel
from imod.mf6.wel import derive_cellid_from_points


@pytest.fixture(scope="function")
def sprinkling_svat_index():
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
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy},
        name="svat",
    )
    # fmt: on
    index = (svat != 0).values.ravel()
    return svat, index


@dataclass
class ExpectedCaseData:
    xfail: Optional[str] = None
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
    x_p: Optional[np.ndarray] = None
    y_p: Optional[np.ndarray] = None
    layer_p: Optional[np.ndarray] = None
    id_sprinkling_p: Optional[np.ndarray] = None
    capacity_p: Optional[np.ndarray] = None


class SprinklingGridCases:
    def case_all_svats(
        self, sprinkling_svat_index
    ) -> tuple[SprinklingGridCaseData, ExpectedCaseData]:
        """
        Case where all SVATs (1,2,3,4) have sprinkling cells.
        """
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

    def case_some_svats(
        self, sprinkling_svat_index
    ) -> tuple[SprinklingGridCaseData, ExpectedCaseData]:
        """
        Case where only some SVATs (1,2,4) have sprinkling cells.
        """
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

    def case_inconsistent_active_capacity(
        self, sprinkling_svat_index
    ) -> tuple[SprinklingGridCaseData, ExpectedCaseData]:
        """
        Case where the active capacity is inconsistent between groundwater and
        surfacewater.
        """
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
    def case_one_point_one_art_cell(
        self, sprinkling_svat_index
    ) -> tuple[SprinklingPointsCaseData, ExpectedCaseData]:
        """
        Simple test case for sprinkling points. Each point is mapped to one svat.
        """
        svat, _ = sprinkling_svat_index
        case_data = SprinklingPointsCaseData()
        case_data.art_grid = xr.full_like(svat, 0, dtype=int)
        # fmt: off
        case_data.art_grid.data = np.array(
            [[[0, 1, 0],
              [0, 2, 0],
              [0, 3, 0],],
             [[0, 1, 0],
              [0, 2, 0],
              [0, 3, 0]]]
        )
        # fmt: on
        case_data.x_p = [2.0, 2.0, 2.0]
        case_data.y_p = [3.0, 2.0, 1.0]
        case_data.layer_p = [1, 2, 3]
        case_data.id_sprinkling_p = [1, 2, 3]
        case_data.capacity_p = [10.0, 20.0, 30.0]

        expected_data = ExpectedCaseData()
        expected_data.svat = np.array([1, 2, 3, 4])
        expected_data.svat_gw = np.array([1, 2, 3, 4])
        expected_data.layer = np.array([1, 3, 1, 2])
        expected_data.abs_gw = np.array([10.0, 30.0, 10.0, 20.0])
        expected_data.abs_sw = np.array([0.0, 0.0, 0.0, 0.0])

        return case_data, expected_data

    def case_one_point_one_art_cell_layer0(
        self, sprinkling_svat_index
    ) -> tuple[SprinklingPointsCaseData, ExpectedCaseData]:
        """
        Simple test case for sprinkling points but now with layer 0. All
        extractions should be assigned to surface water extractions.

        Note that even though the input layer is 0, the expected output layer is
        set to 1 as this is required by MetaSWAP.
        """
        svat, _ = sprinkling_svat_index
        case_data = SprinklingPointsCaseData()
        case_data.art_grid = xr.full_like(svat, 0, dtype=int)
        # fmt: off
        case_data.art_grid.data = np.array(
            [[[0, 1, 0],
              [0, 2, 0],
              [0, 3, 0],],
             [[0, 1, 0],
              [0, 2, 0],
              [0, 3, 0]]]
        )
        # fmt: on
        case_data.x_p = [2.0, 2.0, 2.0]
        case_data.y_p = [3.0, 2.0, 1.0]
        case_data.layer_p = [0, 0, 0]
        case_data.id_sprinkling_p = [1, 2, 3]
        case_data.capacity_p = [10.0, 20.0, 30.0]

        expected_data = ExpectedCaseData()
        expected_data.svat = np.array([1, 2, 3, 4])
        expected_data.svat_gw = np.array([1, 2, 3, 4])
        expected_data.layer = np.array([1, 1, 1, 1])
        expected_data.abs_gw = np.array([0.0, 0.0, 0.0, 0.0])
        expected_data.abs_sw = np.array([10.0, 30.0, 10.0, 20.0])

        return case_data, expected_data

    def case_one_point_one_art_cell__one_subunit(
        self, sprinkling_svat_index
    ) -> tuple[SprinklingPointsCaseData, ExpectedCaseData]:
        """
        Simple test case for sprinkling points. Each point is mapped to one
        svat. Only one subunit is used, similar to when imported from iMOD5
        DBASE
        """
        svat, _ = sprinkling_svat_index
        case_data = SprinklingPointsCaseData()
        case_data.art_grid = xr.full_like(svat, 0, dtype=int)
        # fmt: off
        case_data.art_grid.data = np.array(
            [[[0, 1, 0],
              [0, 2, 0],
              [0, 3, 0],],
             [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]]
        )
        # fmt: on
        case_data.x_p = [2.0, 2.0, 2.0]
        case_data.y_p = [3.0, 2.0, 1.0]
        case_data.layer_p = [1, 2, 3]
        case_data.id_sprinkling_p = [1, 2, 3]
        case_data.capacity_p = [10.0, 20.0, 30.0]

        expected_data = ExpectedCaseData()
        expected_data.svat = np.array([1, 2])
        expected_data.svat_gw = np.array([1, 2])
        expected_data.layer = np.array([1, 3])
        expected_data.abs_gw = np.array([10.0, 30.0])
        expected_data.abs_sw = np.array([0.0, 0.0])

        return case_data, expected_data

    def case_multi_point_one_art_cell(
        self, sprinkling_svat_index
    ) -> tuple[SprinklingPointsCaseData, ExpectedCaseData]:
        """
        Case where multiple points are assigned to the same SVAT. Not a common
        usecase. Usually multiple cells coupled to one point.
        """
        svat, _ = sprinkling_svat_index
        case_data = SprinklingPointsCaseData()
        case_data.art_grid = xr.full_like(svat, 0, dtype=int)
        # fmt: off
        case_data.art_grid.data = np.array(
            [[[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]],
             [[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]]]
        )
        # fmt: on
        case_data.x_p = [2.0, 2.0, 2.0]
        case_data.y_p = [3.0, 2.0, 1.0]
        case_data.layer_p = [1, 2, 3]
        case_data.id_sprinkling_p = [1, 1, 1]
        case_data.capacity_p = [10.0, 20.0, 30.0]

        expected_data = ExpectedCaseData()
        expected_data.xfail = "Multiple points cannot be connected to one grid cell"
        return case_data, expected_data

    def case_one_point_multi_art_cell(
        self, sprinkling_svat_index
    ) -> tuple[SprinklingPointsCaseData, ExpectedCaseData]:
        """
        Case where one point is assigned to multiple art_grid cells. Quite a
        common usecase. The point is located in the centre of the grid, where
        there is only an svat in subunit 1. In subunit 0 this cell is not
        active, therefore sprinkling capacity is assigned to surface water.
        """
        svat, _ = sprinkling_svat_index
        case_data = SprinklingPointsCaseData()
        case_data.art_grid = xr.full_like(svat, 0, dtype=int)
        # fmt: off
        case_data.art_grid.data = np.array(
            [[[0, 1, 0],
              [0, 1, 0],
              [0, 1, 0]],
             [[0, 1, 0],
              [0, 1, 0],
              [0, 1, 0]]]
        )
        # fmt: on
        case_data.x_p = [2.0]
        case_data.y_p = [2.0]
        case_data.layer_p = [2]
        case_data.id_sprinkling_p = [1]
        case_data.capacity_p = [10.0]

        expected_data = ExpectedCaseData()
        expected_data.svat = np.array([1, 2, 3, 4])
        expected_data.svat_gw = np.array([1, 2, 4, 4])
        expected_data.layer = np.array([1, 1, 2, 2])
        expected_data.abs_gw = np.array([0.0, 0.0, 10.0, 10.0])
        expected_data.abs_sw = np.array([10.0, 10.0, 0.0, 0.0])

        return case_data, expected_data

    def case_one_point_multi_art_cell_layer0(
        self, sprinkling_svat_index
    ) -> tuple[SprinklingPointsCaseData, ExpectedCaseData]:
        """
        Case where one point is assigned to multiple art_grid cells but now with
        layer 0. This should all be assigned as surface water and svat should be
        equal to groundwater svats.

        Note that even though the input layer is 0, the expected output layer is
        set to 1 as this is required by MetaSWAP.
        """
        svat, _ = sprinkling_svat_index
        case_data = SprinklingPointsCaseData()
        case_data.art_grid = xr.full_like(svat, 0, dtype=int)
        # fmt: off
        case_data.art_grid.data = np.array(
            [[[0, 1, 0],
              [0, 1, 0],
              [0, 1, 0]],
             [[0, 1, 0],
              [0, 1, 0],
              [0, 1, 0]]]
        )
        # fmt: on
        case_data.x_p = [2.0]
        case_data.y_p = [2.0]
        case_data.layer_p = [0]
        case_data.id_sprinkling_p = [1]
        case_data.capacity_p = [10.0]

        expected_data = ExpectedCaseData()
        expected_data.svat = np.array([1, 2, 3, 4])
        expected_data.svat_gw = np.array([1, 2, 3, 4])
        expected_data.layer = np.array([1, 1, 1, 1])
        expected_data.abs_gw = np.array([0.0, 0.0, 0.0, 0.0])
        expected_data.abs_sw = np.array([10.0, 10.0, 10.0, 10.0])

        return case_data, expected_data

    def case_art_grid_outside(
        self, sprinkling_svat_index
    ) -> tuple[SprinklingPointsCaseData, ExpectedCaseData]:
        """
        Case where art grid is located outside the active SVAT area, but still
        in the model domain. The well is inside the model domain. Sprinkling
        should not be assigned.
        """
        svat, _ = sprinkling_svat_index
        case_data = SprinklingPointsCaseData()
        case_data.art_grid = xr.full_like(svat, 0, dtype=int)
        # fmt: off
        case_data.art_grid.data = np.array(
            [[[0, 0, 4],
              [0, 0, 0],
              [0, 0, 0]],
             [[0, 0, 4],
              [0, 0, 0],
              [0, 0, 0]]]
        )

        # fmt: on
        case_data.x_p = [2.0]
        case_data.y_p = [2.0]
        case_data.layer_p = [3]
        case_data.id_sprinkling_p = [4]
        case_data.capacity_p = [40.0]

        expected_data = ExpectedCaseData()
        expected_data.svat = np.array([])
        expected_data.svat_gw = np.array([])
        expected_data.layer = np.array([])
        expected_data.abs_gw = np.array([])
        expected_data.abs_sw = np.array([])

        return case_data, expected_data

    def case_point_outside(
        self, sprinkling_svat_index
    ) -> tuple[SprinklingPointsCaseData, ExpectedCaseData]:
        """
        Case where one point is located outside the active SVAT area, but still in
        the model domain. The well should be assigned as surface water extraction.
        """
        svat, _ = sprinkling_svat_index
        case_data = SprinklingPointsCaseData()
        case_data.art_grid = xr.full_like(svat, 0, dtype=int)
        # fmt: off
        case_data.art_grid.data = np.array(
            [[[0, 0, 0],
              [0, 0, 0],
              [0, 5, 0]],
             [[0, 0, 0],
              [0, 0, 0],
              [0, 5, 0]]]
        )
        # fmt: on
        case_data.x_p = [3.0]
        case_data.y_p = [1.0]
        case_data.layer_p = [3]
        case_data.id_sprinkling_p = [5]
        case_data.capacity_p = [40.0]

        expected_data = ExpectedCaseData()
        expected_data.svat = np.array([2])
        expected_data.svat_gw = np.array([2])
        expected_data.layer = np.array([1])
        expected_data.abs_gw = np.array([0.0])
        expected_data.abs_sw = np.array([40.0])
        return case_data, expected_data


@parametrize_with_cases("case_data, expected_data", cases=SprinklingGridCases)
def test_grid_simple_model(
    fixed_format_parser: Callable,
    sprinkling_svat_index: tuple[xr.DataArray, np.ndarray],
    case_data: SprinklingGridCaseData,
    expected_data: ExpectedCaseData,
):
    svat, index = sprinkling_svat_index

    # Well
    well_layer = [3, 2, 1]
    well_y = [3.0, 2.0, 1.0]
    well_x = [2.0, 2.0, 2.0]
    well_rate_values = [-5.0] * 3
    well_rate = xr.DataArray(well_rate_values, dims=("ncellid",))
    well_id_values = ["0", "1", "2"]
    well_id = xr.DataArray(well_id_values, dims=("ncellid",))
    cellids = derive_cellid_from_points(svat, well_x, well_y, well_layer)
    well = Mf6Wel(cellids, well_rate, well_id)

    sprinkling = msw.SprinklingGrid(
        case_data.max_abstraction_groundwater,
        case_data.max_abstraction_surfacewater,
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        sprinkling.write(output_dir, index, svat, None, well)

        results = fixed_format_parser(
            output_dir / msw.SprinklingGrid._file_name,
            msw.SprinklingGrid._metadata_dict,
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
def test_grid_simple_model_1_subunit(
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
    well_y = [3.0, 1.0]
    well_x = [2.0, 2.0]
    well_rate_values = [-5.0] * 2
    well_rate = xr.DataArray(well_rate_values, dims=("ncellid",))
    well_id_values = ["0", "2"]
    well_id = xr.DataArray(well_id_values, dims=("ncellid",))
    cellids = derive_cellid_from_points(svat, well_x, well_y, well_layer)
    well = Mf6Wel(cellids, well_rate, well_id)

    sprinkling = msw.SprinklingGrid(
        case_data.max_abstraction_groundwater.isel(subunit=[0]),
        case_data.max_abstraction_surfacewater.isel(subunit=[0]),
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        sprinkling.write(output_dir, index, svat, None, well)

        results = fixed_format_parser(
            output_dir / msw.SprinklingGrid._file_name,
            msw.SprinklingGrid._metadata_dict,
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


@parametrize_with_cases("case_data, expected_data", cases=SprinklingPointsCases)
def test_points_simple_model(
    fixed_format_parser: Callable,
    sprinkling_svat_index: tuple[xr.DataArray, np.ndarray],
    case_data: SprinklingPointsCaseData,
    expected_data: ExpectedCaseData,
):
    if expected_data.xfail:
        pytest.xfail(expected_data.xfail)
    svat, index = sprinkling_svat_index

    # Well
    n_wells = len(case_data.x_p)
    well_rate_values = [-5.0] * n_wells
    well_rate = xr.DataArray(well_rate_values, dims=("ncellid",))
    well_id_values = [str(i) for i in np.arange(n_wells)]
    well_id = xr.DataArray(well_id_values, dims=("ncellid",))

    cellids = derive_cellid_from_points(
        svat, case_data.x_p, case_data.y_p, case_data.layer_p
    )
    well = Mf6Wel(cellids, well_rate, well_id)

    layer_template = xr.DataArray(
        [1.0, 2.0, 3.0], coords={"layer": [1, 2, 3]}, dims=("layer",)
    )
    grid_2d_template = xr.ones_like(svat.isel(subunit=0, drop=True), dtype=float)
    mf6_dis_template = layer_template * grid_2d_template

    dis = StructuredDiscretization(
        top=grid_2d_template,
        bottom=-mf6_dis_template,
        idomain=mf6_dis_template.astype(int),
    )

    sprinkling = msw.SprinklingPoints(
        case_data.art_grid,
        case_data.x_p,
        case_data.y_p,
        case_data.layer_p,
        case_data.id_sprinkling_p,
        case_data.capacity_p,
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        sprinkling.write(output_dir, index, svat, dis, well)

        results = fixed_format_parser(
            output_dir / msw.SprinklingPoints._file_name,
            msw.SprinklingPoints._metadata_dict,
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


@pytest.mark.unittest_jit
def test_sprinkling_from_imod5_data__points(cap_data_sprinkling_points):
    with pytest.raises(TypeError):
        msw.SprinklingGrid.from_imod5_data(cap_data_sprinkling_points)


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
    sprinkling = msw.SprinklingGrid.from_imod5_data(cap_data_sprinkling_grid)

    # Assert
    assert isinstance(sprinkling, msw.SprinklingGrid)
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
    expected_vars = {
        "id_sprinkling_p",
        "capacity_p",
        "layer_p",
        "y_p",
        "x_p",
        "id_sprinkling",
    }

    # Act
    sprinkling = msw.SprinklingPoints.from_imod5_data(cap_data_sprinkling_points)

    # Assert
    assert sprinkling.dataset.sizes == {"subunit": 2, "id": 2, "x": 3, "y": 3}
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
        directory / msw.SprinklingGrid._file_name,
        msw.SprinklingGrid._metadata_dict,
    )

    # Assert
    np.testing.assert_equal(results["svat"], [1, 2])
    np.testing.assert_equal(results["svat_groundwater"], [1, 2])
    np.testing.assert_equal(results["layer"], [2, 1])
    np.testing.assert_equal(results["max_abstraction_surfacewater"], [0.0, 30.0])
    np.testing.assert_equal(results["max_abstraction_groundwater"], [15.0, 0.0])

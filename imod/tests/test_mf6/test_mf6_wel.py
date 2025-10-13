import sys
from contextlib import nullcontext as does_not_raise
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xugrid as xu
from pytest_cases import parametrize, parametrize_with_cases

import imod
from imod.common.utilities.grid import broadcast_to_full_domain
from imod.formats.prj.prj import open_projectfile_data
from imod.logging.config import LoggerType
from imod.logging.loglevel import LogLevel
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.npf import NodePropertyFlow
from imod.mf6.wel import LayeredWell, Well, derive_cellid_from_points
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError
from imod.tests.fixtures.flow_basic_fixture import BasicDisSettings
from imod.typing.grid import ones_like

times = [
    datetime(1981, 11, 30),
    datetime(1981, 12, 31),
    datetime(1982, 1, 31),
    datetime(1982, 2, 28),
    datetime(1982, 3, 31),
    datetime(1982, 4, 30),
]


class BasicGridCases:
    @staticmethod
    def case_structured(basic_dis):
        idomain, top, bottom = basic_dis
        return (
            imod.mf6.StructuredDiscretization,
            idomain,
            top,
            bottom,
        )

    @staticmethod
    def case_unstructured(basic_unstructured_dis):
        idomain, top, bottom = basic_unstructured_dis
        return imod.mf6.VerticesDiscretization, idomain, top, bottom


class GridAgnosticWellCases:
    def case_well_stationary(self, well_high_lvl_test_data_stationary):
        obj = imod.mf6.Well(*well_high_lvl_test_data_stationary)
        sizes_expected = {
            "ncellid": 12,
            "dim_cellid": 3,
            "species": 2,
        }
        cellid_expected = np.array(
            [
                [1, 1, 9],
                [1, 2, 9],
                [1, 3, 9],
                [1, 1, 8],
                [1, 2, 8],
                [1, 3, 8],
                [2, 4, 7],
                [2, 5, 7],
                [2, 6, 7],
                [2, 4, 6],
                [2, 5, 6],
                [2, 6, 6],
            ],
            dtype=np.int64,
        )
        rate_expected = np.array(np.ones((12,), dtype=np.float32))
        return obj, sizes_expected, cellid_expected, rate_expected

    def case_well_stationary_multilevel(self, well_high_lvl_test_data_stationary):
        x, y, screen_top, _, rate_wel, concentration = (
            well_high_lvl_test_data_stationary
        )
        screen_bottom = [-20.0] * 12
        obj = imod.mf6.Well(x, y, screen_top, screen_bottom, rate_wel, concentration)
        sizes_expected = {
            "ncellid": 18,
            "dim_cellid": 3,
            "species": 2,
        }
        cellid_expected = np.array(
            [
                [1, 1, 9],
                [1, 2, 9],
                [1, 3, 9],
                [1, 1, 8],
                [1, 2, 8],
                [1, 3, 8],
                [2, 1, 9],
                [2, 2, 9],
                [2, 3, 9],
                [2, 1, 8],
                [2, 2, 8],
                [2, 3, 8],
                [2, 4, 7],
                [2, 5, 7],
                [2, 6, 7],
                [2, 4, 6],
                [2, 5, 6],
                [2, 6, 6],
            ],
            dtype=np.int64,
        )
        rate_expected = np.array([0.25] * 6 + [0.75] * 6 + [1.0] * 6)
        return obj, sizes_expected, cellid_expected, rate_expected

    def case_well_point_filter(self, well_high_lvl_test_data_stationary):
        x, y, screen_point, _, rate_wel, concentration = (
            well_high_lvl_test_data_stationary
        )
        obj = imod.mf6.Well(x, y, screen_point, screen_point, rate_wel, concentration)
        sizes_expected = {
            "ncellid": 12,
            "dim_cellid": 3,
            "species": 2,
        }
        cellid_expected = np.array(
            [
                [1, 1, 9],
                [1, 2, 9],
                [1, 3, 9],
                [1, 1, 8],
                [1, 2, 8],
                [1, 3, 8],
                [2, 4, 7],
                [2, 5, 7],
                [2, 6, 7],
                [2, 4, 6],
                [2, 5, 6],
                [2, 6, 6],
            ],
            dtype=np.int64,
        )
        rate_expected = np.array(np.ones((12,), dtype=np.float32))
        return obj, sizes_expected, cellid_expected, rate_expected

    def case_well_transient(self, well_high_lvl_test_data_transient):
        obj = imod.mf6.Well(*well_high_lvl_test_data_transient)
        sizes_expected = {
            "ncellid": 12,
            "time": 5,
            "dim_cellid": 3,
            "species": 2,
        }
        cellid_expected = np.array(
            [
                [1, 1, 9],
                [1, 2, 9],
                [1, 3, 9],
                [1, 1, 8],
                [1, 2, 8],
                [1, 3, 8],
                [2, 4, 7],
                [2, 5, 7],
                [2, 6, 7],
                [2, 4, 6],
                [2, 5, 6],
                [2, 6, 6],
            ],
            dtype=np.int64,
        )
        rate_expected = np.outer(np.ones((12,), dtype=np.float32), np.arange(5) + 1)
        return obj, sizes_expected, cellid_expected, rate_expected

    def case_layered_well_stationary(self, well_high_lvl_test_data_stationary):
        x, y, _, _, rate_wel, concentration = well_high_lvl_test_data_stationary
        layer = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        obj = imod.mf6.LayeredWell(x, y, layer, rate_wel, concentration)
        sizes_expected = {
            "ncellid": 12,
            "dim_cellid": 3,
            "species": 2,
        }
        cellid_expected = np.array(
            [
                [1, 1, 9],
                [1, 2, 9],
                [1, 3, 9],
                [1, 1, 8],
                [1, 2, 8],
                [1, 3, 8],
                [2, 4, 7],
                [2, 5, 7],
                [2, 6, 7],
                [2, 4, 6],
                [2, 5, 6],
                [2, 6, 6],
            ],
            dtype=np.int64,
        )
        rate_expected = np.array(np.ones((12,), dtype=np.float32))
        return obj, sizes_expected, cellid_expected, rate_expected

    def case_layered_well_stationary_with_id_and_cleanup(
        self, basic_dis, well_high_lvl_test_data_stationary
    ):
        x, y, _, _, rate_wel, concentration = well_high_lvl_test_data_stationary
        layer = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        id = ["12", "11", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]
        rate = [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        rate_wel = xr.DataArray(rate, dims=("index",))
        wel = imod.mf6.LayeredWell(x, y, layer, rate_wel, concentration, id=id)

        idomain, top, bottom = basic_dis
        top = top.isel(layer=0, drop=True)
        dis_normal = imod.mf6.StructuredDiscretization(top, bottom, idomain)
        wel.cleanup(dis_normal)

        sizes_expected = {
            "ncellid": 12,
            "dim_cellid": 3,
            "species": 2,
        }
        cellid_expected = np.array(
            [
                [1, 1, 9],
                [1, 2, 9],
                [1, 3, 9],
                [1, 1, 8],
                [1, 2, 8],
                [1, 3, 8],
                [2, 4, 7],
                [2, 5, 7],
                [2, 6, 7],
                [2, 4, 6],
                [2, 5, 6],
                [2, 6, 6],
            ],
            dtype=np.int64,
        )
        rate_expected = np.array(rate)
        return wel, sizes_expected, cellid_expected, rate_expected

    def case_layered_well_transient(self, well_high_lvl_test_data_transient):
        x, y, _, _, rate_wel, concentration = well_high_lvl_test_data_transient
        layer = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        obj = imod.mf6.LayeredWell(x, y, layer, rate_wel, concentration)
        sizes_expected = {
            "ncellid": 12,
            "time": 5,
            "dim_cellid": 3,
            "species": 2,
        }
        cellid_expected = np.array(
            [
                [1, 1, 9],
                [1, 2, 9],
                [1, 3, 9],
                [1, 1, 8],
                [1, 2, 8],
                [1, 3, 8],
                [2, 4, 7],
                [2, 5, 7],
                [2, 6, 7],
                [2, 4, 6],
                [2, 5, 6],
                [2, 6, 6],
            ],
            dtype=np.int64,
        )
        rate_expected = np.outer(np.ones((12,), dtype=np.float32), np.arange(5) + 1)
        return obj, sizes_expected, cellid_expected, rate_expected


@parametrize_with_cases(
    ["wel", "sizes_expected", "cellid_expected", "rate_expected"],
    cases=GridAgnosticWellCases,
)
def test_to_mf6_pkg(basic_dis, wel, sizes_expected, cellid_expected, rate_expected):
    # Arrange
    idomain, top, bottom = basic_dis
    active = idomain == 1
    k = xr.ones_like(idomain)

    dim_cellid_expected = np.array(["layer", "row", "column"])

    # Act
    mf6_wel = wel.to_mf6_pkg(active, top, bottom, k)
    mf6_ds = mf6_wel.dataset

    # Assert
    assert dict(mf6_ds.sizes) == sizes_expected
    np.testing.assert_equal(mf6_ds.coords["dim_cellid"].values, dim_cellid_expected)
    np.testing.assert_equal(mf6_ds["cellid"].values, cellid_expected)
    np.testing.assert_equal(mf6_ds["rate"].values, rate_expected)


def test_to_mf6_pkg__validate(well_high_lvl_test_data_stationary):
    # Arrange
    wel = imod.mf6.Well(*well_high_lvl_test_data_stationary)

    # Act
    errors = wel._validate(wel._write_schemata)
    assert len(errors) == 0

    # Set rates with index exceeding 3 to NaN.
    wel.dataset["rate"] = wel.dataset["rate"].where(wel.dataset.coords["index"] < 3)
    errors = wel._validate(wel._write_schemata)
    assert len(errors) == 1


def test_to_mf6_pkg__validate_filter_top(well_high_lvl_test_data_stationary):
    # Arrange
    x, y, screen_top, screen_bottom, rate_wel, concentration = (
        well_high_lvl_test_data_stationary
    )
    screen_top = [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    screen_bottom = [
        0.0,
        -2.0,
        -2.0,
        -2.0,
        -2.0,
        -2.0,
        -2.0,
        -2.0,
        -2.0,
        -2.0,
        -2.0,
        -2.0,
    ]
    well_test_data = (x, y, screen_top, screen_bottom, rate_wel, concentration)

    wel = imod.mf6.Well(*well_test_data)

    # Act
    kwargs = {"screen_top": wel.dataset["screen_top"]}
    errors = wel._validate(wel._write_schemata, **kwargs)

    # Assert
    assert len(errors) == 1
    assert (
        str(errors["screen_bottom"][0])
        == "not all values comply with criterion: <= screen_top"
    )


@parametrize_with_cases("dis", cases=BasicGridCases)
def test_to_mf6_pkg__logging_with_message(
    tmp_path, dis, well_high_lvl_test_data_transient
):
    # Arrange
    logfile_path = tmp_path / "logfile.txt"
    _, idomain, top, bottom = dis
    modified_well_fixture = list(well_high_lvl_test_data_transient)

    # create an idomain where layer 1 is active and layer 2 and 3 are inactive.
    # layer 1 has a bottom at -5, layer 2 at -35 and layer 3 at -135
    # so only wells that have a filter top above -5 will end up in the simulation
    active = idomain == 1

    active.loc[1, ...] = True
    active.loc[2:, ...] = False

    # modify the well filter top and filter bottoms so that
    # well 0 is not placed
    # well 1 is partially placed
    # well 2 is fully placed
    # well 3 is partially placed
    # well 4 is fully placed
    # well 5 is partially placed
    # wells 6 to 11 are not placed
    modified_well_fixture[2] = [
        -6.0,
        -1.0,
        0.0,
        -2.0,
        0.0,
        -3.0,
        -6.0,
        -6.0,
        -6.0,
        -6.0,
        -6.0,
        -6.0,
    ]
    modified_well_fixture[3] = [
        -102.0,
        -102.0,
        -5,
        -102.0,
        -5,
        -102.0,
        -1020.0,
        -1020.0,
        -1020.0,
        -1020.0,
        -1020.0,
        -1020.0,
    ]

    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )

        wel = imod.mf6.Well(*modified_well_fixture)

        k = ones_like(idomain)
        _ = wel.to_mf6_pkg(active, top, bottom, k)

    # the wells that were fully or partially placed should not appear in the log message
    # but all the wells that are completely left out should be listed
    with open(logfile_path, "r") as log_file:
        log = log_file.read()
        assert "Some wells were not placed" in log
        assert "id = 1 x" not in log
        assert "id = 2 x" not in log
        assert "id = 3 x" not in log
        assert "id = 4 x" not in log
        assert "id = 5 x" not in log
        assert "id = 0 x" in log
        assert "id = 6 x" in log
        assert "id = 7 x" in log
        assert "id = 8 x" in log
        assert "id = 9 x" in log
        assert "id = 10 x" in log
        assert "id = 11 x" in log


@parametrize_with_cases("dis", cases=BasicGridCases)
def test_to_mf6_pkg__logging_without_message(
    tmp_path, dis, well_high_lvl_test_data_transient
):
    # This test activates logging, and then converts a high level well package to
    # an MF6 package, in such a way that all the wells can be placed.
    # Logging is active, and the log file should not include the "Some wells were not placed"
    # message
    logfile_path = tmp_path / "logfile.txt"
    _, idomain, top, bottom = dis

    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )

        wel = imod.mf6.Well(*well_high_lvl_test_data_transient)
        active = idomain == 1
        k = ones_like(idomain)

        active.loc[1, ...] = True
        active.loc[2:, ...] = True

        # Act
        _ = wel.to_mf6_pkg(active, top, bottom, k)

    with open(logfile_path, "r") as log_file:
        log = log_file.read()
        assert "Some wells were not placed" not in log


@parametrize_with_cases("dis", cases=BasicGridCases)
def test_to_mf6_pkg__error_on_all_wells_removed(dis, well_high_lvl_test_data_transient):
    """Drop all wells, then run to_mf6_pkg"""
    _, idomain, top, bottom = dis

    wel = imod.mf6.Well(*well_high_lvl_test_data_transient)
    wel.dataset = wel.dataset.drop_sel(index=wel.dataset["index"])
    active = idomain == 1
    k = ones_like(idomain)

    with pytest.raises(ValidationError, match="No wells were assigned in package"):
        wel.to_mf6_pkg(active, top, bottom, k)


@parametrize_with_cases("dis", cases=BasicGridCases)
def test_to_mf6_pkg__error_on_well_removal(dis, well_high_lvl_test_data_transient):
    """Set k values at well location x=81 to 1e-3, causing it to be dropped.
    Should throw error if error_on_well_removal = True"""
    dis_pkg_type, idomain, top, bottom = dis

    wel = imod.mf6.Well(minimum_k=0.9, *well_high_lvl_test_data_transient)
    active = idomain == 1
    k = ones_like(idomain).astype(float)
    # Set k values at well location x=81 to 1e-3, causing it to be dropped.
    if dis_pkg_type is imod.mf6.VerticesDiscretization:
        dim = k.ugrid.grid.face_dimension
        indices = k.sel(layer=1).ugrid.sel(x=85.0).coords[dim].data
        k.loc[{dim: indices}] = 1e-3
    else:
        k.loc[{"x": 85.0}] = 1e-3

    with pytest.raises(ValidationError, match="x = 81"):
        wel.to_mf6_pkg(active, top, bottom, k, strict_validation=True)

    mf6_wel = wel.to_mf6_pkg(active, top, bottom, k, strict_validation=False)
    assert mf6_wel.dataset.sizes["ncellid"] < wel.dataset.sizes["index"]


@pytest.mark.parametrize("save_flows", [True, False])
@pytest.mark.parametrize("print_input", [True, False])
@pytest.mark.parametrize("print_flows", [True, False])
def test_to_mf6_pkg__save_flows(
    basic_dis, well_high_lvl_test_data_transient, save_flows, print_input, print_flows
):
    # Arrange
    idomain, top, bottom = basic_dis

    wel = imod.mf6.Well(
        *well_high_lvl_test_data_transient,
        save_flows=save_flows,
        print_input=print_input,
        print_flows=print_flows,
    )
    active = idomain == 1
    k = xr.ones_like(idomain)

    # Act
    mf6_wel = wel.to_mf6_pkg(active, top, bottom, k)
    mf6_ds = mf6_wel.dataset

    # Assert
    mf6_ds["save_flows"].values[()] == save_flows
    mf6_ds["print_input"].values[()] == print_input
    mf6_ds["print_flows"].values[()] == print_flows


def test_is_empty(well_high_lvl_test_data_transient):
    # Arrange
    wel = imod.mf6.Well(*well_high_lvl_test_data_transient)
    empty_wel_args = ([] for i in range(len(well_high_lvl_test_data_transient)))
    wel_empty = imod.mf6.Well(*empty_wel_args, validate=False)

    # Act/Assert
    assert not wel.is_empty()
    assert wel_empty.is_empty()


@parametrize_with_cases("dis", cases=BasicGridCases)
def test_cleanup(dis, well_high_lvl_test_data_transient):
    # Arrange
    wel = imod.mf6.Well(*well_high_lvl_test_data_transient)
    ds_original = wel.dataset.copy()

    dis_pkg_type, idomain, top, bottom = dis
    if "layer" in top.dims:
        top = top.isel(layer=0, drop=True)
    deep_offset = 100.0
    dis_normal = dis_pkg_type(top, bottom, idomain)
    dis_deep = dis_pkg_type(top - deep_offset, bottom - deep_offset, idomain)

    # Nothing to be cleaned up with default discretization
    wel.cleanup(dis_normal)
    xr.testing.assert_identical(ds_original, wel.dataset)

    # Cleanup
    wel.cleanup(dis_deep)
    assert not ds_original.identical(wel.dataset)
    # Wells filters should be placed downwards at surface level as point filters
    np.testing.assert_array_almost_equal(
        wel.dataset["screen_top"], wel.dataset["screen_bottom"]
    )
    np.testing.assert_array_almost_equal(wel.dataset["screen_top"], top - deep_offset)


class ClipBoxCases:
    @staticmethod
    def case_clip_xy(parameterizable_basic_dis):
        clip_arguments = {
            "x_min": 52.0,
            "x_max": 76.0,
            "y_max": 67.0,
        }
        expected_dims = {
            "layered": {"index": 6, "species": 2},
            "nonlayered": {"index": 6, "species": 2},
        }
        return clip_arguments, expected_dims, does_not_raise(), does_not_raise()

    @staticmethod
    def case_clip_layer_max(parameterizable_basic_dis):
        _, top, bottom = parameterizable_basic_dis
        clip_arguments = {"layer_max": 2, "bottom": bottom, "top": top}

        expected_dims = {
            "layered": {"index": 6, "species": 2},
            "nonlayered": {"index": 6, "species": 2},
        }
        return clip_arguments, expected_dims, does_not_raise(), does_not_raise()

    @staticmethod
    def case_clip_layer_min(parameterizable_basic_dis):
        _, top, bottom = parameterizable_basic_dis
        clip_arguments = {"layer_min": 5, "bottom": bottom, "top": top}

        expected_dims = {
            "layered": {"index": 6, "species": 2},
            "nonlayered": {"index": 6, "species": 2},
        }
        return clip_arguments, expected_dims, does_not_raise(), does_not_raise()

    @staticmethod
    def case_clip_layer_min_layer_max(parameterizable_basic_dis):
        _, top, bottom = parameterizable_basic_dis
        clip_arguments = {"layer_min": 1, "layer_max": 1, "bottom": bottom, "top": top}

        expected_dims = {
            "layered": {"index": 6, "species": 2},
            "nonlayered": {"index": 6, "species": 2},
        }
        return clip_arguments, expected_dims, does_not_raise(), does_not_raise()

    @staticmethod
    def case_clip_top_is_scalar(parameterizable_basic_dis):
        _, _, bottom = parameterizable_basic_dis
        top = 0.0
        clip_arguments = {"layer_max": 2, "bottom": bottom, "top": top}

        expected_dims = {
            "layered": {"index": 6, "species": 2},
            "nonlayered": {"index": 6, "species": 2},
        }
        return clip_arguments, expected_dims, does_not_raise(), does_not_raise()

    @staticmethod
    def case_clip_top_is_non_layered_structuredgrid(parameterizable_basic_dis):
        idomain, top, bottom = parameterizable_basic_dis
        top, bottom = broadcast_to_full_domain(idomain, top, bottom)
        top = top.isel(layer=0).drop_vars("layer")

        clip_arguments = {"layer_max": 2, "bottom": bottom, "top": top}

        expected_dims = {
            "layered": {"index": 6, "species": 2},
            "nonlayered": {"index": 2, "species": 2},
        }
        return clip_arguments, expected_dims, does_not_raise(), does_not_raise()

    @staticmethod
    def case_clip_top_is_layered_structuredgrid(parameterizable_basic_dis):
        idomain, top, bottom = parameterizable_basic_dis

        top, bottom = broadcast_to_full_domain(idomain, top, bottom)
        clip_arguments = {"layer_max": 2, "bottom": bottom, "top": top}

        expected_dims = {
            "layered": {"index": 6, "species": 2},
            "nonlayered": {"index": 2, "species": 2},
        }
        return clip_arguments, expected_dims, does_not_raise(), does_not_raise()

    @staticmethod
    def case_clip_top_is_non_layered_unstructuredgrid(parameterizable_basic_dis):
        idomain, top, bottom = parameterizable_basic_dis
        top, bottom = broadcast_to_full_domain(idomain, top, bottom)
        top = xu.UgridDataArray.from_structured2d(top)
        bottom = xu.UgridDataArray.from_structured2d(bottom)

        top = top.isel(layer=0).drop_vars("layer")

        clip_arguments = {"layer_max": 2, "bottom": bottom, "top": top}

        expected_dims = {
            "layered": {"index": 6, "species": 2},
            "nonlayered": {"index": 2, "species": 2},
        }
        return clip_arguments, expected_dims, does_not_raise(), does_not_raise()

    @staticmethod
    def case_clip_top_is_layered_unstructuredgrid(parameterizable_basic_dis):
        idomain, top, bottom = parameterizable_basic_dis
        top, bottom = broadcast_to_full_domain(idomain, top, bottom)
        top = xu.UgridDataArray.from_structured2d(top)
        bottom = xu.UgridDataArray.from_structured2d(bottom)

        clip_arguments = {"layer_max": 2, "bottom": bottom, "top": top}

        expected_dims = {
            "layered": {"index": 6, "species": 2},
            "nonlayered": {"index": 2, "species": 2},
        }
        return clip_arguments, expected_dims, does_not_raise(), does_not_raise()

    @staticmethod
    def case_clip_missing_top(parameterizable_basic_dis):
        _, _, bottom = parameterizable_basic_dis
        clip_arguments = {"layer_max": 2, "bottom": bottom}

        expected_dims = {
            "layered": {"index": 6, "species": 2},
            "nonlayered": {"index": 6, "species": 2},
        }
        return (
            clip_arguments,
            expected_dims,
            pytest.raises(ValueError),
            does_not_raise(),
        )

    @staticmethod
    def case_clip_missing_bottom(parameterizable_basic_dis):
        _, top, _ = parameterizable_basic_dis
        clip_arguments = {"layer_max": 2, "top": top}

        expected_dims = {
            "layered": {"index": 6, "species": 2},
            "nonlayered": {"index": 6, "species": 2},
        }
        return (
            clip_arguments,
            expected_dims,
            pytest.raises(ValueError),
            does_not_raise(),
        )


@pytest.mark.parametrize(
    "parameterizable_basic_dis",
    [
        BasicDisSettings(
            nlay=10,
            zstop=-10.0,
            xstart=50.0,
            xstop=100.0,
            ystart=50.0,
            ystop=100.0,
            nrow=10,
            ncol=10,
        )
    ],
    indirect=True,
)
@parametrize_with_cases(
    ("clip_box_args", "expected_dims", "expectation", "_"), cases=ClipBoxCases
)
def test_clip_box__well_stationary(
    well_high_lvl_test_data_stationary, clip_box_args, expected_dims, expectation, _
):
    # Arrange
    wel = imod.mf6.Well(*well_high_lvl_test_data_stationary)

    with expectation:
        # Act
        ds = wel.clip_box(**clip_box_args).dataset

        # Assert
        assert dict(ds.sizes) == expected_dims["nonlayered"]


@pytest.mark.parametrize(
    "parameterizable_basic_dis",
    [
        BasicDisSettings(
            nlay=10,
            zstop=-10.0,
            xstart=50.0,
            xstop=100.0,
            ystart=50.0,
            ystop=100.0,
            nrow=10,
            ncol=10,
        )
    ],
    indirect=True,
)
@parametrize_with_cases(
    ("clip_box_args", "expected_dims", "_", "expectation"), cases=ClipBoxCases
)
def test_clip_box__layered_well_stationary(
    well_high_lvl_test_data_stationary, clip_box_args, expected_dims, _, expectation
):
    x, y, _, _, rate_wel, concentration = well_high_lvl_test_data_stationary
    layer = [1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9]
    wel = imod.mf6.LayeredWell(x, y, layer, rate_wel, concentration)

    with expectation:
        # Act
        ds = wel.clip_box(**clip_box_args).dataset

        # Assert
        assert dict(ds.sizes) == expected_dims["layered"]


@pytest.mark.parametrize(
    "parameterizable_basic_dis",
    [BasicDisSettings(nlay=10, zstop=-10.0)],
    indirect=True,
)
def test_clip_box__high_lvl_transient(
    well_high_lvl_test_data_transient, parameterizable_basic_dis
):
    # Arrange
    wel = imod.mf6.Well(*well_high_lvl_test_data_transient)
    _, top, bottom = parameterizable_basic_dis

    # Act & Assert
    # Test clipping x & y without specified time
    ds = wel.clip_box(x_min=52.0, x_max=76.0, y_max=67.0).dataset
    assert dict(ds.sizes) == {"index": 6, "time": 5, "species": 2}

    # Test clipping with z
    ds = wel.clip_box(layer_max=2, top=top, bottom=bottom).dataset
    assert dict(ds.sizes) == {"index": 6, "time": 5, "species": 2}
    ds = wel.clip_box(layer_min=5, top=top, bottom=bottom).dataset
    assert dict(ds.sizes) == {"index": 6, "time": 5, "species": 2}

    # Test clipping with specified time
    timestr = "2000-01-03"
    ds = wel.clip_box(time_min=timestr).dataset
    assert dict(ds.sizes) == {"index": 12, "time": 3, "species": 2}

    # Test clipping with specified time and spatial dimensions
    timestr = "2000-01-03"
    ds = wel.clip_box(x_min=52.0, x_max=76.0, y_max=67.0, time_min=timestr).dataset
    assert dict(ds.sizes) == {"index": 6, "time": 3, "species": 2}

    # Test clipping with specified time inbetween timesteps
    timestr = "2000-01-03 18:00:00"
    ds = wel.clip_box(x_min=52.0, x_max=76.0, y_max=67.0, time_min=timestr).dataset
    assert dict(ds.sizes) == {"index": 6, "time": 3, "species": 2}
    ds_first_time = ds.isel(time=0)
    assert ds_first_time.coords["time"] == np.datetime64(timestr)
    np.testing.assert_allclose(
        ds_first_time["rate"].values, np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    )
    np.testing.assert_allclose(
        ds_first_time["concentration"].values,
        np.array(
            [[30.0, 30.0, 30.0, 30.0, 30.0, 30.0], [69.0, 69.0, 69.0, 69.0, 69.0, 69.0]]
        ),
    )


def test_derive_cellid_from_points(basic_dis, well_high_lvl_test_data_stationary):
    # Arrange
    idomain, _, _ = basic_dis
    x, y, _, _, _, _ = well_high_lvl_test_data_stationary
    layer = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

    dim_cellid_expected = np.array(["layer", "row", "column"])
    cellid_expected = np.array(
        [
            [1, 1, 9],
            [1, 2, 9],
            [1, 3, 9],
            [1, 1, 8],
            [1, 2, 8],
            [1, 3, 8],
            [2, 4, 7],
            [2, 5, 7],
            [2, 6, 7],
            [2, 4, 6],
            [2, 5, 6],
            [2, 6, 6],
        ],
        dtype=np.int64,
    )

    # Act
    cellid = derive_cellid_from_points(idomain, x, y, layer)

    # Assert
    np.testing.assert_array_equal(cellid, cellid_expected)
    np.testing.assert_equal(cellid.coords["dim_cellid"].values, dim_cellid_expected)


@parametrize("wel_class", [Well, LayeredWell])
def test_import_and_convert_to_mf6(imod5_dataset, tmp_path, wel_class):
    data = imod5_dataset[0]
    target_dis = StructuredDiscretization.from_imod5_data(data)
    target_npf = NodePropertyFlow.from_imod5_data(data, target_dis.dataset["idomain"])

    times = list(pd.date_range(datetime(1989, 1, 1), datetime(2013, 1, 1), 8400))

    # import grid-agnostic well from imod5 data (it contains 1 well)
    wel = wel_class.from_imod5_data("wel-WELLS_L3", data, times, minimum_thickness=1.0)
    assert wel.dataset["x"].values[0] == 197910.0
    assert wel.dataset["y"].values[0] == 362860.0
    assert np.mean(wel.dataset["rate"].values) == -317.2059091946156
    # convert to a gridded well
    top = target_dis.dataset["top"]
    bottom = target_dis.dataset["bottom"]
    active = target_dis.dataset["idomain"]
    k = target_npf.dataset["k"]
    mf6_well = wel.to_mf6_pkg(active, top, bottom, k, True)

    # assert mf6 well properties
    assert len(mf6_well.dataset["x"].values) == 1
    assert mf6_well.dataset["x"].values[0] == 197910.0
    assert mf6_well.dataset["y"].values[0] == 362860.0
    assert np.mean(mf6_well.dataset["rate"].values) == -317.2059091946156

    # write the package for validation
    write_context = WriteContext(simulation_directory=tmp_path)
    mf6_well._write("wel", [], write_context)


@parametrize("wel_class", [Well, LayeredWell])
def test_import__as_steady_state(imod5_dataset, wel_class):
    data = imod5_dataset[0]
    times = "steady-state"
    # Import grid-agnostic well from imod5 data (it contains 1 well)
    wel = wel_class.from_imod5_data("wel-WELLS_L3", data, times)

    assert "time" not in wel.dataset.coords
    assert wel.dataset["rate"].shape == (1,)
    np.testing.assert_almost_equal(wel.dataset["rate"].values, -323.89361702)


@parametrize("wel_class", [Well])
def test_import_and_cleanup(imod5_dataset, wel_class: Well):
    data = imod5_dataset[0]
    target_dis = StructuredDiscretization.from_imod5_data(data)

    ntimes = 8399
    times = list(pd.date_range(datetime(1989, 1, 1), datetime(2013, 1, 1), ntimes + 1))

    # Import grid-agnostic well from imod5 data (it contains 1 well)
    wel = wel_class.from_imod5_data("wel-WELLS_L3", data, times)
    assert len(wel.dataset.coords["time"]) == ntimes
    # Cleanup
    wel.cleanup(target_dis)
    # Nothing to be cleaned, single well point is located properly, test that
    # time coordinate has not been dropped.
    assert "time" in wel.dataset.coords
    assert len(wel.dataset.coords["time"]) == ntimes


@parametrize("wel_class", [Well, LayeredWell])
def test_import_simple_wells__steady_state(
    well_simple_import_prj__steady_state, wel_class
):
    imod5dict, _ = open_projectfile_data(well_simple_import_prj__steady_state)
    # Set layer to 1, to avoid validation error.
    if wel_class is LayeredWell:
        imod5dict["wel-ipf1"]["layer"] = [1]
        imod5dict["wel-ipf2"]["layer"] = [1]

    wel1 = wel_class.from_imod5_data("wel-ipf1", imod5dict, "steady-state")
    wel2 = wel_class.from_imod5_data("wel-ipf2", imod5dict, "steady-state")

    assert wel1.dataset["rate"].shape == (13,)
    assert wel2.dataset["rate"].shape == (2,)

    expected = np.array([(7 * -0.2), (6 * -0.5)])
    np.testing.assert_allclose(wel2.dataset["rate"].values, expected)


@parametrize("wel_class", [Well, LayeredWell])
def test_import_simple_wells__transient(well_simple_import_prj__transient, wel_class):
    imod5dict, _ = open_projectfile_data(well_simple_import_prj__transient)
    # Set layer to 1, to avoid validation error.
    if wel_class is LayeredWell:
        imod5dict["wel-ipf1"]["layer"] = [1]
        imod5dict["wel-ipf2"]["layer"] = [1]

    with pytest.raises(ValueError):
        wel_class.from_imod5_data("wel-ipf1", imod5dict, "steady-state")

    with pytest.raises(ValueError):
        wel_class.from_imod5_data("wel-ipf2", imod5dict, "steady-state")

    times = [imod5dict["wel-ipf2"]["time"][0], datetime(2001, 1, 1)]

    wel1 = wel_class.from_imod5_data("wel-ipf1", imod5dict, times)
    wel2 = wel_class.from_imod5_data("wel-ipf2", imod5dict, times)

    assert wel1.dataset["rate"].shape == (1, 13)
    assert wel2.dataset["rate"].shape == (1, 2)
    expected = np.array([(7 * -0.2), (6 * -0.5)])
    np.testing.assert_allclose(wel2.dataset["rate"].values[0], expected)


@parametrize("wel_class", [Well, LayeredWell])
def test_import_multiple_wells(well_regular_import_prj, wel_class):
    imod5dict, _ = open_projectfile_data(well_regular_import_prj)
    times = [
        datetime(1981, 11, 30),
        datetime(1981, 12, 31),
        datetime(1982, 1, 31),
        datetime(1982, 2, 28),
        datetime(1982, 3, 31),
        datetime(1982, 4, 30),
    ]
    # Set layer to 1, to avoid validation error.
    if wel_class is LayeredWell:
        imod5dict["wel-ipf1"]["layer"] = [1]
        imod5dict["wel-ipf2"]["layer"] = [1]
    # import grid-agnostic well from imod5 data (it contains 2 packages with 3 wells each)
    wel1 = wel_class.from_imod5_data("wel-ipf1", imod5dict, times)
    wel2 = wel_class.from_imod5_data("wel-ipf2", imod5dict, times)

    assert np.all(wel1.x == np.array([191112.11, 191171.96, 191231.52]))
    assert np.all(wel2.x == np.array([191112.11, 191171.96, 191231.52]))
    assert wel1.dataset["rate"].shape == (5, 3)
    assert wel2.dataset["rate"].shape == (5, 3)


@parametrize("wel_class", [Well, LayeredWell])
def test_import_from_imod5_with_duplication(well_duplication_import_prj, wel_class):
    imod5dict, _ = open_projectfile_data(well_duplication_import_prj)
    times = [
        datetime(1981, 11, 30),
        datetime(1981, 12, 31),
        datetime(1982, 1, 31),
        datetime(1982, 2, 28),
        datetime(1982, 3, 31),
        datetime(1982, 4, 30),
    ]
    # Set layer to 1, to avoid validation error.
    if wel_class is LayeredWell:
        imod5dict["wel-ipf1"]["layer"] = [1]
        imod5dict["wel-ipf2"]["layer"] = [1]
    # import grid-agnostic well from imod5 data (it contains 2 packages with 3 wells each)
    wel1 = wel_class.from_imod5_data("wel-ipf1", imod5dict, times)
    wel2 = wel_class.from_imod5_data("wel-ipf2", imod5dict, times)

    assert np.all(wel1.x == np.array([191171.96, 191231.52, 191231.52]))
    assert np.all(wel2.x == np.array([191112.11, 191171.96, 191231.52]))
    assert wel1.dataset["rate"].shape == (5, 3)
    assert wel2.dataset["rate"].shape == (5, 3)


@pytest.mark.parametrize("layer", [0, 1])
def test_logmessage_for_layer_assignment_import_imod5(
    tmp_path, well_regular_import_prj, layer
):
    imod5dict = open_projectfile_data(well_regular_import_prj)

    logfile_path = tmp_path / "logfile.txt"
    imod5dict[0]["wel-ipf1"]["layer"] = [layer] * len(imod5dict[0]["wel-ipf1"]["layer"])

    try:
        with open(logfile_path, "w") as sys.stdout:
            # start logging
            imod.logging.configure(
                LoggerType.PYTHON,
                log_level=LogLevel.WARNING,
                add_default_file_handler=False,
                add_default_stream_handler=True,
            )

            _ = imod.mf6.Well.from_imod5_data("wel-ipf1", imod5dict[0], times)

    finally:
        # turn the logger off again
        imod.logging.configure(
            LoggerType.NULL,
            log_level=LogLevel.WARNING,
            add_default_file_handler=False,
            add_default_stream_handler=False,
        )

    # import grid-agnostic well from imod5 data (it contains 2 packages with 3 wells each)
    with open(logfile_path, "r") as log_file:
        log = log_file.read()
        message_required = layer != 0
        message_present = (
            "In well wel-ipf1 a layer was assigned, but this is not\nsupported" in log
        )
        assert message_required == message_present


@pytest.mark.parametrize("remove", ["filt_top", "filt_bot", None])
def test_logmessage_for_missing_filter_settings(
    tmp_path, well_regular_import_prj, remove
):
    imod5dict = open_projectfile_data(well_regular_import_prj)
    logfile_path = tmp_path / "logfile.txt"
    if remove is not None:
        imod5dict[0]["wel-ipf1"]["dataframe"][0] = imod5dict[0]["wel-ipf1"][
            "dataframe"
        ][0].drop(remove, axis=1)

    try:
        with open(logfile_path, "w") as sys.stdout:
            # start logging
            imod.logging.configure(
                LoggerType.PYTHON,
                log_level=LogLevel.WARNING,
                add_default_file_handler=False,
                add_default_stream_handler=True,
            )

            _ = imod.mf6.Well.from_imod5_data("wel-ipf1", imod5dict[0], times)
    except Exception:
        assert remove is not None

    finally:
        # turn the logger off again
        imod.logging.configure(
            LoggerType.NULL,
            log_level=LogLevel.WARNING,
            add_default_file_handler=False,
            add_default_stream_handler=False,
        )

    # import grid-agnostic well from imod5 data (it contains 2 packages with 3 wells each)
    with open(logfile_path, "r") as log_file:
        log = log_file.read()
        message_required = remove is not None
        message_present = (
            "In well wel-ipf1 the 'filt_top' and 'filt_bot' columns were\nnot both found;"
            in log
        )
        assert message_required == message_present


def test_from_imod5_cap_data__grid(cap_data_sprinkling_grid):
    # Arrange
    expected_layer = np.array([2, 1, 2, 1, 2, 1])
    expected_y = np.array([3.0, 3.0, 2.0, 2.0, 1.0, 1.0])
    expected_x = np.array([2.0, 3.0, 2.0, 3.0, 2.0, 3.0])

    # Act
    well = LayeredWell.from_imod5_cap_data(cap_data_sprinkling_grid)

    # Assert
    ds = well.dataset
    np.testing.assert_equal(ds["rate"].to_numpy(), 0.0)
    np.testing.assert_array_equal(ds["layer"].to_numpy(), expected_layer)
    np.testing.assert_array_equal(ds["x"].to_numpy(), expected_x)
    np.testing.assert_array_equal(ds["y"].to_numpy(), expected_y)


@pytest.mark.timeout(300)
def test_from_imod5_cap_data__big_grid(cap_data_sprinkling_grid__big):
    """Test if performance is acceptable for large grids."""
    # Arrange
    bnd_2d = cap_data_sprinkling_grid__big["cap"]["boundary"]
    layer = xr.DataArray([1, 1], coords={"layer": [1, 2]}, dims=["layer"])
    bnd = layer * bnd_2d
    # Act
    well = LayeredWell.from_imod5_cap_data(cap_data_sprinkling_grid__big)
    mf6_wel = well.to_mf6_pkg(bnd.astype(bool), bnd, bnd, bnd)
    # Assert
    assert mf6_wel.dataset.sizes["ncellid"] == bnd_2d.size


def test_from_imod5_cap_data__points(cap_data_sprinkling_points):
    with pytest.raises(NotImplementedError):
        LayeredWell.from_imod5_cap_data(cap_data_sprinkling_points)

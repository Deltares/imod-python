import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest_cases import parametrize_with_cases

import imod.prepare.wells as prepwel
from imod.testing import assert_frame_equal


def test_compute_vectorized_overlap():
    bounds_a = np.array(
        [
            [0.0, 3.0],
            [0.0, 3.0],
        ]
    )
    bounds_b = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
        ]
    )
    actual = prepwel.compute_vectorized_overlap(bounds_a, bounds_b)
    assert np.array_equal(actual, np.array([1.0, 1.0]))


def test_compute_overlap():
    # Three wells
    wells = pd.DataFrame(
        {
            "top": [5.0, 4.0, 3.0],
            "bottom": [4.0, 2.0, -1.0],
        }
    )
    top = xr.DataArray(
        data=[
            [10.0, 10.0, 10.0],
            [0.0, 0.0, 0.0],
        ],
        dims=["layer", "index"],
    )
    bottom = xr.DataArray(
        data=[
            [0.0, 0.0, 0.0],
            [-10.0, -10.0, -10.0],
        ],
        dims=["layer", "index"],
    )
    actual = prepwel.compute_overlap(wells, top, bottom)
    expected = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 1.0])
    assert np.allclose(actual, expected)


class AssignWellCases:
    def case_mix_wells(self):
        # This is a testcase where NOT all the wells are in the domain, but most
        # are. It can be used to verify that validation erros will not occurr if
        # validation is off
        ones = xr.DataArray(
            data=np.ones((2, 3, 3)),
            coords={"layer": [1, 2], "y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5]},
            dims=["layer", "y", "x"],
        )
        top = ones.copy()
        top[0] = 10.0
        top[1] = 0.0
        bottom = ones.copy()
        bottom[0] = 0.0
        bottom[1] = -10.0
        k = ones.copy()
        k[0] = 10.0
        k[1] = 20.0

        wells = pd.DataFrame(
            {
                "x": [0.6, 1.1, 2.3, 5.0],
                "y": [0.6, 1.1, 2.3, 5.0],
                "id": [1, 2, 3, 4],
                "top": [5.0, 4.0, 3.0, 0.0],
                "bottom": [4.0, 2.0, -1.0, 0.0],
                "rate": [1.0, 10.0, 100.0, 0.1],
            }
        )
        return wells, top, bottom, k

    def case_all_in_domain(self):
        # This is a testcase where all where all the wells are in the domain and
        # have valid tops and bottoms
        ones = xr.DataArray(
            data=np.ones((2, 3, 3)),
            coords={"layer": [1, 2], "y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5]},
            dims=["layer", "y", "x"],
        )
        top = ones.copy()
        top[0] = 10.0
        top[1] = 0.0
        bottom = ones.copy()
        bottom[0] = 0.0
        bottom[1] = -10.0
        k = ones.copy()
        k[0] = 10.0
        k[1] = 20.0

        wells = pd.DataFrame(
            {
                "x": [0.6, 1.1, 2.3, 2.6],
                "y": [0.6, 1.1, 2.3, 2.6],
                "id": [1, 2, 3, 4],
                "top": [5.0, 4.0, 3.0, 0.0],
                "bottom": [4.0, 2.0, -1.0, 0.0],
                "rate": [1.0, 10.0, 100.0, 0.1],
            }
        )

        return wells, top, bottom, k


class TestAssignWell:
    @parametrize_with_cases(
        "wells, top, bottom, k", cases=AssignWellCases.case_all_in_domain
    )
    def test_locate_wells__no_kh(self, wells, top, bottom, k):
        id_in_bounds, xy_top, xy_bottom, xy_kh = prepwel.locate_wells(
            wells=wells,
            top=top,
            bottom=bottom,
            k=None,
        )

        assert np.array_equal(id_in_bounds, [1, 2, 3, 4])
        assert np.allclose(xy_top, [[10.0, 10.0, 10.0, 10], [0.0, 0.0, 0.0, 0.0]])
        assert np.allclose(
            xy_bottom, [[0.0, 0.0, 0.0, 0.0], [-10.0, -10.0, -10.0, -10.0]]
        )
        assert xy_kh == 1.0

    @parametrize_with_cases(
        "wells, top, bottom, k", cases=AssignWellCases.case_all_in_domain
    )
    def test_locate_wells(self, wells, top, bottom, k):
        id_in_bounds, xy_top, xy_bottom, xy_kh = prepwel.locate_wells(
            wells=wells,
            top=top,
            bottom=bottom,
            k=k,
        )

        assert np.array_equal(id_in_bounds, [1, 2, 3, 4])
        assert np.allclose(xy_top, [[10.0, 10.0, 10.0, 10], [0.0, 0.0, 0.0, 0.0]])
        assert np.allclose(
            xy_bottom, [[0.0, 0.0, 0.0, 0.0], [-10.0, -10.0, -10.0, -10.0]]
        )
        assert np.allclose(xy_kh, [[10.0, 10.0, 10.0, 10.0], [20.0, 20.0, 20.0, 20.0]])

    @parametrize_with_cases(
        "wells, top, bottom, k", cases=AssignWellCases.case_all_in_domain
    )
    def test_locate_wells_errors(self, wells, top, bottom, k):
        with pytest.raises(TypeError, match="top and bottom"):
            prepwel.locate_wells(wells, top.values, bottom, None)
        with pytest.raises(ValueError, match="bottom grid does not match"):
            small_bottom = bottom.sel(y=slice(2.0, 0.0))
            prepwel.locate_wells(wells, top, small_bottom, None)
        with pytest.raises(ValueError, match="k grid does not match"):
            small_kh = k.sel(y=slice(2.0, 0.0))
            prepwel.locate_wells(wells, top, bottom, small_kh)

    @parametrize_with_cases(
        "wells, top, bottom, k", cases=AssignWellCases.case_all_in_domain
    )
    def test_assign_wells_errors(self, wells, top, bottom, k):
        with pytest.raises(ValueError, match="Columns are missing"):
            faulty_wells = pd.DataFrame({"id": [1], "x": [1.0], "y": [1.0]})
            prepwel.assign_wells(faulty_wells, top, bottom, k)
        with pytest.raises(TypeError, match="top, bottom, and optionally"):
            prepwel.assign_wells(wells, top, bottom.values)
        with pytest.raises(TypeError, match="top, bottom, and optionally"):
            prepwel.assign_wells(wells, top.values, bottom, k)

    @parametrize_with_cases(
        "wells, top, bottom, k", cases=AssignWellCases.case_all_in_domain
    )
    def test_assign_wells__no_kh(self, wells, top, bottom, k):
        actual = prepwel.assign_wells(
            wells=wells,
            top=top,
            bottom=bottom,
        )
        assert isinstance(actual, pd.DataFrame)
        expected = pd.DataFrame(
            {
                "index": [0, 1, 2, 3, 4],
                "id": [1, 2, 3, 3, 4],
                "layer": [1, 1, 1, 2, 2],
                "bottom": [4.0, 2.0, -1.0, -1.0, 0.0],
                "overlap": [1.0, 2.0, 3.0, 1.0, 10.0],
                "rate": [1.0, 10.0, 75.0, 25.0, 0.1],
                "top": [5.0, 4.0, 3.0, 3.0, 0.0],
                "k": [1.0, 1.0, 1.0, 1.0, 1.0],
                "transmissivity": [1.0, 2.0, 3.0, 1.0, 10.0],
                "x": [0.6, 1.1, 2.3, 2.3, 2.6],
                "y": [0.6, 1.1, 2.3, 2.3, 2.6],
            }
        )
        assert_frame_equal(actual, expected, check_like=True)

    @parametrize_with_cases(
        "wells, top, bottom, k", cases=AssignWellCases.case_all_in_domain
    )
    def test_assign_wells(self, wells, top, bottom, k):
        actual = prepwel.assign_wells(
            wells=wells,
            top=top,
            bottom=bottom,
            k=k,
        )
        assert isinstance(actual, pd.DataFrame)
        expected = pd.DataFrame(
            {
                "index": [0, 1, 2, 3, 4],
                "id": [1, 2, 3, 3, 4],
                "layer": [1, 1, 1, 2, 2],
                "bottom": [4.0, 2.0, -1.0, -1.0, 0.0],
                "overlap": [1.0, 2.0, 3.0, 1.0, 10.0],
                "rate": [1.0, 10.0, 60.0, 40.0, 0.1],
                "top": [5.0, 4.0, 3.0, 3.0, 0.0],
                "k": [10.0, 10.0, 10.0, 20.0, 20.0],
                "transmissivity": [10.0, 20.0, 30.0, 20.0, 200.0],
                "x": [0.6, 1.1, 2.3, 2.3, 2.6],
                "y": [0.6, 1.1, 2.3, 2.3, 2.6],
            }
        )
        assert_frame_equal(actual, expected, check_like=True)

    @parametrize_with_cases(
        "wells, top, bottom, k", cases=AssignWellCases.case_all_in_domain
    )
    def test_assign_wells_minimum_thickness(self, wells, top, bottom, k):
        actual = prepwel.assign_wells(
            wells=wells,
            top=top,
            bottom=bottom,
            k=k,
            minimum_thickness=1.01,
        )
        assert isinstance(actual, pd.DataFrame)
        expected = pd.DataFrame(
            {
                "index": [0, 1, 2],
                "id": [2, 3, 4],
                "layer": [1, 1, 2],
                "bottom": [2.0, -1.0, 0.0],
                "overlap": [2.0, 3.0, 10.0],
                "rate": [10.0, 100.0, 0.1],
                "top": [4.0, 3.0, 0.0],
                "k": [10.0, 10.0, 20.0],
                "transmissivity": [20.0, 30.0, 200.0],
                "x": [1.1, 2.3, 2.6],
                "y": [1.1, 2.3, 2.6],
            }
        )
        assert_frame_equal(actual, expected, check_like=True)

    @parametrize_with_cases(
        "wells, top, bottom, k", cases=AssignWellCases.case_all_in_domain
    )
    def test_assign_wells_transient_rate(self, wells, top, bottom, k):
        wells_xr = wells.to_xarray()
        multiplier = xr.DataArray(
            data=np.arange(1.0, 6.0),
            coords={"time": pd.date_range("2000-01-01", "2000-01-05")},
            dims=["time"],
        )
        wells_xr["rate"] = multiplier * wells_xr["rate"]
        transient_wells = wells_xr.to_dataframe().reset_index()

        actual = prepwel.assign_wells(
            wells=transient_wells,
            top=top,
            bottom=bottom,
            k=k,
        )
        assert np.array_equal(actual["id"], np.repeat([1, 2, 3, 3, 4], 5))

        actual = prepwel.assign_wells(
            wells=transient_wells,
            top=top,
            bottom=bottom,
            k=k,
            minimum_thickness=1.01,
        )
        assert np.array_equal(actual["id"], np.repeat([2, 3, 4], 5))

    @parametrize_with_cases(
        "wells, top, bottom, k", cases=AssignWellCases.case_mix_wells
    )
    def test_assign_wells_out_of_domain(self, wells, top, bottom, k):
        wells_xr = wells.to_xarray()
        multiplier = xr.DataArray(
            data=np.arange(1.0, 6.0),
            coords={"time": pd.date_range("2000-01-01", "2000-01-05")},
            dims=["time"],
        )
        wells_xr["rate"] = multiplier * wells_xr["rate"]
        transient_wells = wells_xr.to_dataframe().reset_index()

        actual = prepwel.assign_wells(
            wells=transient_wells, top=top, bottom=bottom, k=k, validate=False
        )
        assert np.array_equal(actual["id"], np.repeat([1, 2, 3, 3], 5))

        actual = prepwel.assign_wells(
            wells=transient_wells,
            top=top,
            bottom=bottom,
            k=k,
            minimum_thickness=1.01,
            validate=False,
        )
        assert np.array_equal(actual["id"], np.repeat([2, 3], 5))

    @parametrize_with_cases(
        "wells, top, bottom, k", cases=AssignWellCases.case_mix_wells
    )
    def test_assign_wells_out_of_domain_invalid(self, wells, top, bottom, k):
        with pytest.raises(ValueError, match="could not be mapped on the grid"):
            _ = prepwel.assign_wells(
                wells=wells,
                top=top,
                bottom=bottom,
                k=k,
            )

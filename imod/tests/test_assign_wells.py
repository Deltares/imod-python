import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas.testing import assert_frame_equal

import imod.prepare.wells as prepwel


def test_vectorized_overlap():
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
    actual = prepwel.vectorized_overlap(bounds_a, bounds_b)
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


class TestAssignWell:
    @pytest.fixture(autouse=True)
    def setup(self):
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
                "rate": [1.0, 10.0, 100.0, 0.0],
            }
        )

        self.wells = wells
        self.top = top
        self.bottom = bottom
        self.k = k

    def test_locate_wells__no_kh(self):
        id_in_bounds, xy_top, xy_bottom, xy_kh = prepwel.locate_wells(
            wells=self.wells,
            top=self.top,
            bottom=self.bottom,
            k=None,
        )

        assert np.array_equal(id_in_bounds, [1, 2, 3])
        assert np.allclose(xy_top, [[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]])
        assert np.allclose(xy_bottom, [[0.0, 0.0, 0.0], [-10.0, -10.0, -10.0]])
        assert xy_kh == 1.0

    def test_locate_wells(self):
        id_in_bounds, xy_top, xy_bottom, xy_kh = prepwel.locate_wells(
            wells=self.wells,
            top=self.top,
            bottom=self.bottom,
            k=self.k,
        )

        assert np.array_equal(id_in_bounds, [1, 2, 3])
        assert np.allclose(xy_top, [[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]])
        assert np.allclose(xy_bottom, [[0.0, 0.0, 0.0], [-10.0, -10.0, -10.0]])
        assert np.allclose(xy_kh, [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])

    def test_locate_wells_errors(self):
        with pytest.raises(TypeError, match="top and bottom"):
            prepwel.locate_wells(self.wells, self.top.values, self.bottom, None)
        with pytest.raises(ValueError, match="bottom grid does not match"):
            small_bottom = self.bottom.sel(y=slice(2.0, 0.0))
            prepwel.locate_wells(self.wells, self.top, small_bottom, None)
        with pytest.raises(ValueError, match="k grid does not match"):
            small_kh = self.k.sel(y=slice(2.0, 0.0))
            prepwel.locate_wells(self.wells, self.top, self.bottom, small_kh)

    def test_assign_wells_errors(self):
        with pytest.raises(ValueError, match="Columns are missing"):
            wells = pd.DataFrame({"id": [1], "x": [1.0], "y": [1.0]})
            prepwel.assign_wells(wells, self.top, self.bottom, self.k)
        with pytest.raises(TypeError, match="top, bottom, and optionally"):
            prepwel.assign_wells(self.wells, self.top, self.bottom.values)
        with pytest.raises(TypeError, match="top, bottom, and optionally"):
            prepwel.assign_wells(self.wells, self.top.values, self.bottom, self.k)

    def test_assign_wells__no_kh(self):
        actual = prepwel.assign_wells(
            wells=self.wells,
            top=self.top,
            bottom=self.bottom,
        )
        assert isinstance(actual, pd.DataFrame)
        expected = pd.DataFrame(
            {
                "index": [0, 1, 2, 3],
                "id": [1, 2, 3, 3],
                "layer": np.array([1, 1, 1, 2], dtype=np.int32),
                "bottom": [4.0, 2.0, -1.0, -1.0],
                "overlap": [1.0, 2.0, 3.0, 1.0],
                "rate": [1.0, 10.0, 75.0, 25.0],
                "top": [5.0, 4.0, 3.0, 3.0],
                "k": [1.0, 1.0, 1.0, 1.0],
                "transmissivity": [1.0, 2.0, 3.0, 1.0],
                "x": [0.6, 1.1, 2.3, 2.3],
                "y": [0.6, 1.1, 2.3, 2.3],
            }
        )
        assert_frame_equal(actual, expected, check_like=True)

    def test_assign_wells(self):
        actual = prepwel.assign_wells(
            wells=self.wells,
            top=self.top,
            bottom=self.bottom,
            k=self.k,
        )
        assert isinstance(actual, pd.DataFrame)
        expected = pd.DataFrame(
            {
                "index": [0, 1, 2, 3],
                "id": [1, 2, 3, 3],
                "layer": np.array([1, 1, 1, 2], dtype=np.int32),
                "bottom": [4.0, 2.0, -1.0, -1.0],
                "overlap": [1.0, 2.0, 3.0, 1.0],
                "rate": [1.0, 10.0, 60.0, 40.0],
                "top": [5.0, 4.0, 3.0, 3.0],
                "k": [10.0, 10.0, 10.0, 20.0],
                "transmissivity": [10.0, 20.0, 30.0, 20.0],
                "x": [0.6, 1.1, 2.3, 2.3],
                "y": [0.6, 1.1, 2.3, 2.3],
            }
        )
        expected["layer"] = expected["layer"].astype(np.int32)
        assert_frame_equal(actual, expected, check_like=True)

    def test_assign_wells_minimum_thickness(self):
        actual = prepwel.assign_wells(
            wells=self.wells,
            top=self.top,
            bottom=self.bottom,
            k=self.k,
            minimum_thickness=1.01,
        )
        assert isinstance(actual, pd.DataFrame)
        expected = pd.DataFrame(
            {
                "index": [0, 1],
                "id": [2, 3],
                "layer": [1, 1],
                "bottom": [2.0, -1.0],
                "overlap": [2.0, 3.0],
                "rate": [10.0, 100.0],
                "top": [4.0, 3.0],
                "k": [10.0, 10.0],
                "transmissivity": [20.0, 30.0],
                "x": [1.1, 2.3],
                "y": [1.1, 2.3],
            }
        )
        expected["layer"] = expected["layer"].astype(np.int32)
        assert_frame_equal(actual, expected, check_like=True)

    def test_assign_wells_transient_rate(self):
        wells_xr = self.wells.to_xarray()
        multiplier = xr.DataArray(
            data=np.arange(1.0, 6.0),
            coords={"time": pd.date_range("2000-01-01", "2000-01-05")},
            dims=["time"],
        )
        wells_xr["rate"] = multiplier * wells_xr["rate"]
        transient_wells = wells_xr.to_dataframe().reset_index()

        actual = prepwel.assign_wells(
            wells=transient_wells,
            top=self.top,
            bottom=self.bottom,
            k=self.k,
        )
        assert np.array_equal(actual["id"], np.repeat([1, 2, 3, 3], 5))

        actual = prepwel.assign_wells(
            wells=transient_wells,
            top=self.top,
            bottom=self.bottom,
            k=self.k,
            minimum_thickness=1.01,
        )
        assert np.array_equal(actual["id"], np.repeat([2, 3], 5))

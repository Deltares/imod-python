import pathlib
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry as sg
import xarray as xr

import imod


@pytest.fixture(scope="module")
def test_shapefile(request):
    geom = sg.Polygon([(0.0, 0.0), (1.1, 0.0), (1.1, 1.1), (0.0, 1.1)])
    gdf = gpd.GeoDataFrame()
    gdf.geometry = [geom]
    gdf["values"] = [2.0]
    pathlib.Path("testvector").mkdir(exist_ok=True)
    gdf.to_file("testvector/shape.shp")

    def teardown():
        try:
            shutil.rmtree("testvector")
        except FileNotFoundError:
            pass

    request.addfinalizer(teardown)


def test_round_extent():
    extent = (2.5, 2.5, 52.5, 52.5)
    cellsize = 5.0
    expected = (0.0, 0.0, 55.0, 55.0)
    assert imod.prepare.spatial.round_extent(extent, cellsize) == expected


def test_fill():
    da = xr.DataArray([np.nan, 1.0, 2.0], {"x": [1, 2, 3]}, ("x",))
    expected = xr.DataArray([1.0, 1.0, 2.0], {"x": [1, 2, 3]}, ("x",))
    actual = imod.prepare.spatial.fill(da)
    assert actual.identical(expected)

    # by dimension
    da = xr.DataArray(
        [[np.nan, np.nan, 1.0], [2.0, 2.0, 2.0]],
        {"x": [1, 2, 3], "y": [1, 2]},
        ("y", "x"),
    )
    expected_y = xr.DataArray(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], {"x": [1, 2, 3], "y": [1, 2]}, ("y", "x")
    )
    expected_x = xr.DataArray(
        [[2.0, 2.0, 1.0], [2.0, 2.0, 2.0]], {"x": [1, 2, 3], "y": [1, 2]}, ("y", "x")
    )
    actual_x = imod.prepare.spatial.fill(da, by="x")
    actual_y = imod.prepare.spatial.fill(da, by="y")
    assert actual_x.identical(expected_x)
    assert actual_y.identical(expected_y)


def test_rasterize():
    geom = sg.Polygon([(0.0, 0.0), (1.1, 0.0), (1.1, 1.1), (0.0, 1.1)])
    gdf = gpd.GeoDataFrame()
    gdf.geometry = [geom]
    gdf["values"] = [2.0]
    coords = {"y": [1.5, 0.5], "x": [0.5, 1.5]}
    dims = ("y", "x")
    like = xr.DataArray(np.full((2, 2), np.nan), coords, dims)
    # No column given, default to 1.0 where polygon
    expected = xr.DataArray([[np.nan, np.nan], [1.0, np.nan]], coords, dims)
    actual = imod.prepare.spatial.rasterize(gdf, like)
    assert actual.identical(expected)

    # Column given, use actual value
    expected = xr.DataArray([[np.nan, np.nan], [2.0, np.nan]], coords, dims)
    actual = imod.prepare.spatial.rasterize(gdf, like, column="values")
    assert actual.identical(expected)


def test_handle_dtype():
    assert imod.prepare.spatial._handle_dtype(np.uint8, None) == (1, 0)
    assert imod.prepare.spatial._handle_dtype(np.uint16, None) == (2, 0)
    assert imod.prepare.spatial._handle_dtype(np.int16, None) == (3, -32768)
    assert imod.prepare.spatial._handle_dtype(np.uint32, None) == (4, 0)
    assert imod.prepare.spatial._handle_dtype(np.int32, None) == (5, -2147483648)
    assert imod.prepare.spatial._handle_dtype(np.float32, None) == (6, np.nan)
    assert imod.prepare.spatial._handle_dtype(np.float64, None) == (7, np.nan)

    with pytest.raises(ValueError):  # out of bounds
        imod.prepare.spatial._handle_dtype(np.uint32, -1)
    with pytest.raises(ValueError):  # invalid dtype
        imod.prepare.spatial._handle_dtype(np.int64, -1)


def test_gdal_rasterize(test_shapefile):
    coords = {"y": [1.5, 0.5], "x": [0.5, 1.5]}
    dims = ("y", "x")
    like = xr.DataArray(np.full((2, 2), np.nan), coords, dims)
    spatial_reference = {"bounds": (0.0, 2.0, 0.0, 2.0), "cellsizes": (1.0, -1.0)}
    expected = xr.DataArray([[np.nan, np.nan], [2.0, np.nan]], coords, dims)

    # Test with like
    actual = imod.prepare.spatial.gdal_rasterize("testvector/shape.shp", "values", like)
    assert actual.identical(expected)

    # Test whether GDAL error results in a RuntimeError
    with pytest.raises(RuntimeError):  # misnamed column
        imod.prepare.spatial.gdal_rasterize("testvector/shape.shp", "value", like)

    # Can't determine dtype without like, raise ValueError
    with pytest.raises(ValueError):
        imod.prepare.spatial.gdal_rasterize(
            "testvector/shape.shp", "values", spatial_reference=spatial_reference
        )

    # Test without like
    actual = imod.prepare.spatial.gdal_rasterize(
        "testvector/shape.shp",
        "values",
        dtype=np.float64,
        spatial_reference=spatial_reference,
    )
    coords = {"y": [1.5, 0.5], "x": [0.5, 1.5], "dx": 1.0, "dy": -1.0}
    expected = xr.DataArray([[np.nan, np.nan], [2.0, np.nan]], coords, dims)
    assert actual.identical(expected)

    # Test integer dtype, and nodata default (0 for uint8)
    expected = xr.DataArray([[0, 0], [2, 0]], coords, dims)
    actual = imod.prepare.spatial.gdal_rasterize(
        "testvector/shape.shp",
        "values",
        dtype=np.uint8,
        spatial_reference=spatial_reference,
    )
    assert actual.identical(expected)

    # test with pathlib
    expected = xr.DataArray([[0, 0], [2, 0]], coords, dims)
    actual = imod.prepare.spatial.gdal_rasterize(
        pathlib.Path("testvector/shape.shp"),
        "values",
        dtype=np.uint8,
        spatial_reference=spatial_reference,
    )
    assert actual.identical(expected)


def test_table(test_shapefile):
    coords = {"y": [1.5, 0.5], "x": [0.5, 1.5]}
    dims = ("y", "x")
    like = xr.DataArray(np.full((2, 2), np.nan), coords, dims)

    expected = pd.DataFrame()
    expected["row_index"] = [1]
    expected["col_index"] = [0]
    expected["id"] = [2]
    expected["area"] = [1.0]

    actual = imod.prepare.spatial.cell_table(
        "testvector/shape.shp", "values", 1.0, like
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_rasterize_table():
    table = pd.DataFrame()
    table["row_index"] = [1]
    table["col_index"] = [0]
    table["id"] = [2]
    table["area"] = [1.0]

    coords = {"y": [1.5, 0.5], "x": [0.5, 1.5]}
    dims = ("y", "x")
    like = xr.DataArray(np.full((2, 2), np.nan), coords, dims)
    expected = xr.DataArray([[np.nan, np.nan], [1.0, np.nan]], coords, dims)

    actual = imod.prepare.spatial.rasterize_table(table, "area", like)
    assert actual.identical(expected)

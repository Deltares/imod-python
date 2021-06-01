import pathlib
import shutil

import geopandas as gpd
import imod
import numpy as np
import pandas as pd
import pytest
import shapely.geometry as sg
import xarray as xr


@pytest.fixture(scope="module")
def test_shapefile(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("testvector")
    geom = sg.Polygon([(0.0, 0.0), (1.1, 0.0), (1.1, 1.1), (0.0, 1.1)])
    gdf = gpd.GeoDataFrame()
    gdf.geometry = [geom]
    gdf["values"] = [2.0]
    gdf.to_file(tmp_dir / "shape.shp")
    return tmp_dir / "shape.shp"


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


def test_laplace_interpolate():
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = imod.util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "test", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)
    da = xr.DataArray(data, **kwargs)
    # remove some values
    da.values[:, 1] = np.nan
    actual = imod.prepare.laplace_interpolate(da, mxiter=5, iter1=30)
    assert np.allclose(actual.values, 1.0)

    ibound = xr.full_like(da, True)
    ibound.values[1, 1] = False
    actual = imod.prepare.laplace_interpolate(da, ibound=ibound, mxiter=5, iter1=30)
    assert np.isnan(actual.values[1, 1])
    actual.values[1, 1] = 1.0
    assert np.allclose(actual.values, 1.0)

    da.values[:] = 1.0
    da.values[1, :] = np.nan
    actual = imod.prepare.laplace_interpolate(da, ibound=ibound, mxiter=5, iter1=30)
    assert np.isnan(actual.values[1, 1])
    actual.values[1, 1] = 1.0
    assert np.allclose(actual.values, 1.0)


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


def test_polygonize():
    nrow, ncol = 2, 2
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 2.0
    ymin, ymax = 0.0, 2.0
    coords = imod.util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "test", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)
    data[0, 1] = 2.0
    data[1, 1] = 3.0
    da = xr.DataArray(data, **kwargs)
    gdf = imod.prepare.polygonize(da)
    assert len(gdf) == 3
    assert sorted(gdf["value"]) == [1.0, 2.0, 3.0]


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
    actual = imod.prepare.spatial.gdal_rasterize(test_shapefile, "values", like)
    assert actual.identical(expected)

    # Test with all_touched=True
    actual = imod.prepare.spatial.gdal_rasterize(
        test_shapefile, "values", like, all_touched=True
    )
    assert (actual == 2.0).all()

    # Test whether GDAL error results in a RuntimeError
    with pytest.raises(RuntimeError):  # misnamed column
        imod.prepare.spatial.gdal_rasterize(test_shapefile, "value", like)

    # Can't determine dtype without like, raise ValueError
    with pytest.raises(ValueError):
        imod.prepare.spatial.gdal_rasterize(
            test_shapefile, "values", spatial_reference=spatial_reference
        )

    # Test without like
    actual = imod.prepare.spatial.gdal_rasterize(
        test_shapefile, "values", dtype=np.float64, spatial_reference=spatial_reference
    )
    coords = {"y": [1.5, 0.5], "x": [0.5, 1.5], "dx": 1.0, "dy": -1.0}
    expected = xr.DataArray([[np.nan, np.nan], [2.0, np.nan]], coords, dims)
    assert actual.identical(expected)

    # Test integer dtype, and nodata default (0 for uint8)
    expected = xr.DataArray([[0, 0], [2, 0]], coords, dims)
    actual = imod.prepare.spatial.gdal_rasterize(
        test_shapefile, "values", dtype=np.uint8, spatial_reference=spatial_reference
    )
    assert actual.identical(expected)

    # test with pathlib
    expected = xr.DataArray([[0, 0], [2, 0]], coords, dims)
    actual = imod.prepare.spatial.gdal_rasterize(
        pathlib.Path(test_shapefile),
        "values",
        dtype=np.uint8,
        spatial_reference=spatial_reference,
    )
    assert actual.identical(expected)


def test_private_celltable(test_shapefile):
    coords = {"y": [1.5, 0.5], "x": [0.5, 1.5]}
    dims = ("y", "x")
    like = xr.DataArray(np.full((2, 2), np.nan), coords, dims)

    expected = pd.DataFrame()
    expected["row_index"] = [1]
    expected["col_index"] = [0]
    expected["values"] = [2]
    expected["area"] = [1.0]

    actual = imod.prepare.spatial._celltable(test_shapefile, "values", 1.0, like)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_celltable(test_shapefile):
    coords = {"y": [1.5, 0.5], "x": [0.5, 1.5]}
    dims = ("y", "x")
    like = xr.DataArray(np.full((2, 2), np.nan), coords, dims)

    expected = pd.DataFrame()
    expected["row_index"] = [1]
    expected["col_index"] = [0]
    expected["values"] = [2]
    expected["area"] = [1.0]

    actual = imod.prepare.spatial.celltable(test_shapefile, "values", 1.0, like)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    # test resolution error:
    with pytest.raises(ValueError):
        actual = imod.prepare.spatial.celltable(test_shapefile, "values", 0.17, like)


def test_rasterize_table():
    table = pd.DataFrame()
    table["row_index"] = [1]
    table["col_index"] = [0]
    table["values"] = [2]
    table["area"] = [1.0]

    coords = {"y": [1.5, 0.5], "x": [0.5, 1.5]}
    dims = ("y", "x")
    like = xr.DataArray(np.full((2, 2), np.nan), coords, dims)
    expected = xr.DataArray([[np.nan, np.nan], [1.0, np.nan]], coords, dims)

    actual = imod.prepare.spatial.rasterize_celltable(table, "area", like)
    assert actual.identical(expected)


def test_zonal_aggregate_raster(tmp_path):
    raster = xr.DataArray(
        data=[[0.0, 0.0], [1.0, 1.0]],
        coords={"x": [0.5, 1.5], "y": [1.5, 0.5]},
        dims=("y", "x"),
        name="my-raster",
    )
    geometries = [
        sg.Polygon(
            [
                (0.0, 1.0),
                (2.0, 1.0),
                (2.0, 2.0),
                (0.0, 2.0),
            ]
        ),
        sg.Polygon(
            [
                (0.0, 0.0),
                (2.0, 0.0),
                (2.0, 1.0),
                (0.0, 1.0),
            ]
        ),
    ]
    gdf = gpd.GeoDataFrame(geometry=geometries)
    gdf["id"] = [1, 2]
    path = tmp_path / "two-zones1.shp"
    gdf.to_file(path)

    actual = imod.prepare.spatial.zonal_aggregate_raster(
        path=path,
        column="id",
        raster=raster,
        resolution=1.0,
        method="mean",
    )
    expected = pd.DataFrame.from_dict(
        {
            "id": [1, 2],
            "area": [2.0, 2.0],
            "my-raster": [0.0, 1.0],
        }
    )
    assert actual.equals(expected)

    raster.name = None
    geometries = [
        sg.Polygon(
            [
                (0.0, 1.5),
                (2.0, 1.5),
                (2.0, 2.0),
                (0.0, 2.0),
            ]
        ),
        sg.Polygon(
            [
                (0.0, 0.0),
                (2.0, 0.0),
                (2.0, 1.5),
                (0.0, 1.5),
            ]
        ),
    ]
    gdf.geometry = geometries
    path = tmp_path / "two-zones2.shp"
    gdf.to_file(path)

    actual = imod.prepare.spatial.zonal_aggregate_raster(
        path=path,
        column="id",
        raster=raster,
        resolution=0.5,
        method="mean",
    )
    expected = pd.DataFrame.from_dict(
        {
            "id": [1, 2],
            "area": [1.0, 3.0],
            "aggregated": [0.0, 2.0 / 3.0],
        }
    )
    assert actual.equals(expected)

    actual = imod.prepare.spatial.zonal_aggregate_raster(
        path=path,
        column="id",
        raster=raster,
        resolution=0.5,
        method=pd.Series.mode,
    )
    expected = pd.DataFrame.from_dict(
        {
            "id": [1, 2],
            "area": [1.0, 3.0],
            "aggregated": [0.0, 1.0],
        }
    )
    assert actual.equals(expected)

    # Now test no overlap
    # The functions internally explicitly return a zero row dataframe
    # Since a method like pd.Series.mode cannot deal with 0 rows.
    raster = xr.DataArray(
        data=[[0.0, 0.0], [1.0, 1.0]],
        coords={"x": [10.5, 11.5], "y": [11.5, 10.5]},
        dims=("y", "x"),
        name="my-raster",
    )
    actual = imod.prepare.spatial.zonal_aggregate_raster(
        path=path,
        column="id",
        raster=raster,
        resolution=0.5,
        method=pd.Series.mode,
    )
    assert len(actual) == 0


def test_zonal_aggregate_polygons(tmp_path):
    raster = xr.DataArray(
        data=[[np.nan, np.nan], [np.nan, np.nan]],
        coords={"x": [0.5, 1.5], "y": [1.5, 0.5]},
        dims=("y", "x"),
    )
    geometries_a = [
        sg.Polygon(
            [
                (0.0, 1.0),
                (2.0, 1.0),
                (2.0, 2.0),
                (0.0, 2.0),
            ]
        ),
        sg.Polygon(
            [
                (0.0, 0.0),
                (2.0, 0.0),
                (2.0, 1.0),
                (0.0, 1.0),
            ]
        ),
    ]
    gdf_a = gpd.GeoDataFrame(geometry=geometries_a)
    gdf_a["id_a"] = [1, 2]
    gdf_a["data_a"] = [3, 4]
    path_a = tmp_path / "two-zones_a.shp"
    gdf_a.to_file(path_a)

    actual = imod.prepare.spatial.zonal_aggregate_polygons(
        path_a=path_a,
        column_a="id_a",
        path_b=path_a,
        column_b="data_a",
        like=raster,
        resolution=1.0,
        method="mean",
    )
    expected = pd.DataFrame.from_dict(
        {
            "id_a": [1, 2],
            "area": [2.0, 2.0],
            "data_a": [3.0, 4.0],
        }
    )
    assert actual.equals(expected)

    geometries_b = [
        sg.Polygon(
            [
                (0.0, 2.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 2.0),
            ]
        ),
        sg.Polygon(
            [
                (1.0, 2.0),
                (1.0, 0.0),
                (2.0, 0.0),
                (2.0, 2.0),
            ]
        ),
    ]
    gdf_b = gpd.GeoDataFrame(geometry=geometries_b)
    gdf_b["data_b"] = [3, 4]
    path_b = tmp_path / "two-zones_b.shp"
    gdf_b.to_file(path_b)

    actual = imod.prepare.spatial.zonal_aggregate_polygons(
        path_a=path_a,
        column_a="id_a",
        path_b=path_b,
        column_b="data_b",
        like=raster,
        resolution=1.0,
        method="mean",
    )
    expected = pd.DataFrame.from_dict(
        {
            "id_a": [1, 2],
            "area": [2.0, 2.0],
            "data_b": [3.5, 3.5],
        }
    )
    assert actual.equals(expected)

    # Now test no overlap
    # The functions internally explicitly return a zero row dataframe
    # Since a method like pd.Series.mode cannot deal with 0 rows.
    raster = xr.DataArray(
        data=[[0.0, 0.0], [1.0, 1.0]],
        coords={"x": [10.5, 11.5], "y": [11.5, 10.5]},
        dims=("y", "x"),
        name="my-raster",
    )
    actual = imod.prepare.spatial.zonal_aggregate_polygons(
        path_a=path_a,
        column_a="id_a",
        path_b=path_b,
        column_b="data_b",
        like=raster,
        resolution=1.0,
        method="mean",
    )
    assert len(actual) == 0

import affine
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xugrid as xu

import imod


@pytest.fixture(scope="function")
def ugrid_ds():
    vertices = np.array(
        [
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1
            [2.0, 0.0],  # 2
            [0.0, 1.0],  # 3
            [1.0, 1.0],  # 4
            [2.0, 1.0],  # 5
            [1.0, 2.0],  # 6
        ]
    )
    faces = np.array(
        [
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 6, -1],
            [4, 5, 6, -1],
        ]
    )
    grid = xu.Ugrid2d(
        node_x=vertices[:, 0],
        node_y=vertices[:, 1],
        fill_value=-1,
        face_node_connectivity=faces,
    )
    darray = xr.DataArray(
        data=np.random.rand(5, 3, grid.n_face),
        coords={"time": pd.date_range("2000-01-01", "2000-01-05"), "layer": [1, 2, 3]},
        dims=["time", "layer", grid.face_dimension],
    )
    ds = grid.to_dataset()
    ds["a"] = darray
    return ds


def test_transform():
    # implicit dx dy
    data = np.ones((2, 3))
    coords = {"x": [0.5, 1.5, 2.5], "y": [1.5, 0.5]}
    dims = ("y", "x")
    da = xr.DataArray(data, coords, dims)
    actual = imod.util.spatial.transform(da)
    expected = affine.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
    assert actual == expected

    # explicit dx dy, equidistant
    coords = {
        "x": [0.5, 1.5, 2.5],
        "y": [1.5, 0.5],
        "dx": ("x", [1.0, 1.0, 1.0]),
        "dy": ("y", [-1.0, -1.0]),
    }
    dims = ("y", "x")
    da = xr.DataArray(data, coords, dims)
    actual = imod.util.spatial.transform(da)
    assert actual == expected

    # explicit dx dy, non-equidistant
    coords = {
        "x": [0.5, 1.5, 3.5],
        "y": [1.5, 0.5],
        "dx": ("x", [1.0, 1.0, 2.0]),
        "dy": ("y", [-1.0, -1.0]),
    }
    dims = ("y", "x")
    da = xr.DataArray(data, coords, dims)
    with pytest.raises(ValueError):
        imod.util.spatial.transform(da)


def test_is_divisor():
    a = np.array([1.0, 0.5, 0.1])
    b = 0.05
    assert imod.util.spatial.is_divisor(a, b)
    assert imod.util.spatial.is_divisor(-a, b)
    assert imod.util.spatial.is_divisor(a, -b)
    assert imod.util.spatial.is_divisor(-a, -b)
    b = 0.07
    assert not imod.util.spatial.is_divisor(a, b)
    assert not imod.util.spatial.is_divisor(-a, b)
    assert not imod.util.spatial.is_divisor(a, -b)
    assert not imod.util.spatial.is_divisor(-a, -b)
    a = 3
    b = 1.5
    assert imod.util.spatial.is_divisor(-a, -b)


def test_empty():
    da = imod.util.spatial.empty_2d(1.0, 0.0, 2.0, -1.0, 10.0, 12.0)
    assert da.isnull().all()
    assert np.allclose(da["x"], [0.5, 1.5])
    assert np.allclose(da["y"], [11.5, 10.5])
    assert da.dims == ("y", "x")
    # Sign on dx, dy should be ignored
    da = imod.util.spatial.empty_2d(-1.0, 0.0, 2.0, 1.0, 10.0, 12.0)
    assert np.allclose(da["x"], [0.5, 1.5])
    assert np.allclose(da["y"], [11.5, 10.5])

    # array-like dx and dy
    da_irregular = imod.util.spatial.empty_2d(
        np.array([1, 2]), 0.0, 2.0, np.array([1, 2]), 10.0, 12.0
    )
    assert np.allclose(da_irregular["x"], [0.5, 2.0])
    assert np.allclose(da_irregular["y"], [11.5, 10.0])

    with pytest.raises(ValueError, match="layer must be 1d"):
        imod.util.spatial.empty_3d(1.0, 0.0, 2.0, -1.0, 10.0, 12.0, [[1, 2]])

    da3d = imod.util.spatial.empty_3d(1.0, 0.0, 2.0, -1.0, 10.0, 12.0, 1)
    assert da3d.ndim == 3
    da3d = imod.util.spatial.empty_3d(1.0, 0.0, 2.0, -1.0, 10.0, 12.0, [1, 3])
    assert np.array_equal(da3d["layer"], [1, 3])
    assert da3d.dims == ("layer", "y", "x")

    times = ["2000-01-01", "2001-01-01"]
    with pytest.raises(ValueError, match="time must be 1d"):
        imod.util.spatial.empty_2d_transient(1.0, 0.0, 2.0, -1.0, 10.0, 12.0, [times])

    da2dt = imod.util.spatial.empty_2d_transient(
        1.0, 0.0, 2.0, -1.0, 10.0, 12.0, times[0]
    )
    assert da2dt.ndim == 3
    da2dt = imod.util.spatial.empty_2d_transient(1.0, 0.0, 2.0, -1.0, 10.0, 12.0, times)
    assert isinstance(da2dt["time"].values[0], np.datetime64)
    assert da2dt.dims == ("time", "y", "x")

    da3dt = imod.util.spatial.empty_3d_transient(
        1.0, 0.0, 2.0, -1.0, 10.0, 12.0, [0, 1], times
    )
    assert da3dt.ndim == 4
    assert da3dt.dims == ("time", "layer", "y", "x")


def test_compliant_ugrid2d(ugrid_ds, write=False):
    uds = imod.util.spatial.mdal_compliant_ugrid2d(ugrid_ds)

    assert isinstance(uds, xr.Dataset)
    for i in range(1, 4):
        assert f"a_layer_{i}" in uds

    assert "mesh2d" in uds
    assert "mesh2d_face_nodes" in uds
    assert "mesh2d_node_x" in uds
    assert "mesh2d_node_y" in uds
    assert "mesh2d_nFaces" in uds.dims
    assert "mesh2d_nNodes" in uds.dims
    assert "mesh2d_nMax_face_nodes" in uds.dims
    attrs = uds["mesh2d"].attrs
    assert "face_coordinates" not in attrs

    assert uds["time"].encoding["dtype"] == np.float64

    if write:
        uds.to_netcdf("ugrid-mixed.nc")


def test_mdal_compliant_roundtrip(ugrid_ds):
    uds = xu.UgridDataset(imod.util.spatial.mdal_compliant_ugrid2d(ugrid_ds))
    uds["b"] = (("time", "layer"), np.random.rand(5, 3))
    uds["c"] = (("layer", "mesh2d_nFaces"), np.random.rand(3, 4))
    back = imod.util.spatial.from_mdal_compliant_ugrid2d(uds)

    assert isinstance(back, xu.UgridDataset)
    assert back["a"].dims == ("time", "layer", "mesh2d_nFaces")
    assert back["b"].dims == ("time", "layer")
    assert back["c"].dims == ("layer", "mesh2d_nFaces")
    assert np.array_equal(back["layer"], [1, 2, 3])


def test_to_ugrid2d(write=False):
    a2d = imod.util.spatial.empty_2d(
        dx=1.0,
        xmin=0.0,
        xmax=2.0,
        dy=1.0,
        ymin=0.0,
        ymax=2.0,
    )

    with pytest.raises(TypeError, match="data must be xarray"):
        imod.util.spatial.to_ugrid2d(a2d.values)
    with pytest.raises(ValueError, match="A name is required"):
        imod.util.spatial.to_ugrid2d(a2d)

    a2d.name = "a"
    with pytest.raises(ValueError, match="Last two dimensions of da"):
        imod.util.spatial.to_ugrid2d(a2d.transpose())

    uds = imod.util.spatial.to_ugrid2d(a2d)
    assert isinstance(uds, xr.Dataset)
    assert "a" in uds

    assert "mesh2d" in uds
    assert "mesh2d_face_nodes" in uds
    assert "mesh2d_node_x" in uds
    assert "mesh2d_node_y" in uds
    assert "mesh2d_nFaces" in uds.dims
    assert "mesh2d_nNodes" in uds.dims
    assert "mesh2d_nMax_face_nodes" in uds.dims
    attrs = uds["mesh2d"].attrs
    assert "face_coordinates" not in attrs

    if write:
        uds.to_netcdf("ugrid-a2d.nc")

    # 2d Dataset
    ds = xr.Dataset()
    ds["a"] = a2d
    ds["b"] = a2d.copy()
    uds = imod.util.spatial.to_ugrid2d(ds)
    assert "a" in uds
    assert "b" in uds
    assert isinstance(uds, xr.Dataset)

    if write:
        uds.to_netcdf("ugrid-a2d-ds.nc")

    # transient 2d
    a2dt = imod.util.spatial.empty_2d_transient(
        dx=1.0,
        xmin=0.0,
        xmax=2.0,
        dy=1.0,
        ymin=0.0,
        ymax=2.0,
        time=pd.date_range("2000-01-01", "2000-01-05"),
    )
    a2dt.name = "a"
    uds = imod.util.spatial.to_ugrid2d(a2dt)
    assert "a" in uds
    assert uds["time"].encoding["dtype"] == np.float64

    if write:
        uds.to_netcdf("ugrid-a2dt.nc")

    # 3d
    a3d = imod.util.spatial.empty_3d(
        dx=1.0,
        xmin=0.0,
        xmax=2.0,
        dy=1.0,
        ymin=0.0,
        ymax=2.0,
        layer=[1, 2, 3],
    )
    a3d.name = "a"
    uds = imod.util.spatial.to_ugrid2d(a3d)
    assert isinstance(uds, xr.Dataset)
    for i in range(1, 4):
        assert f"a_layer_{i}" in uds

    if write:
        uds.to_netcdf("ugrid-a3d.nc")

    # transient 3d
    a3dt = imod.util.spatial.empty_3d_transient(
        dx=1.0,
        xmin=0.0,
        xmax=2.0,
        dy=1.0,
        ymin=0.0,
        ymax=2.0,
        layer=[1, 2, 3],
        time=pd.date_range("2000-01-01", "2000-01-05"),
    )
    a3dt.name = "a"
    uds = imod.util.spatial.to_ugrid2d(a3dt)
    for i in range(1, 4):
        assert f"a_layer_{i}" in uds
    assert uds["time"].encoding["dtype"] == np.float64

    if write:
        uds.to_netcdf("ugrid-a3dt.nc")


def test_gdal_compliant_grid():
    # Arrange
    data = np.ones((2, 3))
    # explicit dx dy, equidistant
    coords = {
        "x": [0.5, 1.5, 2.5],
        "y": [1.5, 0.5],
        "dx": ("x", [1.0, 1.0, 1.0]),
        "dy": ("y", [-1.0, -1.0]),
    }
    dims = ("y", "x")
    da = xr.DataArray(data, coords, dims)

    # Act
    da_compliant = imod.util.spatial.gdal_compliant_grid(da)

    # Assert
    # Test if original dataarray not altered
    assert da.coords["x"].attrs == {}
    assert da.coords["y"].attrs == {}
    # Test if coordinates got correct attributes
    assert da_compliant.coords["x"].attrs["axis"] == "X"
    assert da_compliant.coords["x"].attrs["long_name"] == "x coordinate of projection"
    assert da_compliant.coords["x"].attrs["standard_name"] == "projection_x_coordinate"
    assert da_compliant.coords["y"].attrs["axis"] == "Y"
    assert da_compliant.coords["y"].attrs["long_name"] == "y coordinate of projection"
    assert da_compliant.coords["y"].attrs["standard_name"] == "projection_y_coordinate"


def test_gdal_compliant_grid_error():
    # Arrange
    data = np.ones((2,))
    # explicit dx dy, equidistant
    coords = {
        "y": [1.5, 0.5],
        "dy": ("y", [-1.0, -1.0]),
    }
    dims = ("y",)
    da = xr.DataArray(data, coords, dims)

    # Act
    with pytest.raises(ValueError, match="Missing dimensions: {'x'}"):
        imod.util.spatial.gdal_compliant_grid(da)
import numpy as np
import xarray as xr
import xugrid as xu

from imod.typing.grid import (
    UGRID2D_FROM_STRUCTURED_CACHE,
    GridCache,
    as_ugrid_dataarray,
    enforce_dim_order,
    is_planar_grid,
    is_spatial_grid,
    is_transient_data_grid,
    merge_with_dictionary,
    preserve_gridtype,
)


def test_preserve_gridtype__single_output(basic_unstructured_dis):
    uda, _, da = basic_unstructured_dis

    @preserve_gridtype
    def to_be_decorated(a, b):
        return a * b

    result1 = to_be_decorated(uda, da)
    result2 = to_be_decorated(da, uda)

    # Verify fixture provides expected type
    assert isinstance(da, xr.DataArray)
    assert isinstance(uda, xu.UgridDataArray)

    assert isinstance(result1, xu.UgridDataArray)
    assert isinstance(result2, xu.UgridDataArray)


def test_preserve_gridtype__multiple_outputs(basic_unstructured_dis):
    uda, _, da = basic_unstructured_dis

    @preserve_gridtype
    def to_be_decorated(a, b):
        return a * b, a + b

    result1a, result1b = to_be_decorated(uda, da)
    result2a, result2b = to_be_decorated(da, uda)

    # Verify fixture provides expected type
    assert isinstance(da, xr.DataArray)
    assert isinstance(uda, xu.UgridDataArray)

    assert isinstance(result1a, xu.UgridDataArray)
    assert isinstance(result1b, xu.UgridDataArray)
    assert isinstance(result2a, xu.UgridDataArray)
    assert isinstance(result2b, xu.UgridDataArray)


def test_enforce_dim_order__structured(basic_dis):
    ibound, _, _ = basic_dis

    ibound_wrong_order = ibound.transpose("x", "y", "layer")

    actual = enforce_dim_order(ibound_wrong_order)

    assert actual.dims == ibound.dims
    assert isinstance(actual, type(ibound))


def test_enforce_dim_order__unstructured(basic_unstructured_dis):
    ibound, _, _ = basic_unstructured_dis

    face_dim = ibound.ugrid.grid.face_dimension

    ibound_wrong_order = ibound.transpose(face_dim, "layer")

    actual = enforce_dim_order(ibound_wrong_order)

    assert actual.dims == ibound.dims
    assert isinstance(actual, type(ibound))


def test_is_planar_grid(basic_dis, basic_unstructured_dis):
    discretizations = [basic_dis, basic_unstructured_dis]
    for discr in discretizations:
        ibound, _, _ = discr

        # layer coordinates is present
        assert not is_planar_grid(ibound)

        # set layer coordinates as present but empty
        bottom_layer = ibound.sel(layer=3)
        assert is_planar_grid(bottom_layer)

        # set layer coordinates as  present and not  empty or 0
        bottom_layer = bottom_layer.expand_dims({"layer": [9]})
        assert not is_planar_grid(bottom_layer)

        # set layer coordinates as  present and   0
        bottom_layer.coords["layer"].values[0] = 0
        assert is_planar_grid(bottom_layer)


def test_is_spatial_grid__structured(basic_dis):
    ibound, _, bottom = basic_dis
    ds = xr.Dataset()
    ds["ibound"] = ibound
    ds["bottom"] = bottom

    assert is_spatial_grid(ibound)
    assert not is_spatial_grid(bottom)
    assert is_spatial_grid(ds)


def test_is_transient_data_grid(basic_dis, basic_unstructured_dis):
    discretizations = [basic_dis, basic_unstructured_dis]

    for discr in discretizations:
        ibound, _, _ = discr

        # no time coordinate
        assert not is_transient_data_grid(ibound)

        #  time coordinate but with single value
        ibound = ibound.expand_dims({"time": [1]})
        assert not is_transient_data_grid(ibound)

        #  time coordinate but with several values
        ibound, _, _ = discr
        ibound = ibound.expand_dims({"time": [1, 2]})
        assert is_transient_data_grid(ibound)


def test_is_spatial_grid__unstructured(basic_unstructured_dis):
    ibound, _, bottom = basic_unstructured_dis
    grid = ibound.ugrid.grid
    ds = xr.Dataset()
    # For some reason xarray requires explicit dimnames in this case to assign
    # to dataset.
    ds["ibound"] = (("layer", "mesh2d_nFaces"), ibound)
    ds["bottom"] = bottom
    _ = xu.UgridDataset(ds, grid)


def test_merge_dictionary__unstructured(basic_unstructured_dis):
    ibound, _, bottom = basic_unstructured_dis

    uds = merge_with_dictionary({"ibound": ibound, "bottom": bottom})

    assert isinstance(uds, xu.UgridDataset)
    assert isinstance(uds["ibound"], xu.UgridDataArray)
    assert isinstance(uds["bottom"], xr.DataArray)
    assert uds["ibound"].dims == ("layer", "mesh2d_nFaces")
    assert uds["bottom"].dims == ("layer",)


def test_as_ugrid_dataarray__structured(basic_dis):
    # Arrange
    ibound, top, bottom = basic_dis
    top_3d = top * ibound
    bottom_3d = bottom * ibound
    # Clear cache
    UGRID2D_FROM_STRUCTURED_CACHE.clear()
    # Act
    ibound_disv = as_ugrid_dataarray(ibound)
    top_disv = as_ugrid_dataarray(top_3d)
    bottom_disv = as_ugrid_dataarray(bottom_3d)
    # Assert
    # Test types
    assert isinstance(ibound_disv, xu.UgridDataArray)
    assert isinstance(top_disv, xu.UgridDataArray)
    assert isinstance(bottom_disv, xu.UgridDataArray)
    # Test cache proper size
    assert len(UGRID2D_FROM_STRUCTURED_CACHE.grid_cache) == 1
    # Test that data is different
    assert np.all(ibound_disv != top_disv)
    assert np.all(top_disv != bottom_disv)
    # Test that grid is equal
    assert np.all(ibound_disv.grid == top_disv.grid)
    assert np.all(top_disv.grid == bottom_disv.grid)


def test_as_ugrid_dataarray__unstructured(basic_unstructured_dis):
    # Arrange
    ibound, top, bottom = basic_unstructured_dis
    top_3d = enforce_dim_order(ibound * top)
    bottom_3d = enforce_dim_order(ibound * bottom)
    # Clear cache
    UGRID2D_FROM_STRUCTURED_CACHE.clear()
    # Act
    ibound_disv = as_ugrid_dataarray(ibound)
    top_disv = as_ugrid_dataarray(top_3d)
    bottom_disv = as_ugrid_dataarray(bottom_3d)
    # Assert
    # Test types
    assert isinstance(ibound_disv, xu.UgridDataArray)
    assert isinstance(top_disv, xu.UgridDataArray)
    assert isinstance(bottom_disv, xu.UgridDataArray)
    assert len(UGRID2D_FROM_STRUCTURED_CACHE.grid_cache) == 0


def test_ugrid2d_cache(basic_dis):
    # Arrange
    ibound, _, _ = basic_dis
    # Act
    cache = GridCache(xu.Ugrid2d.from_structured, max_cache_size=3)
    for i in range(5):
        ugrid2d = cache.get_grid(ibound[:, i:, :])
    # Assert
    # Test types
    assert isinstance(ugrid2d, xu.Ugrid2d)
    # Test cache proper size
    assert cache.max_cache_size == 3
    assert len(cache.grid_cache) == 3
    # Check if smallest grid in last cache list by checking if amount of faces
    # correct
    expected_size = ibound[0, i:, :].size
    keys = list(cache.grid_cache.keys())
    last_ugrid = cache.grid_cache[keys[-1]]
    actual_size = last_ugrid.n_face
    assert expected_size == actual_size
    # Test clear cache
    cache.clear()
    assert len(cache.grid_cache) == 0

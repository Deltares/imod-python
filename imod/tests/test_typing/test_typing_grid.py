import xarray as xr
import xugrid as xu

from imod.typing.grid import (
    enforce_dim_order,
    is_planar_grid,
    is_transient_data_grid,
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


def test_is_transient_grid(basic_dis, basic_unstructured_dis):
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

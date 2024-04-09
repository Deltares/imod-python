import xarray as xr
import xugrid as xu

from imod.typing.grid import enforce_dim_order, preserve_gridtype


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

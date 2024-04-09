from imod.util.dims import enforced_dim_order


def test_enforced_dim_order_structured(basic_dis):
    ibound, _, _ = basic_dis

    @enforced_dim_order
    def to_be_decorated(da):
        return da
    
    ibound_wrong_order = ibound.transpose("x", "y", "layer")

    actual = to_be_decorated(ibound_wrong_order)

    assert actual.dims == ibound.dims
    assert isinstance(actual, type(ibound))


def test_enforced_dim_order_unstructured(basic_unstructured_dis):
    ibound, _, _ = basic_unstructured_dis

    @enforced_dim_order
    def to_be_decorated(da):
        return da
    
    face_dim = ibound.ugrid.grid.face_dimension

    ibound_wrong_order = ibound.transpose(face_dim, "layer")

    actual = to_be_decorated(ibound_wrong_order)

    assert actual.dims == ibound.dims
    assert isinstance(actual, type(ibound))
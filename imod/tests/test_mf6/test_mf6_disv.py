import numpy as np
import xarray as xr
import xugrid as xu
import pytest

import imod
from imod.schemata import ValidationError


@pytest.fixture(scope="function")
def idomain_and_bottom():
    diskgrid = xu.data.disk().ugrid.grid

    nlayer = 3
    nface = diskgrid.n_face
    layer = np.array([1, 2, 3])
    coords = {"layer": layer}
    dims = ["layer", diskgrid.face_dimension]
    idomain = xu.UgridDataArray(
        xr.DataArray(np.ones((nlayer, nface), dtype=int), coords=coords, dims=dims),
        diskgrid,
    )
    bottom = (0.0 - idomain.cumsum("layer")).astype(float)
    return idomain, bottom


def test_zero_thickness_validation(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom
    # Create a bottom that has constant values of -1.0 so that thickness is 0.
    bottom = (bottom * 0.0) - 1.0
    disv = imod.mf6.VerticesDiscretization(top=0.0, bottom=bottom, idomain=idomain)

    errors = disv._validate(disv._write_schemata, idomain=idomain)
    assert len(errors) == 1
    error = errors["bottom"][0]
    assert isinstance(error, ValidationError)
    assert error.args[0] == "found thickness <= 0.0"

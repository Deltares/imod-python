import numpy as np
import pytest
import xarray as xr
import xugrid as xu

import imod


@pytest.fixture(scope="function")
def basic_unstructured_dis(basic_dis):
    idomain, top, bottom = basic_dis
    idomain_ugrid = xu.UgridDataArray.from_structured(idomain)
    top_mf6 = top.sel(layer=1)  # Top for modlfow 6 shouldn't contain layer dim

    return idomain_ugrid, top_mf6, bottom


@pytest.fixture(scope="module")
def circle_dis():
    """Basic model discretization"""

    grid_triangles = imod.data.circle()
    grid = grid_triangles.tesselate_centroidal_voronoi()

    nface = grid.n_face
    nlayer = 15

    layer = np.arange(nlayer, dtype=int) + 1

    idomain = xu.UgridDataArray(
        xr.DataArray(
            np.ones((nlayer, nface), dtype=np.int32),
            coords={"layer": layer},
            dims=["layer", grid.face_dimension],
        ),
        grid=grid,
    )
    top = 0.0
    bottom = xr.DataArray(top - (layer * 10.0), dims=["layer"])

    return idomain, top, bottom

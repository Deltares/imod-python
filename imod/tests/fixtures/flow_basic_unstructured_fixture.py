import numpy as np
import pytest
import xarray as xr
import xugrid as xu

import imod


@pytest.fixture(scope="module")
def basic_unstructured_dis():
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

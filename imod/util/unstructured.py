import numpy as np
import xarray as xr
import xugrid as xu


def ones_like_ugrid(grid: xu.Ugrid2d) -> xu.UgridDataArray:
    """
    Create an UgridDataArray of ones with the same shape and coordinates as the
    given Ugrid2d.

    Parameters
    ----------
    grid : xu.Ugrid2d
        The unstructured grid to create an array for.

    Returns
    -------
    xu.UgridDataArray
        An array of ones with the same shape and coordinates as the grid.
    """
    face_dim = grid.face_dimension
    data = xr.DataArray(np.ones(grid.sizes[face_dim]), dims=[face_dim])
    return xu.UgridDataArray(data, grid)

def unstructured_package_dim_check(pkgname, da):
    """
    Check dimension integrity of unstructured grid,
    no time dimension is accepted, data is assumed static.
    """

    msg_end = f"Instead, got {da.dims} for {da.name} in the {pkgname} package. "

    if da.ndim == 0:
        return  # Scalar, no check necessary
    elif da.ndim == 1:
        face_dim = da.ugrid.grid.face_dimension
        if (face_dim not in da.dims) and ("layer" not in da.dims):
            raise ValueError(
                f"Face dimension '{face_dim}' or dimension 'layer' "
                f"not found in 1D UgridDataArray. " + msg_end
            )
    elif da.ndim == 2:
        face_dim = da.ugrid.grid.face_dimension
        if da.dims != ("layer", face_dim):
            raise ValueError(
                f"2D grid should have dimensions ('layer', {face_dim})" + msg_end
            )


def structured_package_dim_check(pkgname, da):
    """
    Check dimension integrity of structured grid,
    no time dimension is accepted, data is assumed static.
    """

    msg_end = f"Instead, got {da.dims} for {da.name} in the {pkgname} package. "

    if da.ndim == 0:
        return  # Scalar, no check necessary
    elif da.ndim == 1:
        if "layer" not in da.dims:
            raise ValueError("1D DataArray dims can only be ('layer',). " + msg_end)
    elif da.ndim == 2:
        if da.dims != ("y", "x"):
            raise ValueError("2D grid should have dimensions ('y', 'x'). " + msg_end)
    elif da.ndim == 3:
        if da.dims != ("layer", "y", "x"):
            raise ValueError(
                "3D grid should have dimensions ('layer', 'y', 'x'). " + msg_end
            )
    else:
        raise ValueError(
            f"Exceeded accepted amount of dimensions for "
            f"for {da.name} in the "
            f"{pkgname} package. "
            f"Got {da.dims}. Can be at max ('layer', 'y', 'x')."
        )


def unstructured_boundary_condition_dim_check(pkgname, da):
    """
    Check dimension integrity of unstructured grid,
    no time dimension is accepted, data is assumed static.
    """

    msg_end = f"Instead, got {da.dims} for {da.name} in the {pkgname} package. "

    if da.ndim < 1:
        raise ValueError(
            "Boundary conditions should be specified as spatial grids. " + msg_end
        )
    elif da.ndim == 1:
        face_dim = da.ugrid.grid.face_dimension
        if face_dim not in da.dims:
            raise ValueError(
                f"Face dimension '{face_dim}' not found in "
                f"1D UgridDataArray. " + msg_end
            )
    elif da.ndim == 2:
        face_dim = da.ugrid.grid.face_dimension
        if (da.dims != ("layer", face_dim)) and (da.dims != ("time", face_dim)):
            raise ValueError(
                f"2D grid should have dimensions ('layer', {face_dim}) "
                f"or ('time', {face_dim}). " + msg_end
            )
    elif da.ndim == 3:
        face_dim = da.ugrid.grid.face_dimension
        if da.dims != ("time", "layer", face_dim):
            raise ValueError(
                f"3D grid should have dimensions ('time', 'layer', {face_dim}) "
                + msg_end
            )


def structured_boundary_condition_dim_check(pkgname, da):
    """
    Check dimension integrity of structured grid
    """

    msg_end = f"Instead, got {da.dims} for {da.name} in the {pkgname} package. "

    if da.ndim < 2:
        raise ValueError(
            "Boundary conditions should be specified as spatial grids. " + msg_end
        )
    elif da.ndim == 2:
        if da.dims != ("y", "x"):
            raise ValueError("2D grid should have dimensions ('y', 'x'). " + msg_end)
        if "layer" not in da.coords:
            raise ValueError(
                f"No 'layer' coordinate assigned to {da.name} "
                f"in the {pkgname} package. "
                f"2D grids still require a 'layer' coordinate. "
                f"You can assign one with `da.assign_coordinate(layer=1)`"
            )
    elif da.ndim == 3:
        if (da.dims != ("layer", "y", "x")) and (da.dims != ("time", "y", "x")):
            raise ValueError(
                "3D grid should have dimensions ('layer', 'y', 'x') "
                "or ('time', 'y', 'x'). " + msg_end
            )
    elif da.ndim == 4:
        if da.dims != ("time", "layer", "y", "x"):
            raise ValueError(
                "4D grid should have dimensions ('time', 'layer', 'y', 'x'). " + msg_end
            )
    else:
        raise ValueError(
            f"Exceeded accepted amount of dimensions for "
            f"for {da.name} in the "
            f"{pkgname} package. "
            f"Got {da.dims}. Can be at max ('time', 'layer', 'y', 'x')."
        )


def check_dim_monotonicity(pkgname, ds):
    for dim in ["x", "layer", "time"]:
        if dim in ds.indexes:
            if not ds.indexes[dim].is_monotonic_increasing:
                raise ValueError(
                    f"{dim} coordinate in {pkgname} not monotonically increasing"
                )

    if "y" in ds.indexes:
        if not ds.indexes["y"].is_monotonic_decreasing:
            raise ValueError(f"y coordinate in {pkgname} not monotonically decreasing")

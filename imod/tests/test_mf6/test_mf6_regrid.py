import numpy as np
import pytest
import xarray as xr
import xugrid as xu

import imod


def grid_data_structured(dtype, value, cellsize) -> xr.DataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size.
    """
    horizontal_range = 10
    y = np.arange(horizontal_range, -cellsize, -cellsize)
    x = np.arange(0, horizontal_range + cellsize, cellsize)

    nlayer = 3

    shape = nlayer, len(x), len(y)
    dims = ("layer", "y", "x")
    layer = np.arange(1, nlayer + 1)

    coords = {"layer": layer, "y": y, "x": x, "dx": cellsize, "dy": cellsize}

    da = xr.DataArray(np.ones(shape, dtype=dtype) * value, coords=coords, dims=dims)

    return da


def grid_data_structured_layered(dtype, value, cellsize) -> xr.DataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size. The values are
    multiplied with the layer index.
    """
    horizontal_range = 10
    y = np.arange(horizontal_range, -cellsize, -cellsize)
    x = np.arange(0, horizontal_range + cellsize, cellsize)

    nlayer = 3

    shape = nlayer, len(x), len(y)
    dims = ("layer", "y", "x")
    layer = np.arange(1, nlayer + 1)

    coords = {"layer": layer, "y": y, "x": x, "dx": cellsize, "dy": cellsize}

    da = xr.DataArray(np.ones(shape, dtype=dtype), coords=coords, dims=dims)
    for ilayer in range(1, nlayer + 1):
        layer_value = ilayer * value
        da.loc[dict(layer=ilayer)] = layer_value
    return da


def grid_data_unstructured(dtype, value, cellsize) -> xu.UgridDataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size.
    First a regular grid is constructed and then this is converted to an ugrid dataarray.
    """
    return xu.UgridDataArray.from_structured(
        grid_data_structured(dtype, value, cellsize)
    )


def grid_data_unstructured_layered(dtype, value, cellsize) -> xu.UgridDataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size. The values are
    multiplied with the layer index. First a regular grid is constructed and then this is converted to an ugrid dataarray.
    """
    return xu.UgridDataArray.from_structured(
        grid_data_structured_layered(dtype, value, cellsize)
    )


def create_package_instances(is_structured):
    grid_data_function = (
        grid_data_structured if is_structured else grid_data_unstructured
    )
    grid_data_function_layered = (
        grid_data_structured_layered
        if is_structured
        else grid_data_unstructured_layered
    )

    packages = [
        imod.mf6.NodePropertyFlow(
            icelltype=grid_data_function(np.int_, 1, 5.0),
            k=grid_data_function(np.float64, 12, 5.0),
            k22=3.0,
        ),
        imod.mf6.SpecificStorage(
            specific_storage=grid_data_function(np.float_, 1.0e-4, 5.0),
            specific_yield=grid_data_function(np.float_, 0.15, 5.0),
            convertible=0,
            transient=False,
        ),
        imod.mf6.StorageCoefficient(
            storage_coefficient=grid_data_function(np.float_, 1.0e-4, 5.0),
            specific_yield=grid_data_function(np.float_, 0.15, 5.0),
            convertible=grid_data_function(np.int32, 0, 5.0),
            transient=True,
        ),
        imod.mf6.Drainage(
            elevation=grid_data_function(np.float_, 1.0e-4, 5.0),
            conductance=grid_data_function(np.float_, 1.0e-4, 5.0),
            print_input=True,
            print_flows=True,
            save_flows=True,
        ),
        imod.mf6.ConstantHead(
            grid_data_function(np.float_, 1.0e-4, 5.0),
            print_input=True,
            print_flows=True,
            save_flows=True,
        ),
        imod.mf6.GeneralHeadBoundary(
            head=grid_data_function(np.float_, 1.0e-4, 5.0),
            conductance=grid_data_function(np.float_, 1.0e-4, 5.0),
        ),
    ]
    if is_structured:
        packages.append(
            imod.mf6.StructuredDiscretization(
                top=20.0,
                bottom=grid_data_function_layered(np.float_, -1, 5.0),
                idomain=grid_data_function(np.int_, 1, 5.0),
            )
        )
    else:
        packages.append(
            imod.mf6.VerticesDiscretization(
                top=20.0,
                bottom=grid_data_function_layered(np.float_, -1, 5.0),
                idomain=grid_data_function(np.int_, 1, 5.0),
            )
        )
    return packages


def test_regrid_structured():
    """
    This test regrids a structured grid to another structured grid of the same size.
    Some of the arrays are entered as grids and others as scalars
    """
    structured_grid_packages = create_package_instances(is_structured=True)
    new_grid = grid_data_structured(np.float64, 12, 2.5)
    new_packages = []
    for package in structured_grid_packages:
        new_packages.append(package.regrid_like(new_grid))

    new_idomain = new_packages[0].dataset["icelltype"]
    for new_package in new_packages:
        errors = new_package._validate(
            new_package._write_schemata,
            idomain=new_idomain,
        )
        assert len(errors) == 0


def test_regrid_unstructured():
    """
    This test regrids a structured grid to another structured grid of the same size.
    Some of the arrays are entered as grids and others as scalars
    """
    unstructured_grid_packages = create_package_instances(is_structured=False)
    new_grid = grid_data_unstructured(np.float64, 12, 2.5)
    new_packages = []
    for package in unstructured_grid_packages:
        new_packages.append(package.regrid_like(new_grid))

    new_idomain = new_packages[0].dataset["icelltype"]
    for new_package in new_packages:
        # package write validation crashes for VerticesDiscretization so we skip that one
        if type(new_package).__name__ != "VerticesDiscretization":
            errors = new_package._validate(
                new_package._write_schemata,
                idomain=new_idomain,
            )
            assert len(errors) == 0
        else:
            continue


def test_regrid_structured_missing_dx_and_dy():
    """
    In imod-python it is not mandatory for data-arrays to have a dx and dy coordinate, but for
    xugrid this is mandatory
    """
    icelltype = grid_data_structured(np.int_, 1, 0.5)
    icelltype = icelltype.drop_vars(["dx", "dy"])
    package = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=2.0,
        k22=3.0,
    )

    new_grid = grid_data_structured(np.float64, 12, 0.25)

    with pytest.raises(
        ValueError,
        match="DataArray icelltype does not have both a dx and dy coordinates",
    ):
        _ = package.regrid_like(new_grid)


def test_regrid(tmp_path):
    """
    This test regrids an irregular grid. However, the new grid is the same as the source grid, so the values
    of the data-arrays should not change.
    """
    grid = imod.data.circle()
    nlayer = 5

    nface = grid.n_face
    layer = np.arange(nlayer, dtype=int) + 1
    k_value = 10.0
    idomain = xu.UgridDataArray(
        xr.DataArray(
            np.ones((nlayer, nface), dtype=np.int32),
            coords={"layer": layer},
            dims=["layer", grid.face_dimension],
        ),
        grid=grid,
    )
    k = xu.full_like(idomain, k_value, dtype=float)

    npf = imod.mf6.NodePropertyFlow(
        icelltype=idomain,
        k=k,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=True,
        save_flows=True,
        alternative_cell_averaging="AMT-HMK",
    )

    new_npf = npf.regrid_like(k)

    # check the rendered versions are the same, they contain the options
    new_rendered = new_npf.render(tmp_path, "regridded", None, False)
    original_rendered = npf.render(tmp_path, "original", None, False)

    new_rendered = new_rendered.replace("regridded", "original")
    assert new_rendered == original_rendered

    # check the arrays
    k_new = new_npf.dataset["k"]
    k_diff = k_new - k
    max_diff = k_diff.max().values[()]
    min_diff = k_diff.min().values[()]
    abs_tol = 1e-13

    assert abs(min_diff) < abs_tol and abs(max_diff) < abs_tol


def test_regrid_not_supported():
    """
    This tests that regridding a package for which it is not implemented does noty lead to a crash
    """
    grid = imod.data.circle()
    nlayer = 5

    nface = grid.n_face
    layer = np.arange(nlayer, dtype=int) + 1
    dispersivity = 1
    idomain = xu.UgridDataArray(
        xr.DataArray(
            np.ones((nlayer, nface), dtype=np.int32),
            coords={"layer": layer},
            dims=["layer", grid.face_dimension],
        ),
        grid=grid,
    )
    disp = xu.full_like(idomain, dispersivity, dtype=float)

    disperion_package = imod.mf6.Dispersion(1e-4, disp, disp)
    with pytest.raises(
        NotImplementedError, match="this package does not support regridding"
    ):
        disperion_package.regrid_like(idomain)

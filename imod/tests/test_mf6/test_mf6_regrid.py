import numpy as np
import pytest
import xarray as xr
import xugrid as xu

import imod

def grid_structured(dtype, value, cellsize)-> xr.DataArray:
    """
    This function creates a dataarray with scalar values for a grid of 3 layers and 9 rows and columns.
    """
    horizontal_range = 10
    y = np.arange(horizontal_range, -cellsize, -cellsize) 
    x = np.arange(0, horizontal_range+cellsize, cellsize) 

    nlayer = 3

    shape = nlayer, len(x), len(y) 
    dims = ("layer", "y", "x")
    layer = np.arange(1, nlayer + 1)

    coords = {"layer": layer, "y": y, "x": x, "dx":cellsize, "dy": cellsize}

    da = xr.DataArray(np.ones(shape, dtype=dtype) * value, coords=coords, dims=dims)

    return da

def test_regrid_structured():
    """
    This test regrids a structured grid to another structured grid of the same size.
    Some of the arrays are entered as grids and others as scalars
    """
    package = imod.mf6.NodePropertyFlow(        
        icelltype = grid_structured(np.int_, 1, 5.0),
        k = grid_structured(np.float64, 12, 5.0), 
        k22 = 3.0,)

    new_grid = grid_structured(np.float64, 12, 2.5)


    new_package = package.regrid_like(new_grid)
    errors = new_package._validate(imod.mf6.NodePropertyFlow._write_schemata,  idomain=new_package.dataset["icelltype"],)
    assert len(errors) == 0

def test_regrid_structured_missing_dx_and_dy():
    '''
    In imod-python it is not mandatory for data-arrays to have a dx and dy coordinate, but for
    xugrid this is mandatory
    '''
    icelltype = grid_structured(np.int_, 1, 0.5)
    icelltype= icelltype.drop_vars(["dx", "dy"])
    package = imod.mf6.NodePropertyFlow(        
        icelltype = icelltype,
        k = 2.0, 
        k22 = 3.0,)

    new_grid = grid_structured(np.float64, 12, 0.25)

    with pytest.raises(ValueError, match="DataArray icelltype does not have both a dx and dy coordinates"):
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

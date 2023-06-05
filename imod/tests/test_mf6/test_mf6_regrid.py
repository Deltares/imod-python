from pathlib import Path
from typing import List

import numpy as np
import pytest
import xarray as xr
import xugrid as xu

import imod
from imod.mf6.pkgbase import Package
from imod.tests.fixtures.mf6_regridding_fixture import (
    grid_data_structured,
    grid_data_structured_layered,
    grid_data_unstructured,
    grid_data_unstructured_layered,
)


def create_package_instances(is_structured: bool) -> List[Package]:
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
        imod.mf6.NodePropertyFlow(  # test package with layer-based array as input
            icelltype=grid_data_function(np.int_, 1, 5.0),
            k=xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": [1, 2, 3]}, ("layer",)),
            k22=3.0,
        ),
        imod.mf6.SpecificStorage(
            specific_storage=grid_data_function(np.float_, 1.0e-4, 5.0),
            specific_yield=grid_data_function(np.float_, 0.15, 5.0),
            convertible=0,
            transient=False,
        ),
        imod.mf6.SpecificStorage(
            specific_storage=xr.DataArray(
                [1.0e-3, 1.0e-4, 2.0e-4], {"layer": [1, 2, 3]}, ("layer",)
            ),
            specific_yield=xr.DataArray(
                [1.0e-3, 1.0e-4, 2.0e-4], {"layer": [1, 2, 3]}, ("layer",)
            ),
            convertible=0,
            transient=False,
        ),
        imod.mf6.SpecificStorage(  # test package with only scalar input
            specific_storage=0.3,
            specific_yield=0.4,
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
        imod.mf6.OutputControl(save_head="all", save_budget="all"),
        imod.mf6.Recharge(grid_data_function(np.float64, 0.002, 5.0).sel(layer=[1])),
        imod.mf6.InitialConditions(
            start=grid_data_function(np.float64, 0.002, 5.0), validate=True
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

    is_valid = (
        lambda pkg: len(pkg._validate(pkg._write_schemata, idomain=new_idomain)) == 0
    )
    assert all(is_valid(new_package) for new_package in new_packages)


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
        # TODO gitlab-398: package write validation crashes for VerticesDiscretization so we skip that one
        if not isinstance(new_package, imod.mf6.VerticesDiscretization):
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


def test_regrid(tmp_path: Path):
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


def test_regrid_not_supported_generic_message(basic_unstructured_dis):
    """
    This tests that regridding a package for which it is not implemented does not lead to a crash.
    It tests a package that should return the generic error message.
    """
    new_grid, _, _ = basic_unstructured_dis
    dispersivity = 1
    disp = xu.full_like(new_grid, dispersivity, dtype=float)

    disperion_package = imod.mf6.Dispersion(1e-4, disp, disp)
    with pytest.raises(
        NotImplementedError, match="Package Dispersion does not support regridding"
    ):
        disperion_package.regrid_like(new_grid)


def test_regrid_not_supported_custom_message(basic_unstructured_dis):
    """
    This tests that regridding a package for which it is not implemented does not lead to a crash.
    It tests a package that should return a custom error message for that package.
    """
    new_grid, _, _ = basic_unstructured_dis
    well = imod.mf6.Well([1.0], [-100.0], [2.0], [2.0], [0.01])
    with pytest.raises(
        NotImplementedError,
        match='The Well Package does not support regridding. \nRegridding can be achieved by passing the new discretization to the "to_mf6_pkg" function using the parameters "top", "bottom" and "active".\n',
    ):
        well.regrid_like(new_grid)

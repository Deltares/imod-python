from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import pytest
import xarray as xr
import xugrid as xu

import imod
from imod.mf6.package import Package
from imod.tests.fixtures.mf6_small_models_fixture import (
    grid_data_structured,
    grid_data_structured_layered,
    grid_data_unstructured,
    grid_data_unstructured_layered,
)
from imod.util.regrid import RegridderWeightsCache


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
            specific_storage=grid_data_function(np.float64, 1.0e-4, 5.0),
            specific_yield=grid_data_function(np.float64, 0.15, 5.0),
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
            storage_coefficient=grid_data_function(np.float64, 1.0e-4, 5.0),
            specific_yield=grid_data_function(np.float64, 0.15, 5.0),
            convertible=grid_data_function(np.int32, 0, 5.0),
            transient=True,
        ),
        imod.mf6.Drainage(
            elevation=grid_data_function(np.float64, 1.0e-4, 5.0),
            conductance=grid_data_function(np.float64, 1.0e-4, 5.0),
            print_input=True,
            print_flows=True,
            save_flows=True,
        ),
        imod.mf6.ConstantHead(
            grid_data_function(np.float64, 1.0e-4, 5.0),
            print_input=True,
            print_flows=True,
            save_flows=True,
        ),
        imod.mf6.GeneralHeadBoundary(
            head=grid_data_function(np.float64, 1.0e-4, 5.0),
            conductance=grid_data_function(np.float64, 1.0e-4, 5.0),
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
                bottom=grid_data_function_layered(np.float64, -1, 5.0),
                idomain=grid_data_function(np.int_, 1, 5.0),
            )
        )
    else:
        packages.append(
            imod.mf6.VerticesDiscretization(
                top=20.0,
                bottom=grid_data_function_layered(np.float64, -1, 5.0),
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

    regrid_cache = RegridderWeightsCache()
    new_packages = []
    for package in structured_grid_packages:
        new_packages.append(package.regrid_like(new_grid, regrid_cache))

    new_idomain = new_packages[0].dataset["icelltype"]

    def is_valid(pkg):
        return len(pkg._validate(pkg._write_schemata, idomain=new_idomain)) == 0

    assert all(is_valid(new_package) for new_package in new_packages)


def test_regrid_unstructured():
    """
    This test regrids a structured grid to another structured grid of the same size.
    Some of the arrays are entered as grids and others as scalars
    """
    unstructured_grid_packages = create_package_instances(is_structured=False)
    new_grid = grid_data_unstructured(np.float64, 12, 2.5)
    regrid_cache = RegridderWeightsCache()

    new_packages = []
    for package in unstructured_grid_packages:
        new_packages.append(package.regrid_like(new_grid, regrid_cache))

    new_idomain = new_packages[0].dataset["icelltype"]
    for new_package in new_packages:
        # TODO github-398: package write validation crashes for VerticesDiscretization so we skip that one
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
    regrid_cache = RegridderWeightsCache()
    with pytest.raises(
        ValueError,
        match="DataArray icelltype does not have both a dx and dy coordinates",
    ):
        _ = package.regrid_like(new_grid, regrid_cache)


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
    regrid_cache = RegridderWeightsCache()
    new_npf = npf.regrid_like(k, regrid_cache)

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


def test_regridding_can_skip_validation():
    """
    This tests if an invalid package can be regridded by turning off validation
    """

    # create a sto package with a negative storage coefficient. This would trigger a validation error if it were turned on.
    storage_coefficient = grid_data_structured(np.float64, -20.0, 0.25)
    specific_yield = grid_data_structured(np.float64, -30.0, 0.25)
    sto_package = imod.mf6.StorageCoefficient(
        storage_coefficient,
        specific_yield,
        transient=True,
        convertible=False,
        save_flows=True,
        validate=False,
    )

    # Regrid the package to a finer domain
    new_grid = grid_data_structured(np.float64, 1.0, 0.025)
    regrid_cache = RegridderWeightsCache()
    regridded_package = sto_package.regrid_like(new_grid, regrid_cache)

    # Check that write validation still fails for the regridded package
    new_bottom = deepcopy(new_grid)
    new_bottom.loc[{"layer": 1}] = 0.0
    new_bottom.loc[{"layer": 2}] = -1.0
    new_bottom.loc[{"layer": 3}] = -2.0
    pkg_errors = regridded_package._validate(
        schemata=imod.mf6.StorageCoefficient._write_schemata,
        idomain=new_grid,
        bottom=new_bottom,
    )

    # Check that the right errors were found
    assert len(pkg_errors) == 2
    assert (
        str(pkg_errors["storage_coefficient"])
        == "[ValidationError('not all values comply with criterion: >= 0.0')]"
    )
    assert (
        str(pkg_errors["specific_yield"])
        == "[ValidationError('not all values comply with criterion: >= 0.0')]"
    )


def test_regridding_layer_based_array():
    """
    This tests if dx/dy coordinates are correctly assigned when when regridding a package with layer-based input
    """
    nlay = 3
    storage_coefficient = xr.DataArray(
        [1e-4, 1e-4, 1e-4], {"layer": np.arange(1, nlay + 1)}, ("layer")
    )
    specific_yield = xr.DataArray(
        [1e-4, 1e-4, 1e-4], {"layer": np.arange(1, nlay + 1)}, ("layer")
    )
    sto_package = imod.mf6.StorageCoefficient(
        storage_coefficient,
        specific_yield=specific_yield,
        transient=True,
        convertible=False,
        save_flows=True,
        validate=False,
    )
    new_grid = grid_data_structured(np.float64, 1.0, 0.025)
    regrid_cache = RegridderWeightsCache()
    regridded_package = sto_package.regrid_like(new_grid, regrid_cache)

    assert (
        regridded_package.dataset.coords["dx"].values[()]
        == new_grid.coords["dx"].values[()]
    )
    assert (
        regridded_package.dataset.coords["dy"].values[()]
        == new_grid.coords["dy"].values[()]
    )

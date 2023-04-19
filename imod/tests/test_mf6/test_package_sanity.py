"""
These tests check whether the different imod.mf6 classes are consistent
in appearance and behavior. An important promise of the package classes
is that they can be dumped to netCDF. This means that all data must
be fully stored in the dataset (and not in separate attributes).

Hence:

* Tests for (private) attributes.
* Tests for whether rendering twice produces the same results.
* Tests whether writing and saving results in the same object state (as Python
  None is turned into numpy NaN).
"""
import inspect

import numpy as np
import pytest
import xarray as xr
import pandas as pd
import imod
from imod.mf6.pkgbase import AdvancedBoundaryCondition, BoundaryCondition, Package
from inspect import signature
import xugrid as xu

ALL_PACKAGES = [
    item
    for _, item in inspect.getmembers(imod.mf6, inspect.isclass)
    if issubclass(item, Package)
]
PACKAGES = [x for x in ALL_PACKAGES if not issubclass(x, BoundaryCondition)]
BOUNDARY_PACKAGES = [
    x
    for x in ALL_PACKAGES
    if issubclass(x, BoundaryCondition) and not issubclass(x, AdvancedBoundaryCondition)
]
ADV_BOUNDARY_PACKAGES = [
    x for x in ALL_PACKAGES if issubclass(x, AdvancedBoundaryCondition)
]

PACKAGE_ATTRIBUTES = {
    "_abc_impl",
    "_pkg_id",
    "_template",
    "_keyword_map",
    "_metadata_dict",
    "_init_schemata",
    "_write_schemata",
    "_grid_data",
    "dataset",
}
BOUNDARY_ATTRIBUTES = PACKAGE_ATTRIBUTES.union({"_period_data", "_auxiliary_data"})
ADV_BOUNDARY_ATTRIBUTES = BOUNDARY_ATTRIBUTES.union({"_package_data"})


def get_darray(dtype):
    """
    helper function for creating an xarray dataset of a given type
    """
    shape = nlay, nrow, ncol = 3, 9, 9
    dx = 10.0
    dy = -10.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    da = xr.DataArray(np.ones(shape, dtype=dtype), coords=coords, dims=dims)
    return da

def get_vertices_discretization():
    grid = imod.data.circle()

    nface = grid.n_face

    nlayer = 2

    idomain = xu.UgridDataArray(
        xr.DataArray(
            np.ones((nlayer, nface), dtype=np.int32),
            coords={"layer": [1, 2]},
            dims=["layer", grid.face_dimension],
        ),
        grid=grid,
    )    
    k = xu.full_like(idomain, 1.0, dtype=np.float64)    
    bottom = k * xr.DataArray([5.0, 0.0], dims=["layer"])
    return imod.mf6.VerticesDiscretization(
        top=10.0, bottom=bottom, idomain=idomain
    )

ALL_INSTANCES = [
    imod.mf6.adv.Advection("upstream"),
    imod.mf6.Buoyancy(
        reference_density=1000.0,
        reference_concentration=[4.0, 25.0],
        density_concentration_slope=[0.7, -0.375],
        modelname=["gwt-1", "gwt-2"],
        species=["salinity", "temperature"],
    ),
    imod.mf6.StructuredDiscretization(
        2.0, get_darray(np.float32), get_darray(np.int32)
    ),
    get_vertices_discretization(),
    imod.mf6.Dispersion(1e-4, 10.0, 10.0, 5.0, 2.0, 4.0, False, True),
    imod.mf6.InitialConditions(start=get_darray(np.float32)),
    imod.mf6.SolutionPresetSimple(modelnames=["gwf-1"]),
    imod.mf6.MobileStorageTransfer(0.35, 0.01, 0.02, 1300.0, 0.1),
    imod.mf6.NodePropertyFlow(get_darray(np.int32), 3.0, True, 32.0, 34.0, 7),
    imod.mf6.OutputControl(),
    imod.mf6.SpecificStorage(0.001, 0.1, True, get_darray(np.int32)),
    imod.mf6.StorageCoefficient(0.001, 0.1, True, get_darray(np.int32)),
    imod.mf6.TimeDiscretization(xr.DataArray(
        data=[0.001, 7.0, 365.0],
        coords={"time": pd.date_range("2000-01-01", "2000-01-03")},
        dims=["time"], ), 23, 1.02),
   
]


def check_attributes(pkg_class, allowed_attributes):
    class_attributes = set(
        [
            name
            for name, member in inspect.getmembers(pkg_class)
            if not name.startswith("__") and not callable(member)
        ]
    )

    assert "_pkg_id" in class_attributes
    # TODO: check for metadata/schema

    difference = class_attributes.difference(allowed_attributes)
    if len(difference) > 0:
        print(
            f"class {pkg_class.__name__} has a nonstandard class attributes: {difference}"
        )
        assert False


@pytest.mark.parametrize("pkg_class", PACKAGES)
def test_package_class_attributes(pkg_class):
    check_attributes(pkg_class, PACKAGE_ATTRIBUTES)


@pytest.mark.parametrize("pkg_class", BOUNDARY_PACKAGES)
def test_boundary_class_attributes(pkg_class):
    check_attributes(pkg_class, BOUNDARY_ATTRIBUTES)


@pytest.mark.parametrize("pkg_class", ADV_BOUNDARY_PACKAGES)
def test_adv_boundary_class_attributes(pkg_class):
    check_attributes(pkg_class, ADV_BOUNDARY_ATTRIBUTES)


@pytest.mark.parametrize("instance", ALL_INSTANCES)
def test_render_twice(instance, tmp_path):
    globaltimes = [np.datetime64("2000-01-01")]
    modeldir = tmp_path / "testdir"

    sig = signature(instance.render)
    if len(sig.parameters) == 0:
        text1 = instance.render()
        text2 = instance.render()
    elif len(sig.parameters) == 3:
        text1 = instance.render(modeldir, "test", False)
        text2 = instance.render(modeldir, "test", False)
    elif len(sig.parameters) ==4:
        text1 = instance.render(modeldir, "test", globaltimes, False)
        text2 = instance.render(modeldir, "test", globaltimes, False)
    else:
        assert False #unexpected nr of arguments
    assert text1 == text2


@pytest.mark.parametrize("instance", ALL_INSTANCES)
def test_save_and_load(instance, tmp_path):
    pkg_class = type(instance)
    path = tmp_path / f"{instance._pkg_id}.nc"
    instance.dataset.to_netcdf(path)
    back = pkg_class.from_file(path)
    assert instance.dataset.equals(back.dataset)

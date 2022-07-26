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

import imod
from imod.mf6.pkgbase import AdvancedBoundaryCondition, BoundaryCondition, Package

ALL_PACKAGES = [
    item
    for _, item in inspect.getmembers(imod.mf6)
    if inspect.isclass(item) and issubclass(item, Package)
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


ALL_INSTANCES = [
    imod.mf6.AdvectionUpstream(),
    imod.mf6.AdvectionCentral(),
    imod.mf6.AdvectionTVD(),
    imod.mf6.Buoyancy(
        reference_density=1000.0,
        reference_concentration=[4.0, 25.0],
        density_concentration_slope=[0.7, -0.375],
        modelname=["gwt-1", "gwt-2"],
        species=["salinity", "temperature"],
    ),
    imod.mf6.StructuredDiscretization(
        get_darray(np.float32), get_darray(np.float32), get_darray(np.int32)
    ),
    # TODO: VerticesDiscretization(),
    imod.mf6.Dispersion(1e-4, 10.0, 10.0, 5.0, 2.0, 4.0, False, True),
    imod.mf6.InitialConditions(start=get_darray(np.float32)),
    imod.mf6.SolutionPresetSimple(["gwf-1"]),
    imod.mf6.MobileStorageTransfer(0.35, 0.01, 0.02, 1300.0, 0.1),
    imod.mf6.NodePropertyFlow(get_darray(np.int32), 3.0, True, 32.0, 34.0, 7),
    # TODO imod.mf6.OutputControl(),
    imod.mf6.SpecificStorage(0.001, 0.1, True, get_darray(np.int32)),
    imod.mf6.StorageCoefficient(0.001, 0.1, True, get_darray(np.int32)),
    # TODO imod.mf6.TimeDiscretization(10.0, 23, 1.02),
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
    text1 = instance.render(modeldir, "test", globaltimes, False)
    text2 = instance.render(modeldir, "test", globaltimes, False)
    assert text1 == text2


@pytest.mark.parametrize("instance", ALL_INSTANCES)
def test_save_and_load(instance, tmp_path):
    pkg_class = type(instance)
    path = tmp_path / f"{instance._pkg_id}.nc"
    instance.dataset.to_netcdf(path)
    back = pkg_class.from_file(path)
    assert instance.dataset.equals(back.dataset)

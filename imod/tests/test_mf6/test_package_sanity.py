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
from inspect import signature

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xugrid as xu

import imod
from imod.mf6.pkgbase import AdvancedBoundaryCondition, BoundaryCondition, Package

from imod.tests.fixtures.package_sanity_structured import ALL_PACKAGE_INSTANCES

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


@pytest.mark.parametrize("instance", ALL_PACKAGE_INSTANCES)
def test_render_twice(instance, tmp_path):
    globaltimes = [np.datetime64("2000-01-01")]
    modeldir = tmp_path / "testdir"

    sig = inspect.signature(instance.render)
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


@pytest.mark.parametrize("instance", ALL_PACKAGE_INSTANCES)
def test_save_and_load(instance, tmp_path):
    pkg_class = type(instance)
    path = tmp_path / f"{instance._pkg_id}.nc"
    instance.dataset.to_netcdf(path)
    back = pkg_class.from_file(path)
    assert instance.dataset.equals(back.dataset)

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

import imod
from imod.mf6.boundary_condition import AdvancedBoundaryCondition, BoundaryCondition
from imod.mf6.package import Package
from imod.tests.fixtures.package_instance_creation import ALL_PACKAGE_INSTANCES

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

HIGH_LEVEL_PACKAGES = [
    imod.mf6.Well,
    imod.mf6.HorizontalFlowBarrierHydraulicCharacteristic,
    imod.mf6.HorizontalFlowBarrierMultiplier,
    imod.mf6.HorizontalFlowBarrierResistance,
    imod.mf6.LayeredWell,
]


def check_attributes(pkg_class):
    class_attributes = {
        name
        for name, member in inspect.getmembers(pkg_class)
        if not name.startswith("__") and not callable(member)
    }

    assert "_pkg_id" in class_attributes
    # TODO: check for metadata/schema


@pytest.mark.parametrize("pkg_class", PACKAGES)
def test_package_class_attributes(pkg_class):
    check_attributes(pkg_class)


@pytest.mark.parametrize("pkg_class", BOUNDARY_PACKAGES)
def test_boundary_class_attributes(pkg_class):
    check_attributes(pkg_class)


@pytest.mark.parametrize("pkg_class", ADV_BOUNDARY_PACKAGES)
def test_adv_boundary_class_attributes(pkg_class):
    check_attributes(pkg_class)


@pytest.mark.parametrize("instance", ALL_PACKAGE_INSTANCES)
def test_render_twice(instance, tmp_path):
    globaltimes = [np.datetime64("2000-01-01")]
    modeldir = tmp_path / "testdir"

    sig = inspect.signature(instance.render)
    if any(isinstance(instance, pack) for pack in HIGH_LEVEL_PACKAGES):
        with pytest.raises(NotImplementedError):
            instance.render(modeldir, "test", globaltimes, False)
        return
    elif len(sig.parameters) == 0:
        text1 = instance.render()
        text2 = instance.render()
    elif len(sig.parameters) == 3:
        text1 = instance.render(modeldir, "test", False)
        text2 = instance.render(modeldir, "test", False)
    elif len(sig.parameters) == 4:
        text1 = instance.render(modeldir, "test", globaltimes, False)
        text2 = instance.render(modeldir, "test", globaltimes, False)
    else:
        assert False  # unexpected nr of arguments
    assert text1 == text2


@pytest.mark.parametrize("instance", ALL_PACKAGE_INSTANCES)
def test_save_and_load(instance, tmp_path):
    pkg_class = type(instance)
    path = tmp_path / f"{instance._pkg_id}.nc"
    instance.to_netcdf(path)
    back = pkg_class.from_file(path)
    assert instance.dataset.equals(back.dataset)


@pytest.mark.parametrize("instance", ALL_PACKAGE_INSTANCES)
def test_repr(instance):
    assert isinstance(instance.__repr__(), str)
    assert isinstance(instance._repr_html_(), str)


@pytest.mark.parametrize("instance", ALL_PACKAGE_INSTANCES)
def test_from_dataset(instance):
    pkg_class = type(instance)
    ds = instance.dataset
    new_instance = pkg_class._from_dataset(ds)
    assert isinstance(new_instance, pkg_class)
    assert instance.dataset.equals(new_instance.dataset)

import pytest

from imod.tests.fixtures.msw_package_instance_creation import MSW_PACKAGE_INSTANCES

NAMES = [type(instance).__name__ for instance in MSW_PACKAGE_INSTANCES]


@pytest.mark.parametrize("instance", MSW_PACKAGE_INSTANCES, ids=NAMES)
@pytest.mark.parametrize("engine", ["netcdf4", "zarr", "zarr.zip"])
def test_msw_save_and_load(instance, engine, tmp_path):
    pkg_class = type(instance)
    path = instance.to_file(tmp_path, instance._file_name, engine=engine)
    back = pkg_class.from_file(path)
    assert instance.dataset.equals(back.dataset)

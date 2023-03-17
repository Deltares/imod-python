import pytest
import xarray as xr
from imod.mf6.lak import Lake
import pathlib
import imod.tests.fixtures.mf6_lake_package_fixture as mf_lake

@pytest.mark.usefixtures("naardermeer", "ijsselmeer")

def test_mf6_lake_table_write(naardermeer, ijsselmeer, tmp_path):
    directory = pathlib.Path("mymodel")
    lake_package = Lake.from_lakes_and_outlets([naardermeer(has_lake_table=True), ijsselmeer(has_lake_table=True)])
    actual = mf_lake.write_and_read(lake_package, tmp_path, "lake-test")
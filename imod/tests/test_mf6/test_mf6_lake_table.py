import pytest
import xarray as xr
from imod.mf6.lak import Lake

@pytest.mark.usefixtures("naardermeer", "ijsselmeer")

def test_mf6_lake_table_write(naardermeer, ijsselmeer):

    actual = Lake.from_lakes_and_outlets([naardermeer(has_lake_table=True), ijsselmeer(has_lake_table=True)])

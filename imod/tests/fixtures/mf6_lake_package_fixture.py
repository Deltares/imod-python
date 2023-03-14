
import xarray as xr
import pytest
import numpy as np
    
from imod.mf6.lak import (
    Lake,
    LakeData,
    OutletManning,

)

@pytest.fixture(scope="function")
def naardermeer(basic_dis):
    idomain, _, _ = basic_dis
    is_lake = xr.full_like(idomain, False, dtype=bool)
    is_lake[0, 1, 1] = True
    is_lake[0, 1, 2] = True
    is_lake[0, 2, 2] = True
    return create_lake_data(is_lake, starting_stage=11.0, name="Naardermeer")


@pytest.fixture(scope="function")
def ijsselmeer(basic_dis):
    idomain, _, _ = basic_dis
    is_lake = xr.full_like(idomain, False, dtype=bool)
    is_lake[0, 4, 4] = True
    is_lake[0, 4, 5] = True
    is_lake[0, 5, 5] = True
    return create_lake_data(
        is_lake,
        starting_stage=15.0,
        name="IJsselmeer",
    )


@pytest.fixture(scope="function")
def lake_package(naardermeer, ijsselmeer):
    outlet1 = OutletManning("Naardermeer", "IJsselmeer", 23.0, 24.0, 25.0, 26.0)
    outlet2 = OutletManning("IJsselmeer", "Naardermeer", 27.0, 28.0, 29.0, 30.0)
    return Lake.from_lakes_and_outlets([naardermeer, ijsselmeer], [outlet1, outlet2])


def create_lake_data(
    is_lake,
    starting_stage,
    name,
    status=None,
    stage=None,
    rainfall=None,
    evaporation=None,
    runoff=None,
    inflow=None,
    withdrawal=None,
    auxiliary=None,
):
    HORIZONTAL = 0
    connection_type = xr.full_like(is_lake, HORIZONTAL, dtype=np.floating).where(
        is_lake
    )
    bed_leak = xr.full_like(is_lake, 0.2, dtype=np.floating).where(is_lake)
    top_elevation = xr.full_like(is_lake, 0.3, dtype=np.floating).where(is_lake)
    bot_elevation = xr.full_like(is_lake, 0.4, dtype=np.floating).where(is_lake)
    connection_length = xr.full_like(is_lake, 0.5, dtype=np.floating).where(is_lake)
    connection_width = xr.full_like(is_lake, 0.6, dtype=np.floating).where(is_lake)
    return LakeData(
        starting_stage=starting_stage,
        boundname=name,
        connection_type=connection_type,
        bed_leak=bed_leak,
        top_elevation=top_elevation,
        bot_elevation=bot_elevation,
        connection_length=connection_length,
        connection_width=connection_width,
        status=status,
        stage=stage,
        rainfall=rainfall,
        evaporation=evaporation,
        runoff=runoff,
        inflow=inflow,
        withdrawal=withdrawal,
        auxiliary=auxiliary,
    )



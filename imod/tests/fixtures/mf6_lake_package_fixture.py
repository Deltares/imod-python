import numpy as np
import pytest
import xarray as xr
import xugrid as xu

from imod.mf6.lak import Lake, LakeData, OutletManning
from imod.mf6.write_context import WriteContext


def create_lake_table(
    number_rows, starting_stage, starting_sarea, starting_volume, starting_barea=None
):
    """
    creates a lake table. The mandatory columns are stage, area and volume. Optionally a barea column
    is present. The number of rows of the table is an input argument. The first row of the table will
    have the starting values specified in the input arguments. Each next row will have the values of the
    previous row, plus one.
    """
    rownumbers = np.arange(number_rows)

    lake_table = xr.Dataset(coords={"row": rownumbers})
    stage_data = (rownumbers + starting_stage).astype(np.float64)
    volume_data = (rownumbers + starting_volume).astype(np.float64)
    sarea_data = (rownumbers + starting_sarea).astype(np.float64)

    lake_table["stage"] = xr.DataArray(
        coords={"row": rownumbers},
        data=stage_data,
    )
    lake_table["volume"] = xr.DataArray(coords={"row": rownumbers}, data=volume_data)

    lake_table["sarea"] = xr.DataArray(coords={"row": rownumbers}, data=sarea_data)

    if starting_barea is not None:
        barea_data = (rownumbers + starting_barea).astype(np.float64)
        lake_table["barea"] = xr.DataArray(coords={"row": rownumbers}, data=barea_data)

    lake_table_array = lake_table.to_array()
    lake_table_array = lake_table_array.rename({"variable": "column"})
    return lake_table_array


@pytest.fixture(scope="function")
def naardermeer(basic_dis):
    def _naardermeer(has_lake_table=False):
        idomain, _, _ = basic_dis
        is_lake = xr.full_like(idomain, False, dtype=bool)
        is_lake[0, 1, 1] = True
        is_lake[0, 1, 2] = True
        is_lake[0, 2, 2] = True
        lake_table = None
        if has_lake_table:
            lake_table = create_lake_table(3, 10, 20, 30)
        return create_lake_data_structured(
            is_lake, starting_stage=11.0, name="Naardermeer", lake_table=lake_table
        )

    return _naardermeer


@pytest.fixture(scope="function")
def ijsselmeer(basic_dis):
    def _ijsselmeer(has_lake_table=False):
        idomain, _, _ = basic_dis
        is_lake = xr.full_like(idomain, False, dtype=bool)
        is_lake[0, 4, 4] = True
        is_lake[0, 4, 5] = True
        is_lake[0, 5, 5] = True
        lake_table = None
        if has_lake_table:
            lake_table = create_lake_table(6, 8, 9, 10, 11)
        return create_lake_data_structured(
            is_lake, starting_stage=15.0, name="IJsselmeer", lake_table=lake_table
        )

    return _ijsselmeer


@pytest.fixture(scope="function")
def lake_package(naardermeer, ijsselmeer):
    outlet1 = OutletManning("Naardermeer", "IJsselmeer", 23.0, 24.0, 25.0, 26.0)
    outlet2 = OutletManning("IJsselmeer", "Naardermeer", 27.0, 28.0, 29.0, 30.0)
    return Lake.from_lakes_and_outlets(
        [naardermeer(), ijsselmeer()], [outlet1, outlet2]
    )


@pytest.fixture(scope="function")
def lake_table():
    return create_lake_table(5, 2.1, 3.1415, 100)


def create_lake_data_structured(
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
    lake_table=None,
):
    HORIZONTAL = 0
    connection_type = xr.full_like(is_lake, HORIZONTAL, dtype=np.float64).where(is_lake)
    bed_leak = xr.full_like(is_lake, 0.2, dtype=np.float64).where(is_lake)
    top_elevation = xr.full_like(is_lake, 0.3, dtype=np.float64).where(is_lake)
    bot_elevation = xr.full_like(is_lake, 0.4, dtype=np.float64).where(is_lake)
    connection_length = xr.full_like(is_lake, 0.5, dtype=np.float64).where(is_lake)
    connection_width = xr.full_like(is_lake, 0.6, dtype=np.float64).where(is_lake)
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
        lake_table=lake_table,
    )


def write_and_read(package, path, filename, globaltimes=None) -> str:
    write_context = WriteContext(write_directory=path)
    package.write(filename, globaltimes, write_context)
    with open(path / f"{filename}.lak") as f:
        actual = f.read()
    return actual


def create_lake_data_unstructured(
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
    lake_table=None,
):
    HORIZONTAL = 0
    connection_type = xu.full_like(is_lake, HORIZONTAL, dtype=np.float64).where(is_lake)
    bed_leak = xu.full_like(is_lake, 0.2, dtype=np.float64).where(is_lake)
    top_elevation = xu.full_like(is_lake, 0.3, dtype=np.float64).where(is_lake)
    bot_elevation = xu.full_like(is_lake, 0.4, dtype=np.float64).where(is_lake)
    connection_length = xu.full_like(is_lake, 0.5, dtype=np.float64).where(is_lake)
    connection_width = xu.full_like(is_lake, 0.6, dtype=np.float64).where(is_lake)
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
        lake_table=lake_table,
    )

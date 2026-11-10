import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.logging import LoggerType, configure
from imod.logging.loglevel import LogLevel
from imod.mf6.dis import StructuredDiscretization
from imod.msw import GridData
from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage


def _grid(value: float, source: str | None = None) -> xr.DataArray:
    da = xr.DataArray(
        np.full((3, 3), value, dtype=float),
        dims=("y", "x"),
        coords={"y": [3.0, 2.0, 1.0], "x": [1.0, 2.0, 3.0], "dx": 1.0, "dy": -1.0},
    )
    if source is not None:
        da.attrs["source_paths"] = [source]
    return da


@pytest.fixture
def imod5_data_and_dis():
    cap_data = {
        "boundary": _grid(1.0),
        "wetted_area": _grid(0.5, "c:/data/wetted_area.idf"),
        "urban_area": _grid(0.1, "c:/data/urban_area.idf"),
        "landuse": _grid(1.0, "c:/data/landuse.idf"),
        "rootzone_thickness": _grid(100.0, "c:/data/rootzone.idf"),
        "surface_elevation": _grid(0.0, "c:/data/surface_elevation.idf"),
        "soil_physical_unit": _grid(1.0, "c:/data/soil.idf"),
    }
    bnd_data = {"ibound": _grid(1.0)}
    layer = xr.DataArray([1, 1], coords={"layer": [1, 2]}, dims=("layer",))
    idomain = layer * xr.ones_like(cap_data["boundary"], dtype=int)
    top = _grid(0.0)
    dis = StructuredDiscretization(top, top - 0.1, idomain, validate=False)
    return {"cap": cap_data, "bnd": bnd_data}, dis


def test_from_imod5_data__valid(imod5_data_and_dis):
    imod5_data, dis = imod5_data_and_dis
    griddata, msw_active = GridData.from_imod5_data(imod5_data, target_dis=dis)
    assert bool(msw_active.all.all())


@pytest.mark.parametrize(
    "variable, filename",
    [
        ("landuse", "landuse.idf"),
        ("soil_physical_unit", "soil.idf"),
        ("rootzone_thickness", "rootzone.idf"),
        ("surface_elevation", "surface_elevation.idf"),
    ],
)
def test_from_imod5_data__nodata_in_active_cell(imod5_data_and_dis, variable, filename):
    imod5_data, dis = imod5_data_and_dis
    imod5_data["cap"][variable][0, 0] = np.nan

    with pytest.raises(ValueError, match="Found NoData") as excinfo:
        GridData.from_imod5_data(imod5_data, target_dis=dis)
    message = str(excinfo.value)
    assert variable in message
    assert "1 cell(s)" in message
    assert "(1.0, 3.0)" in message
    assert filename in message


def test_from_imod5_data__nodata_in_inactive_cell(imod5_data_and_dis):
    imod5_data, dis = imod5_data_and_dis
    imod5_data["cap"]["boundary"][0, 0] = 0.0
    for var in ["landuse", "soil_physical_unit", "surface_elevation"]:
        imod5_data["cap"][var][0, 0] = np.nan

    griddata, msw_active = GridData.from_imod5_data(imod5_data, target_dis=dis)
    assert not bool(msw_active.all[0, 0])


def test_from_imod5_data__nodata_area_warns(imod5_data_and_dis, capsys):
    imod5_data, dis = imod5_data_and_dis
    imod5_data["cap"]["urban_area"][0, 0] = np.nan

    configure(
        LoggerType.LOGURU,
        log_level=LogLevel.WARNING,
        add_default_file_handler=False,
        add_default_stream_handler=True,
    )
    griddata, msw_active = GridData.from_imod5_data(imod5_data, target_dis=dis)
    captured = capsys.readouterr()
    log = captured.out + captured.err

    assert not bool(msw_active.all[0, 0])
    assert bool(msw_active.all[1, 1])
    assert "urban_area" in log
    assert "urban_area.idf" in log


class _DummyPackage(MetaSwapPackage):
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 99999999, int),
        "value": VariableMetaData(8, 0.0, 10.0, float),
        "name": VariableMetaData(8, None, None, str),
    }

    def __init__(self) -> None:
        super().__init__()


def test_check_range__nan():
    dataframe = pd.DataFrame(
        {"svat": [1, 2], "value": [1.0, np.nan], "name": ["", ""]},
    )
    pkg = _DummyPackage()
    with pytest.raises(ValueError, match="value: found 1 NaN"):
        pkg._check_range(dataframe)


def test_check_range__valid():
    dataframe = pd.DataFrame({"svat": [1, 2], "value": [1.0, 2.0], "name": ["", ""]})
    _DummyPackage()._check_range(dataframe)

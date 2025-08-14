from copy import deepcopy
from datetime import datetime
from filelock import FileLock
from typing import Any
from zipfile import ZipFile

import pytest
import xarray as xr

import imod
from imod.data.sample_data import REGISTRY
from imod.formats.prj.prj import open_projectfile_data


@pytest.fixture(scope="module")
def imod5_dataset():
    tmp_path = imod.util.temporary_directory()
    data = imod.data.imod5_projectfile_data(tmp_path)

    pd = data[1]
    data = data[0]

    _load_imod5_data_in_memory(data)

    # Fix data for ibound  as it contains floating values like 0.34, 0.25 etc.
    ibound = data["bnd"]["ibound"]
    ibound = ibound.where(ibound <= 0, 2)
    data["bnd"]["ibound"] = ibound
    return data, pd


def _load_imod5_data_in_memory(imod5_data):
    """For debugging purposes, load everything in memory"""
    for pkg in imod5_data.values():
        for vardata in pkg.values():
            if isinstance(vardata, xr.DataArray):
                vardata.load()


@pytest.fixture(scope="module")
def imod5_dataset_periods() -> tuple[dict[str, Any], dict[str, list[datetime]]]:
    tmp_path = imod.util.temporary_directory()
    fname_model = REGISTRY.fetch("iMOD5_model.zip")

    lock = FileLock(REGISTRY.path / "iMOD5_model.zip.lock")
    with lock:
        with ZipFile(fname_model) as archive:
            archive.extractall(tmp_path)

        with open(tmp_path / "iMOD5_model_pooch" / "iMOD5_model.prj", "w") as f:
            f.write(period_prj)

        data = open_projectfile_data(tmp_path / "iMOD5_model_pooch" / "iMOD5_model.prj")

    grid_data = data[0]
    period_data = data[1]

    _load_imod5_data_in_memory(grid_data)

    # Fix data for ibound  as it contains floating values like 0.34, 0.25 etc.
    ibound = grid_data["bnd"]["ibound"]
    ibound = ibound.where(ibound <= 0, 1)
    grid_data["bnd"]["ibound"] = ibound
    return grid_data, period_data


@pytest.fixture(scope="module")
def imod5_dataset_transient(imod5_dataset):
    grid_data, period_data = imod5_dataset
    grid_data = deepcopy(grid_data)

    time_factors = [0.5, 0.75, 1.0, 1.25]
    time = [
        datetime(2000, 1, 1),
        datetime(2001, 1, 1),
        datetime(2002, 1, 1),
        datetime(2003, 1, 1),
    ]
    time_da = xr.DataArray(time_factors, dims=["time"], coords={"time": time})

    for pkgname in grid_data.keys():
        if pkgname[:3] in ["riv", "drn", "ghb"]:
            for varname, vardata in grid_data[pkgname].items():
                if isinstance(vardata, xr.DataArray):
                    grid_data[pkgname][varname] = time_da * vardata

    return grid_data, period_data


period_prj = r"""\
0001,(BND),1, Boundary Condition
001,37
1,2,1,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L1.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,2,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L2.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,3,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L3.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,4,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L4.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,5,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L5.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,6,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L6.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,7,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L7.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,8,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L8.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,9,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L9.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,10,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L10.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,11,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L11.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,12,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L12.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,13,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L13.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,14,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L14.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,15,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L15.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,16,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L16.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,17,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L17.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,18,1.0,0.0,-999.99, '.\Database\BND\VERSION_1\IBOUND_L18.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,19,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L19.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,20,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L20.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,21,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L21.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,22,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L22.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,23,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L23.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,24,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L24.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,25,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L25.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,26,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L26.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,27,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L27.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,28,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L28.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,29,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L29.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,30,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L30.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,31,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L31.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,32,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L32.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,33,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L33.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,34,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L34.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,35,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L35.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,36,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L36.IDF' >>> (BND) Boundary Settings (IDF) <<<
1,2,37,1.0,0.0,-999.99,'.\Database\BND\VERSION_1\IBOUND_L37.IDF' >>> (BND) Boundary Settings (IDF) <<<


0001,(TOP),1, Top Elevation
001,37
1,2,1,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L1.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,2,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L2.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,3,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L3.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,4,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L4.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,5,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L5.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,6,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L6.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,7,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L7.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,8,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L8.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,9,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L9.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,10,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L10.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,11,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L11.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,12,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L12.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,13,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L13.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,14,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L14.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,15,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L15.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,16,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L16.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,17,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L17.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,18,1.0,0.0,-999.99, '.\Database\TOP\VERSION_1\TOP_L18.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,19,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L19.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,20,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L20.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,21,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L21.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,22,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L22.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,23,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L23.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,24,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L24.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,25,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L25.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,26,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L26.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,27,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L27.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,28,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L28.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,29,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L29.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,30,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L30.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,31,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L31.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,32,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L32.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,33,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L33.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,34,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L34.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,35,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L35.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,36,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L36.IDF' >>> (TOP) Top of Modellayer (IDF) <<<
1,2,37,1.0,0.0,-999.99,'.\Database\TOP\VERSION_1\TOP_L37.IDF' >>> (TOP) Top of Modellayer (IDF) <<<


0001,(BOT),1, Bottom Elevation
001,37
1,2,1,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L1.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,2,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L2.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,3,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L3.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,4,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L4.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,5,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L5.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,6,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L6.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,7,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L7.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,8,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L8.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,9,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L9.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,10,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L10.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,11,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L11.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,12,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L12.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,13,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L13.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,14,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L14.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,15,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L15.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,16,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L16.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,17,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L17.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,18,1.0,0.0,-999.99, '.\Database\BOT\VERSION_1\BOT_L18.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,19,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L19.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,20,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L20.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,21,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L21.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,22,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L22.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,23,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L23.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,24,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L24.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,25,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L25.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,26,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L26.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,27,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L27.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,28,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L28.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,29,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L29.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,30,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L30.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,31,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L31.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,32,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L32.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,33,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L33.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,34,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L34.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,35,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L35.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,36,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L36.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<
1,2,37,1.0,0.0,-999.99,'.\Database\BOT\VERSION_1\BOT_L37.IDF' >>> (BOT) Bottom of Modellayer (IDF) <<<


0001,(KHV),1, Horizontal Permeability
001,37
1,1,1,1.0,0.0,1.0   >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,2,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L2.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,3,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L3.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,4,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L4.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,5,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L5.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,6,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L6.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,7,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L7.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,8,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L8.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,9,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L9.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,10,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L10.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,11,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L11.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,12,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L12.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,13,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L13.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,14,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L14.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,15,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L15.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,16,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L16.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,17,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L17.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,18,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L18.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,19,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L19.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,20,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L20.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,21,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L21.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,22,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L22.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,23,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L23.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,24,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L24.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,25,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L25.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,26,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L26.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,27,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L27.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,28,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L28.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,29,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L29.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,30,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L30.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,1,31,1.0,0.0,1.0  >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,32,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L32.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,33,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L33.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,34,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L34.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,35,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L35.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,36,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L36.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,37,1.0,0.0,-999.99,'.\Database\KHV\VERSION_1\IPEST_KHV_L37.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<


0001,(KVA),1, Vertical Anisotropy
001,37
1,1,1,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,2,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,3,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,4,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,5,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,6,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,7,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,8,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,9,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,10,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,11,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,12,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,13,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,14,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,15,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,16,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,17,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,18,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,19,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,20,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,21,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,22,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,23,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,24,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,25,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,26,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,27,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,28,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,29,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,30,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,31,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,32,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,33,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,34,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,35,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,36,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,37,1.0,0.0,0.3,'' >>> (KVA) Vertical Anisotropy (IDF) <<<

0001,(SHD),1, Starting Heads
001,37
1,2,1,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L1.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,2,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L2.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,3,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L3.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,4,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L4.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,5,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L5.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,6,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L6.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,7,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L7.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,8,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L8.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,9,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L9.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,10,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L10.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,11,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L11.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,12,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L12.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,13,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L13.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,14,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L14.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,15,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L15.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,16,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L16.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,17,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L17.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,18,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L18.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,19,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L19.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,20,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L20.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,21,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L21.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,22,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L22.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,23,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L23.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,1,24,1.0,0.0,20.5, >>> (SHD) Starting Heads (IDF) <<<
1,1,25,1.0,0.0,20.5, >>> (SHD) Starting Heads (IDF) <<<
1,1,26,1.0,0.0,20.5, >>> (SHD) Starting Heads (IDF) <<<
1,1,27,1.0,0.0,20.5, >>> (SHD) Starting Heads (IDF) <<<
1,1,28,1.0,0.0,20.5, >>> (SHD) Starting Heads (IDF) <<<
1,1,29,1.0,0.0,20.5, >>> (SHD) Starting Heads (IDF) <<<
1,1,30,1.0,0.0,20.5, >>> (SHD) Starting Heads (IDF) <<<
1,1,31,1.0,0.0,20.5, >>> (SHD) Starting Heads (IDF) <<<
1,1,32,1.0,0.0,20.5, >>> (SHD) Starting Heads (IDF) <<<
1,1,33,1.0,0.0,20.5, >>> (SHD) Starting Heads (IDF) <<<
1,1,34,1.0,0.0,20.5, >>> (SHD) Starting Heads (IDF) <<<
1,1,35,1.0,0.0,20.5, >>> (SHD) Starting Heads (IDF) <<<
1,1,36,1.0,0.0,20.5, >>> (SHD) Starting Heads (IDF) <<<
1,1,37,1.0,0.0,20.5, >>> (SHD) Starting Heads (IDF) <<<


0001,(ANI),0, Anisotropy
002,04
1,2,2,1.0,0.0,-999.99,'.\Database\ANI\VERSION_1\ANI_FACTOR.IDF' >>> (FCT) Factor (IDF) <<<
1,2,4,1.0,0.0,-999.99,'.\Database\ANI\VERSION_1\ANI_FACTOR.IDF' >>> (FCT) Factor (IDF) <<<
1,2,6,1.0,0.0,-999.99,'.\Database\ANI\VERSION_1\ANI_FACTOR.IDF' >>> (FCT) Factor (IDF) <<<
1,2,8,1.0,0.0,-999.99,'.\Database\ANI\VERSION_1\ANI_FACTOR.IDF' >>> (FCT) Factor (IDF) <<<
1,2,2,1.0,0.0,-999.99,'.\Database\ANI\VERSION_1\ANI_HOEK.IDF' >>> (FCT) Factor (IDF) <<<
1,2,4,1.0,0.0,-999.99,'.\Database\ANI\VERSION_1\ANI_HOEK.IDF' >>> (FCT) Factor (IDF) <<<
1,2,6,1.0,0.0,-999.99,'.\Database\ANI\VERSION_1\ANI_HOEK.IDF' >>> (FCT) Factor (IDF) <<<
1,2,8,1.0,0.0,-999.99,'.\Database\ANI\VERSION_1\ANI_HOEK.IDF' >>> (ANG) Angle (IDF) <<<


0001,(STO),1, Storage
001,37
1,1,1,1.0,0.0,0.15,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,2,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,3,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,4,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,5,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,6,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,7,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,8,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,9,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,10,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,11,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,12,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,13,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,14,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,15,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,16,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,17,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,18,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,19,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,20,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,21,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,22,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,23,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,24,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,25,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,26,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,27,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,28,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,29,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,30,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,31,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,32,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,33,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,34,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,35,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,36,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<
1,1,37,1.0,0.0,1.0e-5,'' >>> (KVA) Vertical Anisotropy (IDF) <<<

0001,(HFB),1, Horizontal Flow Barrier
001,26
 1,2, 003,   1.000000    ,   10.00000    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_BX.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 005,   1.000000    ,   1000.000    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_SY.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 007,   1.000000    ,   1000.000    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_SY.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 009,   1.000000    ,   1000.000    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_SY.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 011,   1.000000    ,   1000.000    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_SY.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 013,   1.000000    ,   1000.000    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_SY.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 015,   1.000000    ,   101000.0    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 021,   1.000000    ,   400.0000    ,  -999.9900    ,     '.\Database\HFB\VERSION_1\IBV2_BREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 023,   1.000000    ,   101000.0    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 023,   1.000000    ,   400.0000    ,  -999.9900    ,     '.\Database\HFB\VERSION_1\IBV2_BREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 025,   1.000000    ,   101000.0    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 025,   1.000000    ,   400.0000    ,  -999.9900    ,     '.\Database\HFB\VERSION_1\IBV2_BREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 027,   1.000000    ,   101000.0    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 027,   1.000000    ,   400.0000    ,  -999.9900    ,     '.\Database\HFB\VERSION_1\IBV2_BREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 029,   1.000000    ,   101000.0    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 031,   1.000000    ,   400.0000    ,  -999.9900    ,     '.\Database\HFB\VERSION_1\IBV2_BREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 031,   1.000000    ,   101000.0    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 031,   1.000000    ,   400.0000    ,  -999.9900    ,     '.\Database\HFB\VERSION_1\IBV2_BREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 033,   1.000000    ,   101000.0    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 033,   1.000000    ,   101000.0    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 033,   1.000000    ,   400.0000    ,  -999.9900    ,     '.\Database\HFB\VERSION_1\IBV2_BREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 033,   1.000000    ,   400.0000    ,  -999.9900    ,     '.\Database\HFB\VERSION_1\IBV2_BREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 035,   1.000000    ,   101000.0    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 035,   1.000000    ,   400.0000    ,  -999.9900    ,     '.\Database\HFB\VERSION_1\IBV2_BREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 037,   1.000000    ,   101000.0    ,  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
 1,2, 037,   1.000000    ,   400.0000    ,  -999.9900    ,     '.\Database\HFB\VERSION_1\IBV2_BREUKEN_BR.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<

0002,(RIV),1, Rivers
winter
004,003
1,2,0,1.0,0.0,-999.99,'.\Database\RIV\VERSION_1\RIVER_PRIMAIR\IPEST_RIVER_PRIMAIR_COND_GEMIDDELD.IDF' >>> (CON) Conductance (IDF) <<<
1,2,0,1.0,0.0,-999.99,                         '.\Database\RIV\VERSION_1\MAAS\IPEST_COND19912011.IDF' >>> (CON) Conductance (IDF) <<<
1,2,0,1.0,0.0,-999.99,                              '.\Database\RIV\VERSION_1\BELGIE\COND_CAT012.IDF' >>> (CON) Conductance (IDF) <<<
1,2,0,1.0,0.0,-999.99,     '.\Database\RIV\VERSION_1\RIVER_PRIMAIR\RIVER_PRIMAIR_STAGE_GEMIDDELD.IDF' >>> (RST) River Stage (IDF) <<<
1,2,0,1.0,0.0,-999.99,                              '.\Database\RIV\VERSION_1\MAAS\STAGE19912011.IDF' >>> (RST) River Stage (IDF) <<<
1,2,0,1.0,0.0,-999.99,                             '.\Database\RIV\VERSION_1\BELGIE\STAGE_CAT012.IDF' >>> (RST) River Stage (IDF) <<<
1,2,0,1.0,0.0,-999.99,    '.\Database\RIV\VERSION_1\RIVER_PRIMAIR\RIVER_PRIMAIR_BOTTOM_GEMIDDELD.IDF' >>> (RBT) River Bottom (IDF) <<<
1,2,0,1.0,0.0,-999.99,                             '.\Database\RIV\VERSION_1\MAAS\BOTTOM19912011.IDF' >>> (RBT) River Bottom (IDF) <<<
1,2,0,1.0,0.0,-999.99,                               '.\Database\RIV\VERSION_1\BELGIE\BOT_CAT012.IDF' >>> (RBT) River Bottom (IDF) <<<
1,2,0,1.0,0.0,-999.99,    '.\Database\RIV\VERSION_1\RIVER_PRIMAIR\RIVER_PRIMAIR_INFFCT_GEMIDDELD.IDF' >>> (RIF) Infiltration Factor (IDF) <<<
1,2,0,1.0,0.0,-999.99,                             '.\Database\RIV\VERSION_1\MAAS\INFFCT19912011.IDF' >>> (RIF) Infiltration Factor (IDF) <<<
1,1,0,1.0,0.0,1.0,                                   '' >>> (RIF) Infiltration Factor (IDF) <<<
summer
004,003
1,2,0,1.0,0.0,-999.99,'.\Database\RIV\VERSION_1\RIVER_PRIMAIR\IPEST_RIVER_PRIMAIR_COND_GEMIDDELD.IDF' >>> (CON) Conductance (IDF) <<<
1,2,0,1.0,0.0,-999.99,                         '.\Database\RIV\VERSION_1\MAAS\IPEST_COND19912011.IDF' >>> (CON) Conductance (IDF) <<<
1,2,0,1.0,0.0,-999.99,                              '.\Database\RIV\VERSION_1\BELGIE\COND_CAT012.IDF' >>> (CON) Conductance (IDF) <<<
1,2,0,1.0,0.0,-999.99,     '.\Database\RIV\VERSION_1\RIVER_PRIMAIR\RIVER_PRIMAIR_STAGE_GEMIDDELD.IDF' >>> (RST) River Stage (IDF) <<<
1,2,0,1.0,0.0,-999.99,                              '.\Database\RIV\VERSION_1\MAAS\STAGE19912011.IDF' >>> (RST) River Stage (IDF) <<<
1,2,0,1.0,0.0,-999.99,                             '.\Database\RIV\VERSION_1\BELGIE\STAGE_CAT012.IDF' >>> (RST) River Stage (IDF) <<<
1,2,0,1.0,0.0,-999.99,    '.\Database\RIV\VERSION_1\RIVER_PRIMAIR\RIVER_PRIMAIR_BOTTOM_GEMIDDELD.IDF' >>> (RBT) River Bottom (IDF) <<<
1,2,0,1.0,0.0,-999.99,                             '.\Database\RIV\VERSION_1\MAAS\BOTTOM19912011.IDF' >>> (RBT) River Bottom (IDF) <<<
1,2,0,1.0,0.0,-999.99,                               '.\Database\RIV\VERSION_1\BELGIE\BOT_CAT012.IDF' >>> (RBT) River Bottom (IDF) <<<
1,2,0,1.0,0.0,-999.99,    '.\Database\RIV\VERSION_1\RIVER_PRIMAIR\RIVER_PRIMAIR_INFFCT_GEMIDDELD.IDF' >>> (RIF) Infiltration Factor (IDF) <<<
1,2,0,1.0,0.0,-999.99,                             '.\Database\RIV\VERSION_1\MAAS\INFFCT19912011.IDF' >>> (RIF) Infiltration Factor (IDF) <<<
1,1,0,1.0,0.0,1.0,                                   '' >>> (RIF) Infiltration Factor (IDF) <<<


0001,(RCH),1, Recharge
STEADY-STATE
001,001
1,2,1,1.0,0.0,-999.99,'.\Database\RCH\VERSION_1\GWAANVULLING_MEAN_19940114-20111231.IDF' >>> (RCH) Recharge Rate (IDF) <<<

0001,(WEL),1, Wells
STEADY-STATE
001,003
1,2,5,1.0,0.0,-999.99,                                       '.\Database\WEL\VERSION_1\WELLS_L3.IPF' >>> (WRA) Well Rate (IPF) <<<
1,2,7,1.0,0.0,-999.99,                                       '.\Database\WEL\VERSION_1\WELLS_L4.IPF' >>> (WRA) Well Rate (IPF) <<<
1,2,9,1.0,0.0,-999.99,                                       '.\Database\WEL\VERSION_1\WELLS_L5.IPF' >>> (WRA) Well Rate (IPF) <<<

0002,(DRN),1, Drainage
winter
002,002
1,2,1,1.0,0.0,-999.99,                       '.\Database\DRN\VERSION_1\IPEST_DRAINAGE_CONDUCTANCE.IDF' >>> (CON) Conductance (IDF) <<<
1,2,0,1.0,0.0,-999.99,'.\Database\DRN\VERSION_1\RIVER_SECUNDAIR\IPEST_RIVER_SECUNDAIR_COND_WINTER.IDF' >>> (CON) Conductance (IDF) <<<
1,2,1,1.0,0.0,-999.99,                                   '.\Database\DRN\VERSION_1\DRAINAGE_STAGE.IDF' >>> (DEL) Drainage Level (IDF) <<<
1,2,0,1.0,0.0,-999.99,    '.\Database\DRN\VERSION_1\RIVER_SECUNDAIR\RIVER_SECUNDAIR_BOTTOM_WINTER.IDF' >>> (DEL) Drainage Level (IDF) <<<
summer
002,002
1,2,1,1.0,0.0,-999.99,                       '.\Database\DRN\VERSION_1\IPEST_DRAINAGE_CONDUCTANCE.IDF' >>> (CON) Conductance (IDF) <<<
1,2,0,1.0,0.0,-999.99,'.\Database\DRN\VERSION_1\RIVER_SECUNDAIR\IPEST_RIVER_SECUNDAIR_COND_WINTER.IDF' >>> (CON) Conductance (IDF) <<<
1,2,1,1.0,0.0,-999.99,                                   '.\Database\DRN\VERSION_1\DRAINAGE_STAGE.IDF' >>> (DEL) Drainage Level (IDF) <<<
1,2,0,1.0,0.0,-999.99,    '.\Database\DRN\VERSION_1\RIVER_SECUNDAIR\RIVER_SECUNDAIR_BOTTOM_WINTER.IDF' >>> (DEL) Drainage Level (IDF) <<<


0002,(GHB),1, general
winter
002,001
1,2,1,1.0,0.0,-999.99,                       '.\Database\DRN\VERSION_1\IPEST_DRAINAGE_CONDUCTANCE.IDF' >>> (CON) Conductance (IDF) <<<
1,2,1,1.0,0.0,-999.99,                                   '.\Database\DRN\VERSION_1\DRAINAGE_STAGE.IDF' >>> (DEL) Drainage Level (IDF) <<<
summer
002,001
1,2,1,1.0,0.0,-999.99,                       '.\Database\DRN\VERSION_1\IPEST_DRAINAGE_CONDUCTANCE.IDF' >>> (CON) Conductance (IDF) <<<
1,2,1,1.0,0.0,-999.99,                                   '.\Database\DRN\VERSION_1\DRAINAGE_STAGE.IDF' >>> (DEL) Drainage Level (IDF) <<<

0001,(CHD),1, Constant Head
STEADY-STATE
001,37
1,2,1,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L1.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,2,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L2.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,3,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L3.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,4,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L4.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,5,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L5.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,6,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L6.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,7,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L7.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,8,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L8.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,9,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L9.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,10,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L10.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,11,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L11.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,12,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L12.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,13,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L13.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,14,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L14.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,15,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L15.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,16,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L16.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,17,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L17.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,18,1.0,0.0,-999.99, '.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L18.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,19,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L19.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,20,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L20.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,21,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L21.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,22,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L22.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,23,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L23.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,24,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L24.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,25,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L25.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,26,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L26.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,27,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L27.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,28,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L28.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,29,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L29.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,30,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L30.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,31,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L31.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,32,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L32.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,33,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L33.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,34,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L34.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,35,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L35.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,36,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L36.IDF' >>> (SHD) Starting Heads (IDF) <<<
1,2,37,1.0,0.0,-999.99,'.\Database\SHD\VERSION_1\STATIONAIR\\25\HEAD_STEADY-STATE_L37.IDF' >>> (SHD) Starting Heads (IDF) <<<


0001,(PCG),1, Precondition Conjugate-Gradient
 MXITER=  5000
 ITER1=   20
 HCLOSE=  0.1000000E-02
 RCLOSE=  0.1000000
 RELAX=   0.9800000
 NPCOND=  1
 IPRPCG=  1
 MUTPCG=  0
 DAMPPCG= 1.000000
 DAMPPCGT=1.000000
 IQERROR= 0
 QERROR=  0.1000000

Periods
summer
01-04-1990 00:00:00
winter
01-10-1990 00:00:00

Species
"benzene",1

"""

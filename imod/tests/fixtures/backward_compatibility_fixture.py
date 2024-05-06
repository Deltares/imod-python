import pytest
import xarray as xr

import imod


@pytest.fixture(scope="module")
def imod5_dataset():
    tmp_path = imod.util.temporary_directory()
    data = imod.data.imod5_projectfile_data(tmp_path)
    _load_imod5_data_in_memory(data[0])
    return data[0]


def _load_imod5_data_in_memory(imod5_data):
    """For debugging purposes, load everything in memory"""
    for pkg in imod5_data.values():
        for vardata in pkg.values():
            if isinstance(vardata, xr.DataArray):
                vardata.load()

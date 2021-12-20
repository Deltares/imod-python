import os

import xarray as xr

import imod


def test_twri_output():
    dirpath = imod.util.temporary_directory()
    imod.data.twri_output(dirpath)
    contents = os.listdir(dirpath)

    assert contents == [
        "twri.cbc",
        "twri.grb",
        "twri.hds",
    ]


def test_hondsrug_data():
    for f in [
        imod.data.hondsrug_initial,
        imod.data.hondsrug_layermodel,
        imod.data.hondsrug_meteorology,
        imod.data.hondsrug_river,
        imod.data.hondsrug_drainage,
    ]:
        assert isinstance(f(), xr.Dataset)

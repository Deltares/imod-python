import os
from pathlib import Path

import xarray as xr

import imod


def test_twri_output():
    dirpath = imod.util.temporary_directory()
    imod.data.twri_output(dirpath)
    contents = sorted(os.listdir(dirpath))

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


def test_tutorial_03_data__unzipped(tmp_path):
    """
    Test if tutorial 03 material can be unzipped.
    """
    # Act
    prj_path = imod.data.tutorial_03(tmp_path)

    # Assert
    assert isinstance(prj_path, Path)
    assert prj_path.suffix == ".prj"
    n_files = sum(1 for x in prj_path.parent.rglob("*") if x.is_file())
    assert n_files == 107


def test_tutorial_03_data__open_data(tmp_path):
    """
    Test if tutorial 03 material can be opened.
    """
    # Act
    prj_path = imod.data.tutorial_03(tmp_path)
    imod5_data, _ = imod.formats.prj.open_projectfile_data(prj_path)

    # Assert
    expected_keys = {
        "bnd",
        "top",
        "bot",
        "khv",
        "kva",
        "sto",
        "shd",
        "rch",
        "riv",
        "drn-1",
        "drn-2",
        "drn-3",
        "drn-4",
        "pcg",
    }
    missing_keys = expected_keys - set(imod5_data.keys())
    assert len(missing_keys) == 0

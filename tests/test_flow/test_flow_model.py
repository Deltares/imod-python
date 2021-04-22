import pytest
import imod.flow as flow
from copy import deepcopy

import numpy as np
import xarray as xr

import os


@pytest.fixture(scope="module")
def model(basic_dis, three_days, well_df):
    ibound, top, bottom = basic_dis

    times = three_days

    # Layer properties
    kh = 10.0
    kva = 1.0
    sto = 0.001

    # Initial conditions
    starting_head = ibound.copy()

    # Boundary_conditions
    # Create rising trend
    trend = np.cumsum(np.ones(times[:-1].shape))
    trend = xr.DataArray(trend, coords={"time": times[:-1]}, dims=["time"])

    x = ibound.x.values

    head = starting_head.where(starting_head.x.isin([x[0], x[-1]]))
    head = trend * head

    name = "testmodel"

    # Build model
    m = flow.ImodflowModel(name)
    m["pcg"] = flow.PreconditionedConjugateGradientSolver()

    m["bnd"] = flow.Boundary(ibound)
    m["top"] = flow.Top(top)
    m["bottom"] = flow.Bottom(bottom)

    m["khv"] = flow.HorizontalHydraulicConductivity(kh)
    m["kva"] = flow.VerticalAnisotropy(kva)
    m["sto"] = flow.StorageCoefficient(sto)

    m["shd"] = flow.StartingHead(starting_head)

    m["chd"] = flow.ConstantHead(head=10.0)
    m["chd2"] = flow.ConstantHead(head=head)

    m["wel"] = flow.Well(**well_df)
    m.time_discretization(times[-1])

    return m


@pytest.fixture(scope="module")
def model_metaswap(model, metaswap_dict):
    m = deepcopy(model)
    m["cap"] = flow.MetaSwap(**metaswap_dict)

    return m


@pytest.fixture(scope="module")
def model_horizontal_flow_barrier(model, horizontal_flow_barrier_gdf):
    m = deepcopy(model)
    m["hfb"] = flow.HorizontalFlowBarrier(**horizontal_flow_barrier_gdf)

    return m


def test_compose_all_packages(model, tmp_path):
    def depth(d):
        """Recursively walk through nested dict"""
        if isinstance(d, dict):
            return 1 + (max(map(depth, d.values())) if d else 0)
        return 0

    modeldir = tmp_path
    diskey = model._get_pkgkey("dis")
    globaltimes = model[diskey]["time"].values
    composition = model._compose_all_packages(modeldir, globaltimes)

    packages = set(
        [
            "bnd",
            "top",
            "bot",
            "shd",
            "dis",
            "chd",
            "wel",
            "pcg",
            "khv",
            "kva",
            "sto",
        ]
    )

    assert set(composition.keys()) <= packages
    assert depth(composition) == 5


def test_write_model(model, tmp_path):
    model.write(directory=tmp_path)

    # Test if prjfile at least has the right amount of lines
    prjfile = tmp_path / "testmodel.prj"
    with open(prjfile) as f:
        lines = f.readlines()

    assert len(lines) == 73

    # Recursively walk through folder and count files
    n_files = sum([len(files) for r, d, files in os.walk(tmp_path)])

    assert n_files == 18

    # Test if file and directorynames in tmp_path match the following
    files_directories = set(
        [
            "bnd",
            "chd2",
            "config_run.ini",
            "shd",
            "testmodel.prj",
            "time_discretization.tim",
            "wel",
        ]
    )

    symmetric_difference = files_directories ^ set(os.listdir(tmp_path))

    assert len(symmetric_difference) == 0


def test_write_model_metaswap(model_metaswap, tmp_path):
    model_metaswap.write(directory=tmp_path)

    # Test if prjfile at least has the right amount of lines
    prjfile = tmp_path / "testmodel.prj"
    with open(prjfile) as f:
        lines = f.readlines()

    assert len(lines) == 107

    # Recursively walk through folder and count files
    n_files = sum([len(files) for r, d, files in os.walk(tmp_path)])

    assert n_files == 24

    # Test if file and directorynames in tmp_path match the following
    files_directories = set(
        [
            "bnd",
            "cap",
            "chd2",
            "config_run.ini",
            "shd",
            "testmodel.prj",
            "time_discretization.tim",
            "wel",
        ]
    )

    symmetric_difference = files_directories ^ set(os.listdir(tmp_path))

    assert len(symmetric_difference) == 0


def test_write_model_horizontal_flow_barrier(model_horizontal_flow_barrier, tmp_path):
    model_horizontal_flow_barrier.write(directory=tmp_path)

    # Test if prjfile at least has the right amount of lines
    prjfile = tmp_path / "testmodel.prj"
    with open(prjfile) as f:
        lines = f.readlines()

    assert len(lines) == 78

    # Recursively walk through folder and count files
    n_files = sum([len(files) for r, d, files in os.walk(tmp_path)])

    assert n_files == 20

    # Test if file and directorynames in tmp_path match the following
    files_directories = set(
        [
            "bnd",
            "chd2",
            "config_run.ini",
            "hfb",
            "shd",
            "testmodel.prj",
            "time_discretization.tim",
            "wel",
        ]
    )

    symmetric_difference = files_directories ^ set(os.listdir(tmp_path))

    assert len(symmetric_difference) == 0
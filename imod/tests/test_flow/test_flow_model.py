import os
from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

import imod.flow as flow


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
    m["oc"] = flow.OutputControl()

    m.create_time_discretization(times[-1])

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


@pytest.fixture(scope="module")
def model_no_output_control(model):
    m = deepcopy(model)
    m.pop("oc")

    return m


@pytest.fixture(scope="module")
def model_periodic_stress(model):
    m = deepcopy(model)
    times = np.array([np.datetime64("2000-01-01"), np.datetime64("2000-01-02")])

    head_periodic = xr.DataArray([2.0, 1.0], coords={"time": times}, dims=["time"])

    timemap = {times[0]: "summer", times[1]: "winter"}

    m["ghb"] = flow.GeneralHeadBoundary(head=head_periodic, conductance=10.0)
    m["ghb"].periodic_stress(timemap)

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
            "oc",
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

    assert len(lines) == 75

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


def test_write_model_no_oc(model_no_output_control, tmp_path):
    with pytest.raises(ValueError):
        model_no_output_control.write(directory=tmp_path)


def test_write_model_metaswap(model_metaswap, tmp_path):
    model_metaswap.write(directory=tmp_path)

    # Test if prjfile at least has the right amount of lines
    prjfile = tmp_path / "testmodel.prj"
    with open(prjfile) as f:
        lines = f.readlines()

    assert len(lines) == 109

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

    assert len(lines) == 80

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


def test_compose_periods(model_periodic_stress):
    periods_composed = model_periodic_stress._compose_periods()

    compare = {"summer": "01-01-2000 00:00:00", "winter": "02-01-2000 00:00:00"}

    assert periods_composed == compare


def test_render_periods(model_periodic_stress):
    periods_composed = model_periodic_stress._compose_periods()

    compare = (
        "Periods\n" "summer\n" "01-01-2000 00:00:00\n" "winter\n" "02-01-2000 00:00:00"
    )

    rendered = model_periodic_stress._render_periods(periods_composed)

    assert compare == rendered


def test_write_model_periodic(model_periodic_stress, tmp_path):
    model_periodic_stress.write(directory=tmp_path)

    # Test if prjfile at least has the right amount of lines
    prjfile = tmp_path / "testmodel.prj"
    with open(prjfile) as f:
        lines = f.readlines()

    assert len(lines) == 97

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


def test_convert_to_runfile(model_periodic_stress, tmp_path):
    model_periodic_stress.write(directory=tmp_path, convert_to="runfile")

    # Test if inifile at least has the right amount of lines
    inifile = tmp_path / "config_run.ini"
    with open(inifile) as f:
        text = f.read()

    assert "function=runfile" in text
    assert "runfile_out" in text
    assert "sim_type=1" in text
    assert ".run"


def test_convert_to_mf2005(model_periodic_stress, tmp_path):
    model_periodic_stress.write(directory=tmp_path, convert_to="mf2005_namfile")

    # Test if inifile at least has the right amount of lines
    inifile = tmp_path / "config_run.ini"
    with open(inifile) as f:
        text = f.read()

    assert "function=runfile" in text
    assert "namfile_out" in text
    assert "sim_type=2" in text
    assert ".nam" in text


def test_convert_to_mf6(model_periodic_stress, tmp_path):
    model_periodic_stress.write(directory=tmp_path, convert_to="mf6_namfile")

    # Test if inifile at least has the right amount of lines
    inifile = tmp_path / "config_run.ini"
    with open(inifile) as f:
        text = f.read()

    assert "function=runfile" in text
    assert "namfile_out" in text
    assert "sim_type=3" in text
    assert ".nam" in text


def test_wrong_convert_setting(model_periodic_stress, tmp_path):
    with pytest.raises(ValueError):
        model_periodic_stress.write(directory=tmp_path, convert_to="failure")

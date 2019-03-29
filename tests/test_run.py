import pytest
import os
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import imod
from imod import run
from collections import OrderedDict
from pathlib import Path


@pytest.fixture(scope="module")
def make_test_model(request):
    def _make_test_model(transient=False):
        nrow, ncol, nlayer = 8, 10, 3
        times = [
            pd.to_datetime(s)
            for s in [
                "2012-01-01 00:00",
                "2012-02-01 00:00",
                "2012-03-01 00:00",
                "2012-04-01 00:00",
                "2012-05-01 00:00",
            ]
        ]
        ntime = len(times)

        data2d = np.random.rand(nrow, ncol)
        data2d_t = np.random.rand(nrow, ncol, ntime)
        data3d = np.random.rand(nrow, ncol, nlayer)

        dims2d = ("y", "x")
        dims2d_t = ("y", "x", "time")
        dims3d = ("y", "x", "layer")

        coords2d = {
            "y": np.linspace(nrow - 1, 0, num=nrow),
            "x": np.arange(ncol),
            "dx": 1.0,
            "dy": -1.0,
        }
        coords2d_t = {
            "y": np.linspace(nrow - 1, 0, num=nrow),
            "x": np.arange(ncol),
            "time": times,
            "dx": 1.0,
            "dy": -1.0,
        }
        coords3d = {
            "y": np.linspace(nrow - 1, 0, num=nrow),
            "x": np.arange(ncol),
            "layer": np.arange(nlayer) + 1,
            "dx": 1.0,
            "dy": -1.0,
        }

        da2d = xr.DataArray(data2d, coords2d, dims2d)
        da2d_t = xr.DataArray(data2d_t, coords2d_t, dims2d_t)
        da3d = xr.DataArray(data3d, coords3d, dims3d)

        if transient:
            rch_da = da2d_t.assign_coords(layer=-1)
            ghb_head_da = da2d_t.assign_coords(layer=1)
        else:
            rch_da = da2d.assign_coords(layer=-1)
            ghb_head_da = da2d.assign_coords(layer=1)

        ghb_cond_da = da2d.assign_coords(layer=1)

        modeldata = OrderedDict(
            [
                ("bnd", da3d),
                ("shd", da3d),
                ("kdw", da3d),
                ("vcw", da3d),
                ("ghb-head", ghb_head_da),
                ("ghb-cond", ghb_cond_da),
                ("ghb-head-sys2", ghb_head_da),
                ("ghb-cond-sys2", ghb_cond_da),
                ("rch", rch_da),
            ]
        )
        return {
            "modeldata": modeldata,
            "nlayer": nlayer,
            "times": times,
            "ntime": ntime,
        }

    def runfile_teardown():
        try:
            os.remove("runfile.run")
        except:
            pass

        try:
            shutil.rmtree("test_write")
        except:
            pass

    request.addfinalizer(runfile_teardown)
    return _make_test_model


def test_check_input():
    m = OrderedDict()
    m["bnd"] = xr.DataArray(
        np.random.rand(4, 3),
        coords={"y": range(3, -1, -1), "x": range(3), "dx": 1.0, "dy": -1.0},
        dims=("y", "x"),
    )
    m["wel"] = pd.DataFrame({"x": [0, 2], "y": [1, 2], "q": [2000.0, 3000.0]})
    out = run._check_input(m)
    assert len(m) == len(out)


def test_check_input__error():
    m = OrderedDict()
    m["wel"] = xr.DataArray(
        np.random.rand(4, 3),
        coords={"y": range(3, -1, -1), "x": range(3), "dx": 1.0, "dy": -1.0},
        dims=("y", "x"),
    )
    m["bnd"] = pd.DataFrame({"x": [0, 2], "y": [1, 2], "q": [2000.0, 3000.0]})
    with pytest.raises(TypeError):
        run._check_input(m)


def test_data_bounds(make_test_model):
    test_model = make_test_model(transient=True)
    d = run._data_bounds(test_model["modeldata"])
    assert d["nlay"] == 3
    assert d["xmin"] == -0.5
    assert d["xmax"] == 9.5
    assert d["ymin"] == -0.5
    assert d["ymax"] == 7.5
    assert d["dx"] == 1.0
    assert d["dy"] == 1.0
    assert d["nper"] == 4
    assert d["times"][0] == np.datetime64("2012-01-01 00:00")
    assert d["times"][-1] == np.datetime64("2012-05-01 00:00")
    assert d["nper"] + 1 == len(d["times"])


def test_parse__riv():
    key = "riv-stage"
    d = run._parse(key, run.stress_period_schema)
    assert d == {"name": "riv", "field": "stage"}


def test_parse__rivsystems():
    key = "riv-stage-primary"
    d = run._parse(key, run.stress_period_schema)
    assert d == {"name": "riv", "field": "stage", "system": "primary"}


def test_parse__wel():
    key = "wel"
    d = run._parse(key, run.stress_period_schema)
    assert d == {"name": "wel"}


def test_parse__welsystems():
    key = "wel-sys1"
    d = run._parse(key, run.stress_period_schema)
    assert d == {"name": "wel", "system": "sys1"}


def test_parse__errors():
    # package without systems
    key = "rch-sys1"
    with pytest.raises(ValueError):
        run._parse(key, run.stress_period_schema)

    # misnamed field
    key = "riv-head"
    with pytest.raises(ValueError):
        run._parse(key, run.stress_period_schema)

    # mixup of field and system name
    key = "riv-sys1-stage"
    with pytest.raises(ValueError):
        run._parse(key, run.stress_period_schema)

    # one descriptor too much
    key = "riv-stage-sys1-v2"
    with pytest.raises(RuntimeError):
        run._parse(key, run.stress_period_schema)


def test_time_discretisation():
    dates = ["2012-01-01 12:00", "2012-01-02 00:00", "2012-01-12 00:00"]
    times = [pd.to_datetime(s, format="%Y-%m-%d %H:%M") for s in dates]
    d = run._time_discretisation(times)
    assert len(d) == 2

    for i, periodname in enumerate(d.keys()):
        assert times[i].strftime("%Y%m%d%H%M%S") == periodname

    values = list(d.values())
    assert values[0] == 0.5
    assert values[1] == 10.0


def test_pop_package():
    m = OrderedDict()
    m["bnd"] = None
    m["riv-stage"] = None
    m["riv-cond"] = None
    m, package = run._pop_package(m, "riv")
    assert len(m) == 1
    assert len(package) == 2


def test_get_package():
    nrow, ncol, nlayer = 8, 10, 2
    data = np.random.rand(nrow, ncol, nlayer)
    dims = ("y", "x", "layer")
    coords = {
        "y": np.linspace(nrow - 1, 0, num=nrow),
        "x": np.arange(ncol),
        "layer": np.arange(nlayer) + 1,
        "dx": 1.0,
        "dy": -1.0,
    }
    da = xr.DataArray(data=data, coords=coords, dims=dims)
    package = {"ani-angle": da, "ani-factor": da}
    directory = Path("dbase")
    d = run._get_package(package, directory, run.package_schema)
    assert isinstance(d, OrderedDict)


def test_get_package__error():
    # misses one field (ani-factor)
    nrow, ncol, nlayer = 8, 10, 2
    data = np.random.rand(nrow, ncol, nlayer)
    dims = ("y", "x", "layer")
    coords = {
        "y": np.linspace(nrow - 1, 0, num=nrow),
        "x": np.arange(ncol),
        "layer": np.arange(nlayer) + 1,
        "dx": 1.0,
        "dy": -1.0,
    }
    da = xr.DataArray(data=data, coords=coords, dims=dims)
    package = {"ani-angle": da}
    directory = Path("dbase")
    with pytest.raises(KeyError):
        run._get_package(package, directory, run.package_schema)


def test_get_period():
    nrow, ncol, nlayer, ntime = 8, 10, 2, 3
    data = np.random.rand(nrow, ncol, nlayer, ntime)
    dims = ("y", "x", "layer", "time")
    times = list(map(pd.to_datetime, ["2012-01-01", "2013-01-01", "2014-01-01"]))
    coords = {
        "y": np.linspace(nrow - 1, 0, num=nrow),
        "x": np.arange(ncol),
        "layer": np.arange(nlayer) + 1,
        "time": times,
        "dx": 1.0,
        "dy": -1.0,
    }
    da = xr.DataArray(data=data, coords=coords, dims=dims)
    package = {"riv-stage": da, "riv-cond": da, "riv-bot": da, "riv-inff": da}
    directory = Path("dbase")

    d = run._get_period(package, times, directory, run.stress_period_schema)
    assert isinstance(d, OrderedDict)


def test_get_runfile__steady_state(make_test_model):
    test_model = make_test_model(transient=False)
    modeldata = test_model["modeldata"]
    nlayer = test_model["nlayer"]
    d = run.get_runfile(modeldata, directory=Path("dbase"))

    assert d["nper"] == 1

    for name in ["bnd", "shd", "kdw", "vcw"]:
        package = d["packages"][name]["value"]
        for layer in range(1, nlayer + 1):
            assert (
                package[layer]
                == Path("dbase/{}/{}_l{}.idf".format(name, name, layer)).absolute()
            )

    assert (
        d["stress_periods"]["rch"]["value"]["default_system"][-1][0]
        == Path("dbase/rch/rch_l-1.idf").absolute()
    )
    assert (
        d["stress_periods"]["ghb"]["head"]["default_system"][1][0]
        == Path("dbase/ghb/ghb-head_l1.idf").absolute()
    )
    assert (
        d["stress_periods"]["ghb"]["cond"]["default_system"][1][0]
        == Path("dbase/ghb/ghb-cond_l1.idf").absolute()
    )
    assert (
        d["stress_periods"]["ghb"]["head"]["sys2"][1][0]
        == Path("dbase/ghb/ghb-head-sys2_l1.idf").absolute()
    )
    assert (
        d["stress_periods"]["ghb"]["cond"]["sys2"][1][0]
        == Path("dbase/ghb/ghb-cond-sys2_l1.idf").absolute()
    )


def test_get_runfile__transient(make_test_model):
    test_model = make_test_model(transient=True)
    modeldata = test_model["modeldata"]
    times = test_model["times"]
    nlayer = test_model["nlayer"]
    ntime = test_model["ntime"]
    d = run.get_runfile(modeldata, directory=Path("dbase"))

    assert d["nper"] == ntime - 1

    for name in ["bnd", "shd", "kdw", "vcw"]:
        package = d["packages"][name]["value"]
        for layer in range(1, nlayer + 1):
            assert (
                package[layer]
                == Path("dbase/{}/{}_l{}.idf".format(name, name, layer)).absolute()
            )

    for i, time in enumerate(times[:-1]):
        assert (
            d["stress_periods"]["rch"]["value"]["default_system"][-1][i]
            == Path(
                "dbase/rch/rch_{}_l-1.idf".format(time.strftime("%Y%m%d%H%M%S"))
            ).absolute()
        )
        assert (
            d["stress_periods"]["ghb"]["head"]["default_system"][1][i]
            == Path(
                "dbase/ghb/ghb-head_{}_l1.idf".format(time.strftime("%Y%m%d%H%M%S"))
            ).absolute()
        )
        assert (
            d["stress_periods"]["ghb"]["cond"]["default_system"][1][i]
            == Path("dbase/ghb/ghb-cond_l1.idf").absolute()
        )
        assert (
            d["stress_periods"]["ghb"]["head"]["sys2"][1][i]
            == Path(
                "dbase/ghb/ghb-head-sys2_{}_l1.idf".format(
                    time.strftime("%Y%m%d%H%M%S")
                )
            ).absolute()
        )
        assert (
            d["stress_periods"]["ghb"]["cond"]["sys2"][1][i]
            == Path("dbase/ghb/ghb-cond-sys2_l1.idf").absolute()
        )


def test_write_runfile__steady_state(make_test_model):
    test_model = make_test_model(transient=False)
    modeldata = test_model["modeldata"]
    directory = Path(os.getcwd())
    runfile_parameters = run.get_runfile(modeldata, directory)
    path = directory.joinpath("runfile.run")
    run.write_runfile(path, runfile_parameters)

    with open("runfile.run") as f:
        testcontent = f.readlines()

    # TODO: think of robust, less dumb test
    assert len(testcontent) == 38


def test_write_runfile__well_steady_state(make_test_model):
    test_model = make_test_model(transient=False)
    modeldata = test_model["modeldata"]
    weldata = pd.DataFrame(
        {
            "x": np.random.rand(10),
            "y": np.random.rand(10),
            "q": np.random.rand(10),
            "layer": np.full(10, 1),
        }
    )
    modeldata["wel"] = weldata
    directory = Path(os.getcwd())
    runfile_parameters = run.get_runfile(modeldata, directory)
    path = directory.joinpath("runfile.run")
    run.write_runfile(path, runfile_parameters)

    with open("runfile.run") as f:
        testcontent = f.readlines()

    # TODO: think of robust, less dumb test
    assert len(testcontent) == 41


def test_write_runfile__transient(make_test_model):
    test_model = make_test_model(transient=True)
    modeldata = test_model["modeldata"]
    directory = Path(os.getcwd())
    runfile_parameters = run.get_runfile(modeldata, directory)
    path = directory.joinpath("runfile.run")
    run.write_runfile(path, runfile_parameters)

    with open("runfile.run") as f:
        testcontent = f.readlines()

    # TODO: think of robust, less dumb test
    assert len(testcontent) == 62


def test_write_runfile__well_transient(make_test_model):
    test_model = make_test_model(transient=True)
    times = test_model["times"]
    modeldata = test_model["modeldata"]

    df = pd.DataFrame(
        {
            "x": np.random.rand(10),
            "y": np.random.rand(10),
            "q": np.random.rand(10),
            "layer": np.full(10, 1),
        }
    )

    dfs = []
    for t in times[:-1]:
        df_t = df.copy()
        df_t["time"] = t
        dfs.append(df_t)

    modeldata["wel"] = pd.concat(dfs, sort=False)
    directory = Path(os.getcwd())
    runfile_parameters = run.get_runfile(modeldata, directory)
    path = directory.joinpath("runfile.run")
    run.write_runfile(path, runfile_parameters)

    with open("runfile.run") as f:
        testcontent = f.readlines()

    # TODO: think of robust, less dumb test
    assert len(testcontent) == 71


def test_write__transient(make_test_model):
    test_model = make_test_model(transient=True)
    modeldata = test_model["modeldata"]
    directory = Path(os.getcwd())
    path = directory.joinpath("test_write")
    imod.write(path, modeldata)

    nlayer = test_model["nlayer"]
    runfile_parameters = run.get_runfile(modeldata, path)
    for name in ["bnd", "shd", "kdw", "vcw"]:
        package = runfile_parameters["packages"][name]["value"]
        for layer in range(1, 1 + nlayer):
            assert Path(package[layer]).exists()

    for name in ["ghb", "rch"]:
        package = runfile_parameters["stress_periods"][name]
        for field in package.values():
            for system in field.values():
                for layer in system.values():
                    for path in layer:  # data of single stress period
                        assert Path(path).exists()


def test_write__basic_seawat(make_test_model):
    nrow, ncol, nlayer = 8, 10, 3
    times = [
        pd.to_datetime(s)
        for s in [
            "2012-01-01 00:00",
            "2012-02-01 00:00",
            "2012-03-01 00:00",
            "2012-04-01 00:00",
            "2012-05-01 00:00",
        ]
    ]
    ntime = len(times)

    data2d = np.random.rand(nrow, ncol)
    data2d_t = np.random.rand(nrow, ncol, ntime)
    data3d = np.random.rand(nrow, ncol, nlayer)

    dims2d = ("y", "x")
    dims2d_t = ("y", "x", "time")
    dims3d = ("y", "x", "layer")

    coords2d = {
        "y": np.linspace(nrow - 1, 0, num=nrow),
        "x": np.arange(ncol),
        "dx": 1.0,
        "dy": -1.0,
    }
    coords2d_t = {
        "y": np.linspace(nrow - 1, 0, num=nrow),
        "x": np.arange(ncol),
        "time": times,
        "dx": 1.0,
        "dy": -1.0,
    }
    coords3d = {
        "y": np.linspace(nrow - 1, 0, num=nrow),
        "x": np.arange(ncol),
        "layer": np.arange(nlayer) + 1,
        "dx": 1.0,
        "dy": -1.0,
    }

    da2d = xr.DataArray(data2d, coords2d, dims2d)
    da2d_t = xr.DataArray(data2d_t, coords2d_t, dims2d_t)
    da3d = xr.DataArray(data3d, coords3d, dims3d)

    rch_da = da2d_t.assign_coords(layer=1)
    ghb_head_da = da2d_t.assign_coords(layer=1)
    ghb_cond_da = da2d.assign_coords(layer=1)

    model = OrderedDict(
        [
            ("bnd", da3d),
            ("icbund", da3d),
            ("top", da3d),
            ("bot", da3d),
            ("thickness", da3d),
            ("shd", da3d),
            ("sconc", da3d),
            ("khv", da3d),
            ("kva", da3d),
            ("sto", da3d),
            ("por", da3d),
            ("dsp-al", da3d),
            ("dsp-trpt", da3d),
            ("dsp-trpv", da3d),
            ("dsp-dmcoef", da3d),
            ("ghb-head", ghb_head_da),
            ("ghb-cond", ghb_cond_da),
            ("ghb-dens", ghb_head_da),
            ("ghb-conc", ghb_head_da),
            ("ghb-head-sys2", ghb_head_da),
            ("ghb-cond-sys2", ghb_cond_da),
            ("ghb-dens-sys2", ghb_head_da),
            ("rch-rate", rch_da),
            ("rch-conc", rch_da),
        ]
    )

    imod.seawat_write("test_write", model)

    runfile_parameters = imod.run.seawat_get_runfile(model, Path("test_write"))
    for name in ["top", "bot", "thickness", "shd", "sconc", "khv", "kva", "sto", "por"]:
        package = runfile_parameters["packages"][name]["value"]
        for layer in range(1, 1 + nlayer):
            assert Path(package[layer]).exists()

    package = runfile_parameters["packages"]["dsp"]["al"]
    for layer in range(1, 1 + nlayer):
        assert Path(package[layer]).exists()

    for name in ["ghb", "rch"]:
        package = runfile_parameters["stress_periods"][name]
        for field in package.values():
            for system in field.values():
                for layer in system.values():
                    for path in layer:  # data of single stress period
                        assert Path(path).exists()

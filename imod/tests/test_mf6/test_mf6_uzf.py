import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod
from imod.schemata import ValidationError


@pytest.fixture(scope="function")
def test_data():
    shape = nlay, nrow, ncol = 2, 2, 2
    dx, dy = 10, -10
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    nper = 4
    time = pd.date_range("2018-01-01", periods=nper, freq="H")
    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    idomain = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
    idomain[:, 0, 0] = 0
    active = idomain.where(idomain == 1)
    active_time = (
        xr.DataArray(np.ones(time.shape), coords={"time": time}, dims=("time",))
        * active
    )

    d = {}
    d["kv_sat"] = active * 10.0
    d["theta_sat"] = active * 0.1
    d["theta_res"] = active * 0.05
    d["theta_init"] = active * 0.08
    d["epsilon"] = active * 7.0
    d["surface_depression_depth"] = active * 1.0
    d["infiltration_rate"] = active_time * 0.003
    d["et_pot"] = active_time * 0.002
    d["extinction_depth"] = active_time * 1.0
    d["groundwater_ET_function"] = "linear"
    return d


def test_wrong_dtype(test_data):
    test_data["kv_sat"] = test_data["kv_sat"].astype(np.int32)
    with pytest.raises(ValidationError):
        imod.mf6.UnsaturatedZoneFlow(**test_data)


def test_landflag(test_data):
    expected = np.ones((2, 2, 2))
    expected[:, 0, 0] = 0
    uzf = imod.mf6.UnsaturatedZoneFlow(**test_data)
    assert np.all(uzf["landflag"].values == expected)


def test_iuzno(test_data):
    expected = np.array([[[0, 1], [2, 3]], [[0, 4], [5, 6]]])
    uzf = imod.mf6.UnsaturatedZoneFlow(**test_data)
    assert np.all(uzf["iuzno"].values == expected)


def test_ivertcon(test_data):
    expected = np.array([[[0, 4], [5, 6]], [[0, 0], [0, 0]]])
    uzf = imod.mf6.UnsaturatedZoneFlow(**test_data)
    assert np.all(uzf["ivertcon"].values == expected)


def test_checkoptions(test_data):
    uzf = imod.mf6.UnsaturatedZoneFlow(**test_data)
    assert bool(uzf["simulate_et"]) is True
    assert bool(uzf["linear_gwet"]) is True
    assert bool(uzf["simulate_gwseep"]) is False
    test_data.pop("extinction_depth")

    with pytest.raises(ValueError):
        uzf = imod.mf6.UnsaturatedZoneFlow(**test_data)


def test_to_sparsedata(test_data):
    uzf = imod.mf6.UnsaturatedZoneFlow(**test_data)
    uzf.fill_stress_perioddata()
    bin_data = uzf[list(uzf._period_data)]
    arrdict = uzf._ds_to_arrdict(bin_data.isel(time=0))
    layer = bin_data.isel(time=0)["layer"].values
    struct_array = uzf._to_struct_array(arrdict, layer)
    expected_iuzno = np.array([1, 2, 3, 4, 5, 6])

    assert struct_array.dtype[0] == np.dtype(
        "int32"
    )  # pylint: disable=unsubscriptable-object
    assert struct_array.dtype[1] == np.dtype(
        "float64"
    )  # pylint: disable=unsubscriptable-object
    assert np.all(struct_array["iuzno"] == expected_iuzno)
    assert len(struct_array.dtype) == 8
    assert len(struct_array) == 6


def test_fill_perioddata(test_data):
    uzf = imod.mf6.UnsaturatedZoneFlow(**test_data)
    assert uzf["root_potential"].item() is None
    uzf.fill_stress_perioddata()
    assert np.all(uzf["root_potential"] == xr.full_like(uzf["kv_sat"], 0.0))


def test_packagedata(test_data):
    uzf = imod.mf6.UnsaturatedZoneFlow(**test_data)
    packagedata = uzf._package_data_to_sparse()
    assert len(packagedata.dtype) == 12
    assert len(packagedata) == 6


def test_render(test_data):
    uzf = imod.mf6.UnsaturatedZoneFlow(**test_data)
    directory = pathlib.Path("mymodel")
    globaltimes = pd.date_range("2018-01-01", periods=4, freq="H")
    actual = uzf.render(directory, "uzf", globaltimes, True)

    expected = textwrap.dedent(
        """\
    begin options
      simulate_et
      linear_gwet
    end options

    begin dimensions
      nuzfcells 6
      ntrailwaves 7
      nwavesets 40
    end dimensions

    begin packagedata
      open/close mymodel/uzf/uzf-pkgdata.dat
    end packagedata

    begin period 1
      open/close mymodel/uzf/uzf-0.dat
    end period

    begin period 2
      open/close mymodel/uzf/uzf-1.dat
    end period

    begin period 3
      open/close mymodel/uzf/uzf-2.dat
    end period

    begin period 4
      open/close mymodel/uzf/uzf-3.dat
    end period

    """
    )
    assert actual == expected

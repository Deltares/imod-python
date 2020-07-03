# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:49:16 2020

@author: engelen
"""

import xarray as xr
import imod
import numpy as np
import pandas as pd

import pytest
import pathlib
import textwrap


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


def test_landflag():
    d = test_data()
    expected = np.ones((2, 2, 2))
    expected[:, 0, 0] = 0

    uzf = imod.mf6.UnsaturatedZoneFlow(**d)

    assert np.all(uzf["landflag"].values == expected)


def test_iuzno():
    d = test_data()
    expected = np.array([[[0, 1], [2, 3]], [[0, 4], [5, 6]]])

    uzf = imod.mf6.UnsaturatedZoneFlow(**d)

    assert np.all(uzf["iuzno"].values == expected)


def test_ivertcon():
    d = test_data()
    expected = np.array([[[0, 4], [5, 6]], [[0, 0], [0, 0]]])

    uzf = imod.mf6.UnsaturatedZoneFlow(**d)

    assert np.all(uzf["ivertcon"].values == expected)


def test_checkoptions():
    d = test_data()

    uzf = imod.mf6.UnsaturatedZoneFlow(**d)
    assert uzf["simulate_et"] == True
    assert uzf["linear_gwet"] == True
    assert uzf["simulate_gwseep"] == False

    d.pop("extinction_depth")
    with pytest.raises(ValueError):
        uzf = imod.mf6.UnsaturatedZoneFlow(**d)


def test_to_sparsedata():
    d = test_data()

    uzf = imod.mf6.UnsaturatedZoneFlow(**d)
    uzf.fill_stress_perioddata()
    bin_data = uzf[[*(uzf._binary_data)]]
    arrlist = uzf._ds_to_arrlist(bin_data.isel(time=0))
    layer = uzf._check_layer_presence(bin_data.isel(time=0))
    sparse_data = uzf.to_sparse(arrlist, layer)

    expected_iuzno = np.array([1, 2, 3, 4, 5, 6])

    assert sparse_data.dtype[0] == np.dtype("int32")
    assert sparse_data.dtype[1] == np.dtype("float64")
    assert np.all(sparse_data["iuzno"] == expected_iuzno)
    assert len(sparse_data.dtype) == 8
    assert len(sparse_data) == 6

    txtformat = uzf.get_textformat(sparse_data)
    assert txtformat == "%4d %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f"


def test_fill_perioddata():
    d = test_data()

    uzf = imod.mf6.UnsaturatedZoneFlow(**d)
    assert uzf["root_potential"] == None

    uzf.fill_stress_perioddata()
    assert np.all(uzf["root_potential"] == xr.full_like(uzf["kv_sat"], 0.0))


def test_packagedata():
    d = test_data()

    uzf = imod.mf6.UnsaturatedZoneFlow(**d)

    packagedata = uzf.get_packagedata()

    assert len(packagedata.dtype) == 12
    assert len(packagedata) == 6

    txtformat = uzf.get_textformat(packagedata)
    expected = "%4d %4d %4d %4d %4d %4d %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f"

    assert txtformat == expected


def test_render():
    d = test_data()

    uzf = imod.mf6.UnsaturatedZoneFlow(**d)

    directory = pathlib.Path("mymodel")
    globaltimes = pd.date_range("2018-01-01", periods=4, freq="H")
    actual = uzf.render(directory, "uzf", globaltimes)

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
      open/close mymodel/uzf/uzf-pkgdata.bin
    end packagedata
    
    begin period 1
      open/close mymodel/uzf/uzf-0.bin
    end period
    
    begin period 2
      open/close mymodel/uzf/uzf-1.bin
    end period
    
    begin period 3
      open/close mymodel/uzf/uzf-2.bin
    end period
    
    begin period 4
      open/close mymodel/uzf/uzf-3.bin
    end period
        
    
     """
    )
    assert actual == expected

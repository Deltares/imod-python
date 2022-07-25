import pathlib
import textwrap

import numpy as np

import imod


def test_buoyancy_package_simple():
    buy = imod.mf6.Buoyancy(
        reference_density=1000.0,
        reference_concentration=[4.0, 25.0],
        density_concentration_slope=[0.7, -0.375],
        modelname=["gwt-1", "gwt-2"],
        species=["salinity", "temperature"],
    )

    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    expected = textwrap.dedent(
        """\
    begin options
      denseref 1000.0
    end options

    begin dimensions
      nrhospecies 2
    end dimensions

    begin packagedata
      1 0.7 4.0 gwt-1 salinity
      2 -0.375 25.0 gwt-2 temperature
    end packagedata"""
    )
    actual = buy.render(directory, "buy", globaltimes, False)
    print(actual)
    print(expected)
    assert actual == expected


def test_buoyancy_package_full():
    buy = imod.mf6.Buoyancy(
        reference_density=993.0,
        reference_concentration=[4.0, 25.0],
        density_concentration_slope=[0.7, -0.375],
        modelname=["gwt-1", "gwt-2"],
        species=["salinity", "temperature"],
        densityfile="density_out.dat",
        hhformulation_rhs=True,
    )
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    expected = textwrap.dedent(
        """\
    begin options
      hhformulation_rhs
      denseref 993.0
      density fileout density_out.dat
    end options

    begin dimensions
      nrhospecies 2
    end dimensions

    begin packagedata
      1 0.7 4.0 gwt-1 salinity
      2 -0.375 25.0 gwt-2 temperature
    end packagedata"""
    )
    actual = buy.render(directory, "buy", globaltimes, False)
    print(actual)
    print(expected)
    assert actual == expected

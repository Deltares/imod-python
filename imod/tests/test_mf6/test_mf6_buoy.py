import pathlib
import textwrap

import numpy as np

import imod


def test_buoyancy_package_simple():
    buy = imod.mf6.Buoyancy()
    buy.add_species_dependency(0.7, 4, "gwt-1", "salinity")
    buy.add_species_dependency(-0.375, 25, "gwt-2", "temperature")

    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    expected = textwrap.dedent(
        """\
    begin options
    end options

    begin dimensions
        nrhospecies 2
    end dimensions

    begin packagedata
        1  0.7  4    gwt-1   salinity
        2  -0.375  25    gwt-2   temperature
    end packagedata"""
    )
    actual = buy.render(directory, "buy", globaltimes, False)
    assert actual == expected


def test_buoyancy_package_full():
    buy = imod.mf6.Buoyancy(
        hhformulation_rhs=True, denseref=993, densityfile="density_out.dat"
    )
    buy.add_species_dependency(0.7, 4, "gwt-1", "salinity")
    buy.add_species_dependency(-0.375, 25, "gwt-2", "temperature")

    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    expected = textwrap.dedent(
        """\
    begin options
        hhformulation_rhs
        denseref 993
        densityfile fileout density_out.dat
    end options

    begin dimensions
        nrhospecies 2
    end dimensions

    begin packagedata
        1  0.7  4    gwt-1   salinity
        2  -0.375  25    gwt-2   temperature
    end packagedata"""
    )
    actual = buy.render(directory, "buy", globaltimes, False)
    assert actual == expected

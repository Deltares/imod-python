import pathlib
import textwrap

import numpy as np
import pytest

import imod


def test_viscosity_package_simple():
    vsc = imod.mf6.Viscosity(
        reference_viscosity=8.904e-04,
        viscosity_concentration_slope=[1.92e-6],
        reference_concentration=[0.0],
        modelname=["gwt-1"],
        species=["salinity"],
    )

    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    expected = textwrap.dedent(
        """\
    begin options
      viscref 0.0008904
      thermal_formulation linear
      thermal_a2 10.0
      thermal_a3 248.37
      thermal_a4 133.15
    end options

    begin dimensions
      nviscspecies 1
    end dimensions

    begin packagedata
      1 1.92e-06 0.0 gwt-1 salinity
    end packagedata"""
    )
    actual = vsc._render(directory, "vsc", globaltimes, False)

    assert actual == expected


def test_viscosity_package_full():
    vsc = imod.mf6.Viscosity(
        reference_viscosity=8.904e-04,
        viscosity_concentration_slope=[1.92e-6, 0.0],
        reference_concentration=[0.0, 25.0],
        modelname=["gwt-1", "gwt-2"],
        species=["salinity", "temperature"],
        temperature_species_name="temperature",
        thermal_formulation="nonlinear",
        thermal_a2=20.0,
        thermal_a3=348.37,
        thermal_a4=233.15,
        viscosityfile="testfile.vsc"
    )

    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    expected = textwrap.dedent(
        """\
    begin options
      viscref 0.0008904
      temperature_species_name temperature
      thermal_formulation nonlinear
      thermal_a2 20.0
      thermal_a3 348.37
      thermal_a4 233.15
      viscosity fileout testfile.vsc
    end options

    begin dimensions
      nviscspecies 2
    end dimensions

    begin packagedata
      1 1.92e-06 0.0 gwt-1 salinity
      2 0.0 25.0 gwt-2 temperature
    end packagedata"""
    )
    actual = vsc._render(directory, "vsc", globaltimes, False)

    assert actual == expected


def test_viscosity_package_update_transport_names():
    vsc = imod.mf6.Viscosity(
        reference_viscosity=8.904e-04,
        viscosity_concentration_slope=[1.92e-6, 0.0],
        reference_concentration=[0.0, 25.0],
        modelname=["gwt-1", "gwt-2"],
        species=["salinity", "temperature"],
        temperature_species_name="temperature",
        thermal_formulation="nonlinear",
    )
    vsc._update_transport_models(["gwt-1_0", "gwt-2_0"])
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    expected = textwrap.dedent(
        """\
    begin options
      viscref 0.0008904
      temperature_species_name temperature
      thermal_formulation nonlinear
      thermal_a2 10.0
      thermal_a3 248.37
      thermal_a4 133.15
    end options

    begin dimensions
      nviscspecies 2
    end dimensions

    begin packagedata
      1 1.92e-06 0.0 gwt-1_0 salinity
      2 0.0 25.0 gwt-2_0 temperature
    end packagedata"""
    )
    actual = vsc._render(directory, "vsc", globaltimes, False)

    assert actual == expected


def test_viscosity_package_update_transport_names_check():
    vsc = imod.mf6.Viscosity(
        reference_viscosity=8.904e-04,
        viscosity_concentration_slope=[1.92e-6, 0.0],
        reference_concentration=[0.0, 25.0],
        modelname=["gwt-2", "gwt-1"],
        species=["salinity", "temperature"],
        temperature_species_name="temperature",
        thermal_formulation="nonlinear",
    )
    # update the transport models, but in the wrong order
    with pytest.raises(ValueError):
        vsc._update_transport_models(["gwt-1_0", "gwt-2_0"])

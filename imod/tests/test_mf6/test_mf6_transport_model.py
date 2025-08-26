import textwrap

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely import linestrings

import imod
from imod.mf6.adv import AdvectionCentral
from imod.mf6.hfb import SingleLayerHorizontalFlowBarrierResistance
from imod.mf6.write_context import WriteContext


def test_long_package_name():
    m = imod.mf6.GroundwaterTransportModel()
    with pytest.raises(
        KeyError,
        match="Received key with more than 16 characters: 'my_very_long_package_name'Modflow 6 has a character limit of 16.",
    ):
        m["my_very_long_package_name"] = imod.mf6.AdvectionCentral()


def test_transport_model_rendering():
    adv = imod.mf6.AdvectionCentral()
    disp = imod.mf6.Dispersion(1e-4, 1.0, 10.0, 1.0, 2.0, 3.0, False, True)
    m = imod.mf6.GroundwaterTransportModel(print_input=True, save_flows=True)
    m["dsp"] = disp
    m["adv"] = adv
    write_context = WriteContext()
    actual = m._render("GWF_1", write_context)

    actual = m._render("transport", write_context)
    expected = textwrap.dedent(
        """\
      begin options
        print_input
        save_flows
      end options

      begin packages
        dsp6 transport/dsp.dsp dsp
        adv6 transport/adv.adv adv
      end packages
      """
    )
    assert actual == expected


def test_assign_flow_discretization(basic_dis, concentration_fc):
    # define a grid
    _, _, bottom = basic_dis
    idomain = xr.ones_like(
        concentration_fc.isel(species=0, time=0, drop=True), dtype=int
    )

    like = idomain.sel(layer=1).astype(np.float32)
    concentration = concentration_fc.sel(layer=1)

    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=200.0, bottom=bottom, idomain=idomain
    )

    gwf_model["riv-1"] = imod.mf6.River(
        stage=like,
        conductance=like,
        bottom_elevation=like - 2.0,
        concentration=concentration,
        concentration_boundary_type="AUX",
    )

    tpt_model = imod.mf6.GroundwaterTransportModel()
    tpt_model["ssm"] = imod.mf6.SourceSinkMixing.from_flow_model(gwf_model, "salinity")
    tpt_model["advection"] = AdvectionCentral()

    # let the transport model take the discretization from the flow model
    tpt_model["dis"] = gwf_model["dis"]

    # check that the discretization was added to the transport model
    assert len(tpt_model.keys()) == 3
    assert "dis" in tpt_model.keys()
    assert isinstance(tpt_model["dis"], imod.mf6.StructuredDiscretization)


def test_from_flow_model_with_hfb(basic_dis, concentration_fc):
    """
    Test that SourceSinkMixing can be created from a flow model with a HFB
    which has no concentration and should thus be ignored.
    """
    # define a grid
    _, _, bottom = basic_dis
    idomain = xr.ones_like(
        concentration_fc.isel(species=0, time=0, drop=True), dtype=int
    )

    like = idomain.sel(layer=1).astype(np.float32)
    concentration = concentration_fc.sel(layer=1)

    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=200.0, bottom=bottom, idomain=idomain
    )

    gwf_model["riv-1"] = imod.mf6.River(
        stage=like,
        conductance=like,
        bottom_elevation=like - 2.0,
        concentration=concentration,
        concentration_boundary_type="AUX",
    )

    barrier_y = [5.5, 5.5, 5.5]
    barrier_x = [82.0, 40.0, 0.0]

    geometry = gpd.GeoDataFrame(
        geometry=[linestrings(barrier_x, barrier_y)],
        data={
            "resistance": [1e3],
            "layer": [1],
        },
    )

    gwf_model["hfb-1"] = SingleLayerHorizontalFlowBarrierResistance(geometry)

    tpt_model = imod.mf6.GroundwaterTransportModel()
    tpt_model["ssm"] = imod.mf6.SourceSinkMixing.from_flow_model(gwf_model, "salinity")
    tpt_model["advection"] = AdvectionCentral()

    assert "riv-1" in tpt_model["ssm"]["package_names"]
    assert "hfb-1" not in tpt_model["ssm"]["package_names"]


def test_flowmodel_validation(twri_model):
    # initialize transport with a flow model without concentration data
    flow_model = twri_model["GWF_1"]
    with pytest.raises(ValueError):
        imod.mf6.SourceSinkMixing.from_flow_model(flow_model, "salinity")


def test_transport_concentration_loading(tmp_path, flow_transport_simulation):
    flow_transport_simulation.write(tmp_path)
    flow_transport_simulation.run()

    conc_notime = flow_transport_simulation.open_concentration(
        species_ls=["a", "b", "c", "d"]
    )
    assert conc_notime.coords["time"].dtype == float

    conc_time = flow_transport_simulation.open_concentration(
        species_ls=["a", "b", "c", "d"],
        simulation_start_time="2000-01-31",
        time_unit="s",
    )
    assert conc_time.coords["time"].dtype == np.dtype("datetime64[ns]")


def test_transport_balance_loading(tmp_path, flow_transport_simulation):
    flow_transport_simulation.write(tmp_path)
    flow_transport_simulation.run()

    balance_notime = flow_transport_simulation.open_transport_budget(
        species_ls=["a", "b", "c", "d"]
    )
    assert balance_notime.coords["time"].dtype == float

    balance_time = flow_transport_simulation.open_transport_budget(
        species_ls=["a", "b", "c", "d"],
        simulation_start_time="2000-01-31",
        time_unit="s",
    )
    assert balance_time.coords["time"].dtype == np.dtype("datetime64[ns]")

    np.testing.assert_allclose(
        balance_notime.sel(species="a")["source-sink mix_ssm"].values,
        balance_time.sel(species="a")["source-sink mix_ssm"].values,
        rtol=7e-5,
        atol=3e-3,
        equal_nan=True,
    )


def test_transport_output_wrong_species(tmp_path, flow_transport_simulation):
    flow_transport_simulation.write(tmp_path)
    flow_transport_simulation.run()

    with pytest.raises(ValueError):
        # Should be ["a", "b", "c", "d"]
        flow_transport_simulation.open_transport_budget(species_ls=["a", "b"])

    with pytest.raises(ValueError):
        # Should be ["a", "b", "c", "d"]
        flow_transport_simulation.open_concentration(species_ls=["a", "b"])


def test_transport_clip_box(tmp_path, flow_transport_simulation):
    x_min = 300.0
    flow_transport_simulation_clipped = flow_transport_simulation.clip_box(x_min=x_min)
    flow_transport_simulation_clipped.write(tmp_path)
    flow_transport_simulation_clipped.run()

    conc = flow_transport_simulation_clipped.open_concentration(
        species_ls=["a", "b", "c", "d"]
    )
    assert conc.coords["x"].min() > x_min


def test_transport_with_ats(tmp_path, flow_transport_simulation):
    """
    Test that transport model works with ATS and that timesteps are actually
    affected by the ATS settings.
    """
    # Arrange
    sim = flow_transport_simulation
    coords = {"time": [np.datetime64("2000-01-01")]}
    dims = ("time",)
    dt_init = xr.DataArray([1e-3], coords=coords, dims=dims)
    dt_min = xr.DataArray([1e-4], coords=coords, dims=dims)
    dt_max = xr.DataArray([10.0], coords=coords, dims=dims)
    dt_multiplier = xr.DataArray([1.2], coords=coords, dims=dims)
    dt_fail_multiplier = xr.DataArray([0.0], coords=coords, dims=dims)
    sim["ats"] = imod.mf6.AdaptiveTimeStepping(
        dt_init, dt_min, dt_max, dt_multiplier, dt_fail_multiplier
    )
    # Make sure iterations written to csv files to check amount of timesteps.
    sim["solver"]["outer_csvfile"] = "flow_outer.csv"
    sim["transport_solver"]["outer_csvfile"] = "tpt_outer.csv"
    # Act
    # Write with ATS
    path_with_ats = tmp_path / "with_ats"
    path_without_ats = tmp_path / "without_ats"
    sim.write(path_with_ats)
    sim.run()
    # Write without ATS
    sim.pop("ats")
    sim.write(path_without_ats)
    sim.run()
    with_ats_outer = pd.read_csv(path_with_ats / "tpt_outer.csv")
    without_ats_outer = pd.read_csv(path_without_ats / "tpt_outer.csv")

    # Assert
    assert len(with_ats_outer) > len(without_ats_outer)
    dt_init_with_actual = with_ats_outer.loc[0, "totim"]
    dt_init_without_actual = without_ats_outer.loc[0, "totim"]
    assert dt_init.item() == dt_init_with_actual
    assert dt_init.item() != dt_init_without_actual


def test_transport_with_ats_percel(tmp_path, flow_transport_simulation):
    """
    Test that transport model works with ATS and that timesteps are actually
    affected by the ATS percel setting in the advection package.
    """
    # Arrange
    sim = flow_transport_simulation
    coords = {"time": [np.datetime64("2000-01-01")]}
    dims = ("time",)
    dt_init = xr.DataArray([1e-3], coords=coords, dims=dims)
    dt_min = xr.DataArray([1e-4], coords=coords, dims=dims)
    dt_max = xr.DataArray([10.0], coords=coords, dims=dims)
    dt_multiplier = xr.DataArray([1.2], coords=coords, dims=dims)
    dt_fail_multiplier = xr.DataArray([0.0], coords=coords, dims=dims)
    sim["ats"] = imod.mf6.AdaptiveTimeStepping(
        dt_init, dt_min, dt_max, dt_multiplier, dt_fail_multiplier
    )
    # Make sure iterations written to csv files to check amount of timesteps.
    sim["solver"]["outer_csvfile"] = "flow_outer.csv"
    sim["transport_solver"]["outer_csvfile"] = "tpt_outer.csv"
    # Act
    # Write and run without percel
    path_with_percel = tmp_path / "with_percel"
    path_without_percel = tmp_path / "without_percel"
    sim.write(path_without_percel)
    sim.run()
    # Write and run with percel
    ats_percel = 0.1
    sim["tpt_a"]["adv"] = imod.mf6.AdvectionTVD(ats_percel=ats_percel)
    sim.write(path_with_percel)
    sim.run()
    with_percel_outer = pd.read_csv(path_with_percel / "tpt_outer.csv")
    without_percel_outer = pd.read_csv(path_without_percel / "tpt_outer.csv")
    # Assert
    assert with_percel_outer.size > without_percel_outer.size

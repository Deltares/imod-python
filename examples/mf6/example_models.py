"""
Example models
==============

This source file contains functions that create a simulation that can be used
in examples that are not focussed on building a simulation, but on doing
something with it ( such as regridding)

"""
import numpy as np
import scipy.ndimage
import xarray as xr

import imod
from imod.typing.grid import nan_like


def create_twri_simulation() -> imod.mf6.Modflow6Simulation:
    """There is a separate example contained in `TWRI
    <https://deltares.gitlab.io/imod/imod-python/examples/mf6/ex01_twri.html#sphx-glr-examples-mf6-ex01-twri-py>`_
    that you should look at if you are interested in the model building
    """

    nlay = 3
    nrow = 15
    ncol = 15
    shape = (nlay, nrow, ncol)

    dx = 5000.0
    dy = 5000.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = dy * nrow
    dims = ("layer", "y", "x")

    layer = np.array([1, 2, 3])
    y = np.arange(ymax, ymin, -dy) - 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x, "dx": dx, "dy": -dy}

    # %%
    # Create DataArrays
    # -----------------
    #
    # Now that we have the grid coordinates setup, we can start defining model
    # parameters. The model is characterized by:
    #
    # * a constant head boundary on the left
    # * a single drain in the center left of the model
    # * uniform recharge on the top layer
    # * a number of wells scattered throughout the model.

    idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)
    bottom = xr.DataArray([-200.0, -300.0, -450.0], {"layer": layer}, ("layer",))

    # Constant head
    constant_head = xr.full_like(idomain, np.nan, dtype=float).sel(layer=[1, 2])
    constant_head[..., 0] = 0.0

    # Drainage
    elevation = xr.full_like(idomain.sel(layer=1), np.nan, dtype=float)
    conductance = xr.full_like(idomain.sel(layer=1), np.nan, dtype=float)
    elevation[7, 1:10] = np.array([0.0, 0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0, 100.0])
    conductance[7, 1:10] = 1.0

    # Recharge
    rch_rate = xr.full_like(idomain.sel(layer=1), 3.0e-8, dtype=float)

    # Well
    screen_layer = [2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # we set the screen top and bottoms such that each well falls in one layer and is long enough not to be filtered out
    perforation_length = 50
    delta_z = 0.1

    screen_bottom = bottom[screen_layer] + delta_z
    screen_top = screen_bottom + delta_z + perforation_length

    # we compute the x and y cooordinates of the wells based on the row and column indices presented in the original twri model
    well_y = (
        ymax
        - np.array(
            [
                5.0,
                4.0,
                6.0,
                9.0,
                9.0,
                9.0,
                9.0,
                11.0,
                11.0,
                11.0,
                11.0,
                13.0,
                13.0,
                13.0,
                13.0,
            ]
        )
        * abs(dy)
        + dy / 2
    )
    well_x = (
        np.array(
            [
                11.0,
                6.0,
                12.0,
                8.0,
                10.0,
                12.0,
                14.0,
                8.0,
                10.0,
                12.0,
                14.0,
                8.0,
                10.0,
                12.0,
                14.0,
            ]
        )
        * dx
        - dx / 2
    )
    well_rate = [-5.0] * 15

    # Node properties
    icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    # %%
    # Build the model
    # ---------------
    #
    # The first step is to define an empty model, the parameters and boundary
    # conditions are added in the form of the familiar MODFLOW packages.

    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=200.0, bottom=bottom, idomain=idomain
    )
    gwf_model["chd"] = imod.mf6.ConstantHead(
        constant_head, print_input=True, print_flows=True, save_flows=True
    )
    gwf_model["drn"] = imod.mf6.Drainage(
        elevation=elevation,
        conductance=conductance,
        print_input=True,
        print_flows=True,
        save_flows=True,
    )
    gwf_model["ic"] = imod.mf6.InitialConditions(start=0.0)
    gwf_model["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=True,
        save_flows=True,
    )
    gwf_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=1.0e-5,
        specific_yield=0.15,
        transient=False,
        convertible=0,
    )
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
    gwf_model["rch"] = imod.mf6.Recharge(rch_rate)
    gwf_model["wel"] = imod.mf6.Well(
        x=well_x,
        y=well_y,
        screen_top=screen_top,
        screen_bottom=screen_bottom,
        rate=well_rate,
    )

    # %%
    # Attach it to a simulation
    # ---------------

    simulation = imod.mf6.Modflow6Simulation("ex01-twri")
    simulation["GWF_1"] = gwf_model
    # Define solver settings
    simulation["solver"] = imod.mf6.Solution(
        modelnames=["GWF_1"],
        print_option="summary",
        csv_output=False,
        no_ptc=True,
        outer_dvclose=1.0e-4,
        outer_maximum=500,
        under_relaxation=None,
        inner_dvclose=1.0e-4,
        inner_rclose=0.001,
        inner_maximum=100,
        linear_acceleration="cg",
        scaling_method=None,
        reordering_method=None,
        relaxation_factor=0.97,
    )
    # Collect time discretization
    simulation.create_time_discretization(
        additional_times=["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]
    )
    return simulation


def create_hondsrug_simulation() -> imod.mf6.Modflow6Simulation:
    """
    There is a separate example contained in `hondsrug
    <https://deltares.gitlab.io/imod/imod-python/examples/mf6/hondsrug.html#sphx-glr-examples-mf6-hondsrug-py>`_
    that you should look at if you are interested in the model building
    """

    # create model
    gwf_model = imod.mf6.GroundwaterFlowModel()

    # %%
    # This package allows specifying a regular MODFLOW grid. This grid is assumed
    # to be rectangular horizontally, but can be distorted vertically.
    #
    # Load data
    # ---------
    #
    # We'll load the data from the examples that come with this package.

    layermodel = imod.data.hondsrug_layermodel()

    # Make sure that the idomain is provided as integers
    idomain = layermodel["idomain"].astype(int)

    # We only need to provide the data for the top as a 2D array. Modflow 6 will
    # compare the top against the uppermost active bottom cell.
    top = layermodel["top"].max(dim="layer")

    bot = layermodel["bottom"]

    # %%
    # discretization package - DIS
    # ===========================
    #
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=top, bottom=bot, idomain=idomain
    )

    # %%
    # flow package - NPF
    # ===========================
    #

    k = layermodel["k"]

    gwf_model["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=0,
        k=k,
        k33=k,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=True,
        save_flows=True,
    )

    # %%
    # Initial conditions package - IC
    # ================================
    #

    initial = imod.data.hondsrug_initial()
    interpolated_head_larger = initial["head"]

    xmin = 237_500.0
    xmax = 250_000.0
    ymin = 559_000.0
    ymax = 564_000.0

    interpolated_head = interpolated_head_larger.sel(
        x=slice(xmin, xmax), y=slice(ymax, ymin)
    )

    starting_head = nan_like(idomain).combine_first(interpolated_head)
    # Consequently ensure no data is specified in inactive cells:
    starting_head = starting_head.where(idomain == 1)

    gwf_model["ic"] = imod.mf6.InitialConditions(starting_head)

    # %%
    # Constant head package - CHD
    # ===========================
    #

    def outer_edge(da):
        data = da.copy()
        from_edge = scipy.ndimage.binary_erosion(data)
        is_edge = (data == 1) & (from_edge == 0)
        return is_edge.astype(bool)

    like_2d = xr.full_like(idomain.isel(layer=0), 1)
    like_2d
    edge = outer_edge(xr.full_like(like_2d.drop_vars("layer"), 1))

    gwf_model["chd"] = imod.mf6.ConstantHead(
        starting_head.where((idomain > 0) & edge),
        print_input=False,
        print_flows=True,
        save_flows=True,
    )

    # %%
    #
    # Recharge
    # ========

    xmin = 230_000.0
    xmax = 257_000.0
    ymin = 550_000.0
    ymax = 567_000.0

    meteorology = imod.data.hondsrug_meteorology()
    pp = meteorology["precipitation"]
    evt = meteorology["evapotranspiration"]

    pp = pp.sel(x=slice(xmin, xmax), y=slice(ymax, ymin)) / 1000.0  # from mm/d to m/d
    evt = evt.sel(x=slice(xmin, xmax), y=slice(ymax, ymin)) / 1000.0  # from mm/d to m/d

    # %%
    # Recharge - Steady state
    # -----------------------

    pp_ss = pp.sel(time=slice("2000-01-01", "2009-12-31"))
    pp_ss_mean = pp_ss.mean(dim="time")

    # %%
    # **Evapotranspiration**
    evt_ss = evt.sel(time=slice("2000-01-01", "2009-12-31"))
    evt_ss_mean = evt_ss.mean(dim="time")

    # %%
    # For the recharge calculation, a first estimate
    # is the difference between the precipitation and evapotranspiration values.

    rch_ss = pp_ss_mean - evt_ss_mean

    # %%
    # Recharge - Transient
    # --------------------

    pp_trans = pp.sel(time=slice("2010-01-01", "2015-12-31"))
    evt_trans = evt.sel(time=slice("2010-01-01", "2015-12-31"))

    rch_trans = pp_trans - evt_trans
    rch_trans = rch_trans.where(rch_trans > 0, 0)  # check negative values

    rch_trans_yr = rch_trans.resample(time="A", label="left").mean()
    rch_trans_yr

    starttime = "2009-12-31"

    # Add first steady-state
    timedelta = np.timedelta64(1, "s")  # 1 second duration for initial steady-state
    starttime_steady = np.datetime64(starttime) - timedelta
    rch_ss = rch_ss.assign_coords(time=starttime_steady)

    rch_ss_trans = xr.concat([rch_ss, rch_trans_yr], dim="time")
    rch_ss_trans

    rch_ss_trans = imod.prepare.Regridder(method="mean").regrid(rch_ss_trans, like_2d)
    rch_ss_trans

    rch_total = rch_ss_trans.where(
        idomain["layer"] == idomain["layer"].where(idomain > 0).min("layer")
    )
    rch_total

    rch_total = rch_total.transpose("time", "layer", "y", "x")
    rch_total

    gwf_model["rch"] = imod.mf6.Recharge(rch_total)

    # %%
    # Drainage package - DRN
    # =======================

    drainage = imod.data.hondsrug_drainage()

    pipe_cond = drainage["conductance"]
    pipe_elev = drainage["elevation"]

    pipe_cond

    gwf_model["drn-pipe"] = imod.mf6.Drainage(
        conductance=pipe_cond, elevation=pipe_elev
    )

    # %%
    # River package - RIV
    # ===================

    river = imod.data.hondsrug_river()
    riv_cond = river["conductance"]
    riv_stage = river["stage"]
    riv_bot = river["bottom"]

    gwf_model["riv"] = imod.mf6.River(
        conductance=riv_cond, stage=riv_stage, bottom_elevation=riv_bot
    )

    # %%
    # Storage package - STO
    # ======================

    ss = 0.0003
    layer = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    sy = xr.DataArray(
        [0.16, 0.16, 0.16, 0.16, 0.15, 0.15, 0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14],
        {"layer": layer},
        ("layer",),
    )
    times_sto = np.array(
        [
            "2009-12-30T23:59:59.00",
            "2009-12-31T00:00:00.00",
            "2010-12-31T00:00:00.00",
            "2011-12-31T00:00:00.00",
            "2012-12-31T00:00:00.00",
            "2013-12-31T00:00:00.00",
            "2014-12-31T00:00:00.00",
        ],
        dtype="datetime64[ns]",
    )

    transient = xr.DataArray(
        [False, True, True, True, True, True, True], {"time": times_sto}, ("time",)
    )

    gwf_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=ss,
        specific_yield=sy,
        transient=transient,
        convertible=0,
        save_flows=True,
    )

    # %%
    # Output Control package - OC
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="last", save_budget="last")

    # %%
    # Model simulation
    # ================

    simulation = imod.mf6.Modflow6Simulation("mf6-mipwa2-example")
    simulation["GWF_1"] = gwf_model

    # %%
    # Solver settings
    # ---------------

    simulation["solver"] = imod.mf6.Solution(
        modelnames=["GWF_1"],
        print_option="summary",
        csv_output=False,
        no_ptc=True,
        outer_dvclose=1.0e-4,
        outer_maximum=500,
        under_relaxation=None,
        inner_dvclose=1.0e-4,
        inner_rclose=0.001,
        inner_maximum=100,
        linear_acceleration="cg",
        scaling_method=None,
        reordering_method=None,
        relaxation_factor=0.97,
    )

    # %%
    # Assign time discretization
    # --------------------------

    simulation.create_time_discretization(
        additional_times=[
            "2009-12-30T23:59:59.000000000",
            "2015-12-31T00:00:00.000000000",
        ]
    )

    return simulation

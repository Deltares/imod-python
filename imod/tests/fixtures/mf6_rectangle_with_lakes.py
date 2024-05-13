import numpy as np
import pytest
import xarray as xr

import imod
from imod.mf6.lak import Lake, LakeData


def create_lake(idomain, xmin_index, xmax_index, ymin_index, ymax_index, name):
    is_lake = xr.full_like(idomain, False, dtype=bool)
    is_lake.values[0, xmin_index : xmax_index + 1, ymin_index : ymax_index + 1] = True

    lake_table = None
    return create_lake_data_structured(
        is_lake, starting_stage=11.0, name=name, lake_table=lake_table
    )


@pytest.fixture(scope="function")
def rectangle_with_lakes():
    shape = nlay, nrow, ncol = 2, 30, 30

    dx = 100.0
    dy = -100.0

    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")
    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}
    bottom = xr.DataArray([-200.0, -300.0], {"layer": layer}, ("layer",))

    # Discretization data
    icelltype = xr.DataArray([1, 0], {"layer": layer}, ("layer",))
    like = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
    idomain = like.astype(np.int8)
    k = xr.DataArray([1.0e-3, 1.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-4, 1.0e-4], {"layer": layer}, ("layer",))

    # Constant head
    head = xr.full_like(like, np.nan).sel(layer=[1, 2])
    head[..., 0] = 0.0

    # Create and fill the groundwater model.
    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=0.0, bottom=bottom, idomain=idomain
    )
    gwf_model["chd"] = imod.mf6.ConstantHead(
        head, print_input=True, print_flows=True, save_flows=True
    )
    gwf_model["chd"] = imod.mf6.SpecificStorage(
        specific_storage=1.0e-5,
        specific_yield=0.15,
        convertible=0,
        transient=True,
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
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")

    simulation = imod.mf6.Modflow6Simulation("ex01-twri")
    simulation["GWF_1"] = gwf_model
    # Define solver settings
    simulation["solver"] = imod.mf6.Solution(
        modelnames=["GWF_1"],
        print_option="summary",
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
    simulation.create_time_discretization(additional_times=["2000-01-01", "2000-01-02"])
    lake1 = create_lake(idomain, 2, 3, 2, 3, "first_lake")
    lake2 = create_lake(idomain, 12, 13, 12, 13, "second_lake")
    simulation["GWF_1"]["lake"] = Lake.from_lakes_and_outlets([lake1, lake2])
    return simulation


def create_lake_data_structured(
    is_lake,
    starting_stage,
    name,
    status=None,
    stage=None,
    rainfall=None,
    evaporation=None,
    runoff=None,
    inflow=None,
    withdrawal=None,
    auxiliary=None,
    lake_table=None,
):
    HORIZONTAL = 0
    connection_type = xr.full_like(is_lake, HORIZONTAL, dtype=np.float64).where(is_lake)
    bed_leak = xr.full_like(is_lake, 0.2, dtype=np.float64).where(is_lake)
    top_elevation = xr.full_like(is_lake, 0.0, dtype=np.float64).where(is_lake)
    bot_elevation = xr.full_like(is_lake, -1.0, dtype=np.float64).where(is_lake)
    connection_length = xr.full_like(is_lake, 0.5, dtype=np.float64).where(is_lake)
    connection_width = xr.full_like(is_lake, 0.6, dtype=np.float64).where(is_lake)
    return LakeData(
        starting_stage=starting_stage,
        boundname=name,
        connection_type=connection_type,
        bed_leak=bed_leak,
        top_elevation=top_elevation,
        bot_elevation=bot_elevation,
        connection_length=connection_length,
        connection_width=connection_width,
        status=status,
        stage=stage,
        rainfall=rainfall,
        evaporation=evaporation,
        runoff=runoff,
        inflow=inflow,
        withdrawal=withdrawal,
        auxiliary=auxiliary,
        lake_table=lake_table,
    )

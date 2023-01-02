import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy import nan

from imod import mf6, msw


def make_coupled_mf6_model():
    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
    dz = np.array([1, 10, 100])
    layer = [1, 2, 3]
    dx = 1.0
    dy = -1.0

    nlay = len(layer)
    nrow = len(y)
    ncol = len(x)

    freq = "D"
    times = pd.date_range(start="1/1/1971", end="1/3/1971", freq=freq)

    like = xr.DataArray(
        data=np.ones((nlay, nrow, ncol)),
        dims=("layer", "y", "x"),
        coords={"layer": layer, "y": y, "x": x, "dx": dx, "dy": dy},
    )
    idomain = like.astype(np.int32)

    top = 0.0
    bottom = top - xr.DataArray(
        np.cumsum(layer * dz), coords={"layer": layer}, dims="layer"
    )

    head = xr.full_like(like, np.nan)
    head[..., 0] = -1.0
    head[..., -1] = -1.0

    head = head.expand_dims(time=times)

    gwf_model = mf6.GroundwaterFlowModel()
    gwf_model["dis"] = mf6.StructuredDiscretization(
        idomain=idomain, top=top, bottom=bottom
    )
    gwf_model["chd"] = mf6.ConstantHead(
        head, print_input=True, print_flows=True, save_flows=True
    )

    icelltype = xr.full_like(bottom, 0, dtype=int)
    k = xr.DataArray(np.ones((nlay)), {"layer": layer}, ("layer",))
    k33 = xr.DataArray(np.ones((nlay)), {"layer": layer}, ("layer",))
    gwf_model["npf"] = mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        variable_vertical_conductance=True,
        dewatered=False,
        perched=False,
        save_flows=True,
    )

    gwf_model["ic"] = mf6.InitialConditions(head=0.5)
    gwf_model["sto"] = mf6.SpecificStorage(1e-3, 0.1, True, 0)

    recharge = xr.zeros_like(idomain.sel(layer=1), dtype=float)
    recharge[:, 0] = np.nan
    recharge[:, -1] = np.nan

    gwf_model["rch_msw"] = mf6.Recharge(recharge)

    gwf_model["oc"] = mf6.OutputControl(save_head="last", save_budget="last")

    # Create wells
    wel_layer = 3

    ix = np.tile(np.arange(ncol) + 1, nrow)
    iy = np.repeat(np.arange(nrow) + 1, ncol)
    rate = np.zeros(ix.shape)
    layer = np.full_like(ix, wel_layer)

    gwf_model["wells_msw"] = mf6.WellDisStructured(
        layer=layer, row=iy, column=ix, rate=rate
    )

    # Attach it to a simulation
    simulation = mf6.Modflow6Simulation("test")
    simulation["GWF_1"] = gwf_model
    # Define solver settings
    simulation["solver"] = mf6.Solution(
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
    simulation.create_time_discretization(additional_times=times)

    return simulation


def make_msw_model():
    unsaturated_database = "./unsat_database"

    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
    subunit = [0, 1]
    dx = 1.0
    dy = -1.0

    nrow = len(y)
    ncol = len(x)

    freq = "D"
    times = pd.date_range(start="1/1/1971", end="1/3/1971", freq=freq)

    # fmt: off
    area = xr.DataArray(
        np.array(
            [
                [[0.5, 0.5, 0.5],
                 [nan, nan, nan],
                 [1.0, 1.0, 1.0]],

                [[0.5, 0.5, 0.5],
                 [1.0, 1.0, 1.0],
                 [nan, nan, nan]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )

    active = xr.DataArray(
        np.array(
            [[False, True, False],
             [False, True, False],
             [False, True, False]]),
        dims=("y", "x"),
        coords={"y": y, "x": x}
    )
    # fmt: on
    msw_grid = xr.ones_like(active, dtype=float)

    precipitation = msw_grid.expand_dims(time=times[:-1])
    evapotranspiration = msw_grid.expand_dims(time=times[:-1]) * 1.2

    # %% Vegetation
    day_of_year = np.arange(1, 367)
    vegetation_index = np.arange(1, 4)

    coords = {"day_of_year": day_of_year, "vegetation_index": vegetation_index}

    soil_cover = xr.DataArray(
        data=np.zeros(day_of_year.shape + vegetation_index.shape),
        coords=coords,
        dims=("day_of_year", "vegetation_index"),
    )
    soil_cover[132:254, :] = 1.0
    leaf_area_index = soil_cover * 3

    vegetation_factor = xr.zeros_like(soil_cover)
    vegetation_factor[132:142, :] = 0.7
    vegetation_factor[142:152, :] = 0.9
    vegetation_factor[152:162, :] = 1.0
    vegetation_factor[162:192, :] = 1.2
    vegetation_factor[192:244, :] = 1.1
    vegetation_factor[244:254, :] = 0.7

    # %% Landuse
    landuse_index = np.arange(1, 4)
    names = ["grassland", "maize", "potatoes"]

    coords = {"landuse_index": landuse_index}

    landuse_names = xr.DataArray(data=names, coords=coords, dims=("landuse_index",))
    vegetation_index_da = xr.DataArray(
        data=vegetation_index, coords=coords, dims=("landuse_index",)
    )
    lu = xr.ones_like(vegetation_index_da, dtype=float)

    # %% Well

    wel_layer = 3

    ix = np.tile(np.arange(ncol) + 1, nrow)
    iy = np.repeat(np.arange(nrow) + 1, ncol)
    rate = np.zeros(ix.shape)
    layer = np.full_like(ix, wel_layer)

    well = mf6.WellDisStructured(layer=layer, row=iy, column=ix, rate=rate)

    # %% Modflow 6
    dz = np.array([1, 10, 100])
    layer = [1, 2, 3]

    top = 0.0
    bottom = top - xr.DataArray(
        np.cumsum(layer * dz), coords={"layer": layer}, dims="layer"
    )

    idomain = xr.full_like(msw_grid, 1, dtype=int).expand_dims(layer=layer)
    dis = mf6.StructuredDiscretization(top=top, bottom=bottom, idomain=idomain)

    # %% Initiate model
    msw_model = msw.MetaSwapModel(unsaturated_database=unsaturated_database)
    msw_model["grid"] = msw.GridData(
        area,
        xr.full_like(area, 1, dtype=int),
        xr.full_like(area, 1.0, dtype=float),
        xr.full_like(active, 1.0, dtype=float),
        xr.full_like(active, 1, dtype=int),
        active,
    )

    msw_model["ic"] = msw.InitialConditionsRootzonePressureHead(initial_pF=2.2)

    # %% Meteo
    msw_model["meteo_grid"] = msw.MeteoGrid(precipitation, evapotranspiration)
    msw_model["mapping_prec"] = msw.PrecipitationMapping(precipitation)
    msw_model["mapping_evt"] = msw.EvapotranspirationMapping(precipitation * 1.5)

    # %% Sprinkling
    msw_model["sprinkling"] = msw.Sprinkling(
        max_abstraction_groundwater=xr.full_like(msw_grid, 100.0),
        max_abstraction_surfacewater=xr.full_like(msw_grid, 100.0),
        well=well,
    )

    # %% Ponding
    msw_model["ponding"] = msw.Ponding(
        ponding_depth=xr.full_like(area, 0.0),
        runon_resistance=xr.full_like(area, 1.0),
        runoff_resistance=xr.full_like(area, 1.0),
    )

    # %% Scaling Factors
    msw_model["scaling"] = msw.ScalingFactors(
        scale_soil_moisture=xr.full_like(area, 1.0),
        scale_hydraulic_conductivity=xr.full_like(area, 1.0),
        scale_pressure_head=xr.full_like(area, 1.0),
        depth_perched_water_table=xr.full_like(msw_grid, 1.0),
    )

    # %% Infiltration
    msw_model["infiltration"] = msw.Infiltration(
        infiltration_capacity=xr.full_like(area, 1.0),
        downward_resistance=xr.full_like(msw_grid, -9999.0),
        upward_resistance=xr.full_like(msw_grid, -9999.0),
        bottom_resistance=xr.full_like(msw_grid, -9999.0),
        extra_storage_coefficient=xr.full_like(msw_grid, 0.1),
    )

    # %% Vegetation
    msw_model["crop_factors"] = msw.AnnualCropFactors(
        soil_cover=soil_cover,
        leaf_area_index=leaf_area_index,
        interception_capacity=xr.zeros_like(soil_cover),
        vegetation_factor=vegetation_factor,
        interception_factor=xr.ones_like(soil_cover),
        bare_soil_factor=xr.ones_like(soil_cover),
        ponding_factor=xr.ones_like(soil_cover),
    )

    # %% Landuse options
    msw_model["landuse_options"] = msw.LanduseOptions(
        landuse_name=landuse_names,
        vegetation_index=vegetation_index_da,
        jarvis_o2_stress=xr.ones_like(lu),
        jarvis_drought_stress=xr.ones_like(lu),
        feddes_p1=xr.full_like(lu, 99.0),
        feddes_p2=xr.full_like(lu, 99.0),
        feddes_p3h=lu * [-2.0, -4.0, -3.0],
        feddes_p3l=lu * [-8.0, -5.0, -5.0],
        feddes_p4=lu * [-80.0, -100.0, -100.0],
        feddes_t3h=xr.full_like(lu, 5.0),
        feddes_t3l=xr.full_like(lu, 1.0),
        threshold_sprinkling=lu * [-8.0, -5.0, -5.0],
        fraction_evaporated_sprinkling=xr.full_like(lu, 0.05),
        gift=xr.full_like(lu, 20.0),
        gift_duration=xr.full_like(lu, 0.25),
        rotational_period=lu * [10, 7, 7],
        start_sprinkling_season=lu * [120, 180, 150],
        end_sprinkling_season=lu * [230, 230, 240],
        interception_option=xr.ones_like(lu, dtype=int),
        interception_capacity_per_LAI=xr.zeros_like(lu),
        interception_intercept=xr.ones_like(lu),
    )

    # %% Metaswap Mappings
    msw_model["mod2svat"] = msw.CouplerMapping(modflow_dis=dis, well=well)

    # %% Output Control
    msw_model["oc_idf"] = msw.IdfMapping(area, -9999.0)
    msw_model["oc_var"] = msw.VariableOutputControl()
    msw_model["oc_time"] = msw.TimeOutputControl(time=times)

    return msw_model


@pytest.fixture(scope="function")
def coupled_mf6_model():
    return make_coupled_mf6_model()


@pytest.fixture(scope="function")
def msw_model():
    return make_msw_model()

from imod import mf6
import pytest_cases
import xarray as xr
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from numpy import float64, int_

def grid_sizes() -> (
    tuple[
        list[float],
        list[float],
        NDArray[int_],
        float,
        float,
        NDArray[float64],
    ]
):
    x = [100.0, 200.0, 300.0, 400.0, 500.0]
    y = [300.0, 200.0, 100.0]
    dz = np.array([0.2, 10.0, 100.0])

    layer = np.arange(len(dz)) + 1
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    return x, y, layer, dx, dy, dz

@pytest_cases.fixture(scope="function")
def coupled_mf6_model_storage_coefficient(
    active_idomain: xr.DataArray,
) -> mf6.Modflow6Simulation:
    coupled_mf6_model = make_coupled_mf6_model(active_idomain)

    gwf_model = coupled_mf6_model["GWF_1"]
    gwf_model = convert_storage_package(gwf_model)
    # reassign gwf model
    coupled_mf6_model["GWF_1"] = gwf_model

    return coupled_mf6_model

def make_coupled_mf6_model(idomain: xr.DataArray) -> mf6.Modflow6Simulation:
    _, nrow, ncol = idomain.shape
    gwf_model = make_mf6_model(idomain)
    times = get_times()
    head = xr.full_like(idomain.astype(np.float64), np.nan)
    head[0, :, 0] = -2.0
    head = head.expand_dims(time=times)
    gwf_model["chd"] = mf6.ConstantHead(
        head, print_input=True, print_flows=True, save_flows=True
    )
    gwf_model["rch_msw"] = make_recharge_pkg(idomain)
    gwf_model["wells_msw"] = create_wells(nrow, ncol, idomain)

    simulation = make_mf6_simulation(gwf_model)
    return simulation

def make_mf6_model(idomain: xr.DataArray) -> mf6.GroundwaterFlowModel:
    _, _, layer, _, _, dz = grid_sizes()
    nlay = len(layer)

    top = 0.0
    bottom = top - xr.DataArray(np.cumsum(dz), coords={"layer": layer}, dims="layer")

    gwf_model = mf6.GroundwaterFlowModel()
    gwf_model["dis"] = mf6.StructuredDiscretization(
        idomain=idomain, top=top, bottom=bottom
    )

    icelltype = xr.full_like(bottom, 0, dtype=int)
    k_values = np.ones(nlay)
    k_values[1, ...] = 0.01

    k = xr.DataArray(k_values, {"layer": layer}, ("layer",))
    k33 = xr.DataArray(k_values / 10.0, {"layer": layer}, ("layer",))
    gwf_model["npf"] = mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        variable_vertical_conductance=True,
        dewatered=False,
        perched=False,
        save_flows=True,
    )

    gwf_model["ic"] = mf6.InitialConditions(start=-2.0)
    gwf_model["sto"] = mf6.SpecificStorage(1e-3, 0.1, True, 0)
    gwf_model["oc"] = mf6.OutputControl(save_head="last", save_budget="last")
    return gwf_model

def convert_storage_package(
    gwf_model: mf6.GroundwaterFlowModel,
) -> mf6.GroundwaterFlowModel:
    """
    Convert existing groundwater flow model with a specific storage to a model
    with a storage coefficient.
    """
    # Specific storage package
    sto_ds = gwf_model.pop("sto").dataset

    # Confined: S = Ss * b
    # Where 'S' is storage coefficient, 'Ss' specific
    # storage, and 'b' thickness.
    # https://en.wikipedia.org/wiki/Specific_storage

    dis_ds = gwf_model["dis"].dataset
    top = dis_ds["bottom"].shift(layer=1)
    top[0] = dis_ds["top"]
    b = top - dis_ds["bottom"]

    sto_ds["storage_coefficient"] = sto_ds["specific_storage"] * b
    sto_ds = sto_ds.drop_vars("specific_storage")

    gwf_model["sto"] = mf6.StorageCoefficient(**sto_ds)
    return gwf_model

def make_recharge_pkg(idomain: xr.DataArray) -> mf6.Recharge:
    idomain_l1 = idomain.sel(layer=1)
    recharge = xr.zeros_like(idomain_l1, dtype=float)
    # Deactivate cells where coupled MetaSWAP model is inactive as well.
    recharge[:, 0] = np.nan
    recharge = recharge.where(idomain_l1)

    return mf6.Recharge(recharge)

def create_wells(
    nrow: int, ncol: int, idomain: xr.DataArray, wel_layer: int | None = None
) -> mf6.WellDisStructured:
    """
    Create wells, deactivate inactive cells. This function wouldn't be necessary
    if iMOD Python had a package to specify wells based on grids.
    """

    if wel_layer is None:
        wel_layer = 3

    is_inactive = ~idomain.sel(layer=wel_layer).astype(bool)
    id_inactive = np.argwhere(is_inactive.values) + 1

    ix = np.tile(np.arange(ncol) + 1, nrow)
    iy = np.repeat(np.arange(nrow) + 1, ncol)

    to_deactivate = np.full_like(ix, False, dtype=bool)
    for i in id_inactive:
        is_cell = (iy == i[0]) & (ix == i[1])
        to_deactivate = to_deactivate | is_cell

    ix_active = ix[~to_deactivate]
    iy_active = iy[~to_deactivate]

    rate = np.zeros(ix_active.shape)
    layer = np.full_like(ix_active, wel_layer)

    return mf6.WellDisStructured(
        layer=layer, row=iy_active, column=ix_active, rate=rate
    )


def make_mf6_simulation(gwf_model: mf6.GroundwaterFlowModel) -> mf6.Modflow6Simulation:
    times = get_times()
    simulation = mf6.Modflow6Simulation("test")
    simulation["GWF_1"] = gwf_model
    simulation["solver"] = mf6.Solution(
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
    simulation.create_time_discretization(additional_times=times)
    return simulation

def get_times() -> pd.DatetimeIndex:
    freq = "D"
    return pd.date_range(start="1/1/1971", end="8/1/1971", freq=freq)
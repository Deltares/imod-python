from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy import float_, int_
from numpy.typing import NDArray

from imod import mf6


def grid_sizes() -> (
    Tuple[
        List[float],
        List[float],
        NDArray[int_],
        float,
        float,
        NDArray[float_],
    ]
):
    x = [100.0, 200.0, 300.0, 400.0, 500.0]
    y = [300.0, 200.0, 100.0]
    dz = np.array([0.2, 10.0, 100.0])

    layer = np.arange(len(dz)) + 1
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    return x, y, layer, dx, dy, dz


def get_times() -> pd.DatetimeIndex:
    freq = "D"
    return pd.date_range(start="1/1/1971", end="8/1/1971", freq=freq)


def create_wells(nrow: int, ncol: int, idomain: xr.DataArray) -> mf6.WellDisStructured:
    """
    Create wells, deactivate inactive cells. This function wouldn't be necessary
    if iMOD Python had a package to specify wells based on grids.
    """

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


def make_mf6_model(idomain: xr.DataArray) -> mf6.GroundwaterFlowModel:
    times = get_times()
    _, _, layer, _, _, dz = grid_sizes()
    nlay = len(layer)

    top = 0.0
    bottom = top - xr.DataArray(np.cumsum(dz), coords={"layer": layer}, dims="layer")

    head = xr.full_like(idomain.astype(np.float64), np.nan)
    head[0, :, 0] = -2.0

    head = head.expand_dims(time=times)

    gwf_model = mf6.GroundwaterFlowModel()
    gwf_model["dis"] = mf6.StructuredDiscretization(
        idomain=idomain, top=top, bottom=bottom
    )
    gwf_model["chd"] = mf6.ConstantHead(
        head, print_input=True, print_flows=True, save_flows=True
    )

    icelltype = xr.full_like(bottom, 0, dtype=int)
    k_values = np.ones((nlay))
    k_values[1, ...] = 0.001

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


def make_mf6_simulation(gwf_model):
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


def make_coupled_ribasim_mf6_model(idomain: xr.DataArray):
    # The bottom of the ribasim trivial model is located at 0.0 m: the surface
    # level of the groundwater model.
    gwf_model = make_mf6_model(idomain)

    template = xr.full_like(idomain.isel(layer=[0]), np.nan, dtype=np.float64)
    stage = template.copy()
    conductance = template.copy()
    bottom_elevation = template.copy()

    # Conductance is area divided by resistance (dx * dy / c0)
    # Assume the entire cell is wetted.
    stage[:, 1, 3] = 0.5
    conductance[:, 1, 3] = (100.0 * 100.0) / 1.0
    bottom_elevation[:, 1, 3] = 0.0

    gwf_model["riv-1"] = mf6.River(
        stage=stage,
        conductance=conductance,
        bottom_elevation=bottom_elevation,
    )

    # The k-value is only 0.001, so we'll use an appropriately low recharge value...
    rate = xr.full_like(template, 1.0e-5)
    rate[:, 1, 3] = np.nan
    gwf_model["rch"] = mf6.Recharge(rate=rate)

    simulation = make_mf6_simulation(gwf_model)
    return simulation


def make_idomain() -> xr.DataArray:
    x, y, layer, dx, dy, _ = grid_sizes()

    nlay = len(layer)
    nrow = len(y)
    ncol = len(x)

    return xr.DataArray(
        data=np.ones((nlay, nrow, ncol), dtype=np.int32),
        dims=("layer", "y", "x"),
        coords={"layer": layer, "y": y, "x": x, "dx": dx, "dy": dy},
    )


@pytest.fixture(scope="function")
def ribasim_model():
    import ribasim_testmodels

    return ribasim_testmodels.trivial_model()


@pytest.fixture(scope="function")
def coupled_ribasim_mf6_model():
    idomain = make_idomain()

    mf6_sim = make_coupled_ribasim_mf6_model(idomain)
    return mf6_sim

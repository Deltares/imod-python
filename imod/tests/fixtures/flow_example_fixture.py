import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod.flow as flow


def create_imodflow_model():
    shape = nlay, nrow, ncol = 3, 9, 9

    dx = 100.0
    dy = -100.0
    dz = np.array([5, 30, 100])
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")
    kh = 10.0
    kva = 1.0
    sto = 0.001

    # Some coordinates
    times = pd.date_range(start="1/1/2018", end="12/1/2018", freq="MS")
    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    # top & bottom
    surface = 0.0
    interfaces = np.insert((surface - np.cumsum(dz)), 0, surface)
    bottom = xr.DataArray(interfaces[1:], coords={"layer": layer}, dims="layer")
    top = xr.DataArray(interfaces[:-1], coords={"layer": layer}, dims="layer")

    # constant head & ibound
    ibound = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
    starting_head = ibound.copy()
    trend = np.ones(times[:-1].shape)
    trend = np.cumsum(trend)
    trend_da = xr.DataArray(trend, coords={"time": times[:-1]}, dims=["time"])
    is_x_edge = starting_head.x.isin([x[0], x[-1]])
    head_edge = starting_head.where(is_x_edge)
    head_edge_rising = trend_da * head_edge
    ibound = ibound.where(head_edge.isnull(), other=-2)

    # general head boundary
    is_x_central = starting_head.x == x[4]
    head_central = starting_head.where(is_x_central).sel(layer=1)
    period_times = times[[0, 6]] - np.timedelta64(365, "D")
    periods_da = xr.DataArray([10, 4], coords={"time": period_times}, dims=["time"])
    head_periodic = periods_da * head_central
    timemap = {
        period_times[0]: "winter",
        period_times[1]: "summer",
    }

    # Wells
    wel_df = pd.DataFrame()
    wel_df["id_name"] = np.arange(len(x)).astype(str)
    wel_df["x"] = x
    wel_df["y"] = y
    wel_df["rate"] = dx * dy * -1 * 0.5
    wel_df["time"] = np.tile(times[:-1], 2)[: len(x)]
    wel_df["layer"] = 2

    # Fill the model
    m = flow.ImodflowModel("imodflow")
    m["pcg"] = flow.PreconditionedConjugateGradientSolver()
    m["bnd"] = flow.Boundary(ibound)
    m["top"] = flow.Top(top)
    m["bottom"] = flow.Bottom(bottom)
    m["khv"] = flow.HorizontalHydraulicConductivity(kh)
    m["kva"] = flow.VerticalAnisotropy(kva)
    m["sto"] = flow.StorageCoefficient(sto)
    m["shd"] = flow.StartingHead(starting_head)
    m["chd"] = flow.ConstantHead(head=head_edge_rising)
    m["ghb"] = flow.GeneralHeadBoundary(head=head_periodic, conductance=10.0)
    m["ghb"].periodic_stress(timemap)
    m["ghb2"] = flow.GeneralHeadBoundary(head=head_periodic + 10.0, conductance=1.0)
    m["ghb2"].periodic_stress(timemap)
    m["wel"] = flow.Well(**wel_df)
    m["oc"] = flow.OutputControl(save_head=-1, save_flux=-1)
    m.create_time_discretization(additional_times=times[-1])
    return m


@pytest.fixture(scope="session")
def imodflow_model():
    return create_imodflow_model()

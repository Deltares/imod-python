import numpy as np
import pandas as pd
import xarray as xr

import imod

# Discretization
nrow = 1
ncol = 100
nlay = 50

dz = 1.0
dx = np.array([2.0] * 25 + [1.0] * 50 + [2.0] * 25)
dy = -1.0
x = np.full(ncol, 0.0) + dx.cumsum() - 0.5 * dx

# scale parameters with discretization
qscaled = 0.03 * (dz * abs(dy))

# Fresh water injection with well
# Add the arguments as a list, so pandas doesn't complain about having to set
# an index.
weldata = pd.DataFrame()
weldata["x"] = [0.5]
weldata["y"] = [0.5]
weldata["q"] = [qscaled]

# Setup ibound
bnd = xr.DataArray(
    data=np.full((nlay, nrow, ncol), 1.0),
    coords={
        "y": [0.5],
        "x": x,
        "layer": np.arange(1, 1 + nlay),
        "dx": ("x", dx),
        "dy": dy,
    },
    dims=("layer", "y", "x"),
)

top1D = xr.DataArray(
    np.arange(nlay * dz, 0.0, -dz), {"layer": np.arange(1, 1 + nlay)}, ("layer")
)
bot = top1D - 1.0
# We define constant head here, after generating the tops, or we'd end up with negative top values
bnd[:, :, -1] = -1

# daterange = pd.date_range("2000-01-01", "2001-01-01", freq="MS")
stage = xr.full_like(bnd, 1.0)
# stage = xr.concat([stage.assign_coords(time=t) for t in daterange], dim="time")

# Fill model
m = imod.wq.SeawatModel("HenryCase")
m["bas"] = imod.wq.BasicFlow(ibound=bnd, top=50.0, bottom=bot, starting_head=1.0)
m["lpf"] = imod.wq.LayerPropertyFlow(
    k_horizontal=10.0, k_vertical=10.0, specific_storage=0.0
)
m["btn"] = imod.wq.BasicTransport(
    icbund=bnd, starting_concentration=35.0, porosity=0.35
)
# m["adv"] = imod.wq.AdvectionTVD(courant=1.0)
m["adv"] = imod.wq.AdvectionMOC(
    courant=0.75,
    tracking="hybrid",
    weighting_factor=0.5,
    dconcentration_epsilon=1.0e-5,
    nplane=2,
    nparticles_no_advection=10,
    nparticles_advection=40,
    cell_min_nparticles=5,
    cell_max_nparticles=80,
)

m["dsp"] = imod.wq.Dispersion(longitudinal=0.1, diffusion_coefficient=1.0e-9)
m["vdf"] = imod.wq.VariableDensityFlow(density_concentration_slope=0.71)
m["wel"] = imod.wq.Well(
    id_name="well", x=weldata["x"], y=weldata["y"], rate=weldata["q"]
)
m["riv"] = imod.wq.River(
    stage=stage,
    conductance=xr.full_like(bnd, 1.0),
    bottom_elevation=xr.full_like(bnd, 0.0),
    density=xr.full_like(bnd, 1000.0),
)
m["riv2"] = imod.wq.River(
    stage=stage,
    conductance=xr.full_like(bnd, 1.0),
    bottom_elevation=xr.full_like(bnd, 0.0),
    density=xr.full_like(bnd, 1000.0),
)
m["pksf"] = imod.wq.ParallelKrylovFlowSolver(
    max_iter=150,
    inner_iter=100,
    hclose=0.001,
    rclose=500.0,
    relax=0.98,
    partition="uniform",
)
m["pkst"] = imod.wq.ParallelKrylovTransportSolver(
    max_iter=150, inner_iter=50, cclose=1.0e-6, partition="uniform"
)
m["oc"] = imod.wq.OutputControl(save_head_idf=True, save_concentration_idf=True)
m.time_discretization(times=pd.date_range("2000-01-01", "2001-01-01", freq="M"))
m.write("HenryCase")

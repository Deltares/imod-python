import imod
import numpy as np
import pandas as pd
import xarray as xr

# Discretization
nrow = 1  # number of rows
ncol = 45  # number of columns
nlay = 76  # number of layers

dz = 4.0
dx = 20.0
dy = -dx

# setup ibound
bnd = xr.DataArray(
    data=np.full((nlay, nrow, ncol), 1.0),
    coords={
        "y": [0.5],
        "x": np.arange(0.5 * dx, dx * ncol, dx),
        "layer": np.arange(1, 1 + nlay),
        "dx": dx,
        "dy": dy,
    },
    dims=("layer", "y", "x"),
)

bnd.plot()

# set up constant heads
bnd[0, :, :] = -1
bnd[75, :, 0:15] = 0.0
bnd[75, :, 30:45] = 0.0
bnd.plot(y="layer", yincrease=False)

# setup tops and bottoms
top1D = xr.DataArray(
    np.arange(nlay * dz, 0.0, -dz), {"layer": np.arange(1, nlay + 1)}, ("layer")
)

bot = top1D - dz

# Define WEL data, need to define the x, y, and pumping rate (q)
weldata = pd.DataFrame()
weldata["x"] = np.full(1, 0.5 * dx)
weldata["y"] = np.full(1, 0.5)
weldata["q"] = 0.28512  # positive, so it's an injection well


# Define the starting concentrations
sconc = xr.DataArray(
    data=np.full((nlay, nrow, ncol), 0.0),
    coords={
        "y": [0.5],
        "x": np.arange(0.5 * dx, dx * ncol, dx),
        "layer": np.arange(1, nlay + 1),
    },
    dims=("layer", "y", "x"),
)

sconc[75, :, 15:30] = 280.0
sconc.plot(y="layer", yincrease=False)

# Define the icbund
icbund = xr.DataArray(
    data=np.full((nlay, nrow, ncol), 1.0),
    coords={
        "y": [0.5],
        "x": np.arange(0.5 * dx, dx * ncol, dx),
        "layer": np.arange(1, nlay + 1),
    },
    dims=("layer", "y", "x"),
)

icbund[75, :, 0:15] = 0.0
icbund[75, :, 30:45] = 0.0
icbund[75, :, 15:30] = -1.0
icbund.plot(y="layer", yincrease=False)

# Define horizontal hydraulic conductivity
khv = xr.DataArray(
    data=np.full((nlay, nrow, ncol), 0.847584),
    coords={
        "y": [0.5],
        "x": np.arange(0.5 * dx, dx * ncol, dx),
        "layer": np.arange(1, nlay + 1),
    },
    dims=("layer", "y", "x"),
)

khv[75, :, 15:30] = 0.0008475
khv.plot(y="layer", yincrease=False)

# Define starting heads
shd = xr.DataArray(
    data=np.full((nlay, nrow, ncol), 0.0),
    coords={
        "y": [0.5],
        "x": np.arange(0.5 * dx, dx * ncol, dx),
        "layer": np.arange(1, nlay + 1),
    },
    dims=("layer", "y", "x"),
)

shd[0, :, :] = np.array(
    [
        10,
        9.772727273,
        9.545454545,
        9.318181818,
        9.090909091,
        8.863636364,
        8.636363636,
        8.409090909,
        8.181818182,
        7.954545455,
        7.727272727,
        7.5,
        7.272727273,
        7.045454545,
        6.818181818,
        6.590909091,
        6.363636364,
        6.136363636,
        5.909090909,
        5.681818182,
        5.454545455,
        5.227272727,
        5,
        4.772727273,
        4.545454545,
        4.318181818,
        4.090909091,
        3.863636364,
        3.636363636,
        3.409090909,
        3.181818182,
        2.954545455,
        2.727272727,
        2.5,
        2.272727273,
        2.045454545,
        1.818181818,
        1.590909091,
        1.363636364,
        1.136363636,
        0.909090909,
        0.681818182,
        0.454545455,
        0.227272727,
        0.00,
    ]
)
shd.plot(y="layer", yincrease=False)


# Finally, we build the model.
m = imod.wq.SeawatModel("Hydrocoin")
m["bas"] = imod.wq.BasicFlow(ibound=bnd, top=304.0, bottom=bot, starting_head=shd)
m["lpf"] = imod.wq.LayerPropertyFlow(
    k_horizontal=khv, k_vertical=khv, specific_storage=0.0
)
m["btn"] = imod.wq.BasicTransport(
    icbund=icbund, starting_concentration=sconc, porosity=0.2
)
m["adv"] = imod.wq.AdvectionTVD(courant=1.0)
m["dsp"] = imod.wq.Dispersion(longitudinal=20.0, diffusion_coefficient=0.0)
m["vdf"] = imod.wq.VariableDensityFlow(density_concentration_slope=0.71)
m["wel"] = imod.wq.Well(
    id_name="wel", x=weldata["x"], y=weldata["y"], rate=weldata["q"]
)
m["pcg"] = imod.wq.PreconditionedConjugateGradientSolver(
    max_iter=150, inner_iter=30, hclose=0.0001, rclose=0.1, relax=0.98, damp=1.0
)
m["gcg"] = imod.wq.GeneralizedConjugateGradientSolver(
    max_iter=150,
    inner_iter=30,
    cclose=1.0e-6,
    preconditioner="mic",
    lump_dispersion=True,
)
m["oc"] = imod.wq.OutputControl(save_head_idf=True, save_concentration_idf=True)
m.time_discretization(starttime="2000-01-01T00:00", endtime="2010-01-01T00:00")


# Now we write the model, including runfile:
m.write()

# You can run the model using the comand prompt and the iMOD SEAWAT executable

## Visualise results
# head = imod.idf.open("Hydrocoin/results/head/*.idf")
# head.plot(yincrease=False)
# conc = imod.idf.open("Hydrocoin/results/conc/*.idf")
# conc.plot(levels=range(0, 35, 5), yincrease=False)

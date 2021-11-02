import numpy as np
import xarray as xr

import imod

# Discretization
nrow = 1
ncol = 160
nlay = 82

dz = 1.875
dx = 3.75
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

# set constant heads
bnd[0, :, 0:40] = 0
bnd[0, :, 121:160] = 0
bnd[1, :, 0] = -1
bnd[1, :, 159] = -1
bnd.plot(y="layer", yincrease=False)

# setup tops and bottoms
top1D = xr.DataArray(
    np.arange(nlay * dz, 0.0, -dz), {"layer": np.arange(1, nlay + 1)}, ("layer")
)

bot = top1D - dz
top = nlay * dz
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

icbund[81, :, :] = -1
icbund[0, :, 41:120] = -1
icbund.plot(y="layer", yincrease=False)

# Define the starting concentration
sconc = xr.DataArray(
    data=np.full((nlay, nrow, ncol), 0.0),
    coords={
        "y": [0.5],
        "x": np.arange(0.5 * dx, dx * ncol, dx),
        "layer": np.arange(1, nlay + 1),
    },
    dims=("layer", "y", "x"),
)

sconc[81, :, :] = 0
sconc[0, :, 41:120] = 280.0

sconc.plot(y="layer", yincrease=False)

# Finally, we build the model.
m = imod.wq.SeawatModel("Elder")
m["bas"] = imod.wq.BasicFlow(ibound=bnd, top=top, bottom=bot, starting_head=0.0)
m["lpf"] = imod.wq.LayerPropertyFlow(
    k_horizontal=0.411, k_vertical=0.411, specific_storage=0.0
)
m["btn"] = imod.wq.BasicTransport(
    icbund=icbund, starting_concentration=sconc, porosity=0.1
)
m["adv"] = imod.wq.AdvectionTVD(courant=1.0)
m["dsp"] = imod.wq.Dispersion(longitudinal=0.0, diffusion_coefficient=0.308)
m["vdf"] = imod.wq.VariableDensityFlow(density_concentration_slope=0.71)
m["wel"] = imod.wq.Well(id_name="wel", x=0.5 * dx, y=0.5, rate=0.28512)
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
m.time_discretization(times=["2000-01-01T00:00", "2020-01-01T00:00"])

# Now we write the model

m.write()

# You can run the model using the comand prompt and the iMOD SEAWAT executable

# Visualise results
# head = imod.idf.open("Elder/results/head/*.idf")
# head.plot(yincrease=False)
# conc = imod.idf.open("Elder/results/conc/*.idf")
# conc.plot(levels=range(0, 35, 5), yincrease=False)

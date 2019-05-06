import imod
import numpy as np
import pandas as pd
import xarray as xr

# Discretization 
nrow = 1 # number of rows
ncol = 80 # number of column
nlay = 40 # number of layers

dz = 1.0 #0.0125
dx = 1.0 #0.0125
dy = -dx


# setup ibound
bnd = xr.DataArray(
    data=np.full((nlay, nrow, ncol), 1.0),
    coords = {
        "y": [0.5],
        "x": np.arange(0.5 * dx, dx * ncol, dx),
        "layer": np.arange(1, 1 + nlay),
        "dx": dx,
        "dy": dy
    },
    dims=("layer", "y", "x"),
)

bnd.plot()

# set constant heads
bnd[21,:,-80] = -1

# setup tops and bottoms

top1D = xr.DataArray(
    np.arange(nlay * dz, 0.0, -dz), {"layer":np.arange(1, nlay +1 )}, ("layer")
)

bot = top1D - dz

# Define WEL data, need to define the x, y, and pumping rate (q)
weldata = pd.DataFrame()
weldata["x"] = 0.5 * dx #np.full(1, 0.5 * dx)
weldata["y"] = 0.5 #np.full(1, 0.5)
weldata["q"] = 0.28512 # positive, so it's an injection well

# Defining the starting concentrations
sconc = xr.DataArray(
    data = np.full((nlay, nrow, ncol), 0.0),
    coords = {
            "y": [0.5], 
            "x": np.arange(0.5 * dx, dx * ncol, dx), 
            "layer": np.arange(1 , nlay + 1)
    },
    dims = ("layer", "y", "x"),
)

sconc[16:24,:,41:80] = 35.0
sconc.plot(y="layer", yincrease=False)

# Finally, we build the model.

m = imod.wq.SeawatModel("SaltwaterPocket")
m["bas"] = imod.wq.BasicFlow(ibound=bnd, top=40, bottom=bot, starting_head=0.0)
m["lpf"] = imod.wq.LayerPropertyFlow(
    k_horizontal=86.4, k_vertical=86.4, specific_storage=0.0
)
m["btn"] = imod.wq.BasicTransport(
    icbund=bnd, starting_concentration=sconc, porosity=0.1
)
m["adv"] = imod.wq.AdvectionTVD(courant=1.0)
m["dsp"] = imod.wq.Dispersion(longitudinal=0.001, diffusion_coefficient=0.0000864)
m["vdf"] = imod.wq.VariableDensityFlow(density_concentration_slope=0.71)
m['wel'] = imod.wq.Well(
    id_name='wel', x=weldata['x'], y=weldata['y'], rate=weldata["q"])
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
m.time_discretization(starttime="2000-01-01T00:00", endtime="2000-01-05T01:00")


# Now we write the model, including runfile:

m.write()


# You can run the model using the comand prompt and the iMOD SEAWAT executable

# Results

#head = imod.idf.open("SaltwaterPocket/results/head/*.idf")
#head.plot(yincrease=False)
#conc = imod.idf.open("SaltwaterPocket/results/conc/*.idf")
#conc.plot(levels=range(0, 35, 5), yincrease=False)
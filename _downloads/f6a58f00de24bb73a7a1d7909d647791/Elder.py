"""
Elder
=====

The classic 2D Elder problem demonstrates free convection.
Traditionally this was created for heat transport, but we use
a modified version for salt transport.
The conceptual model can be seen as a 2D sand box,
with on top a salt lake in the center and fresh lakes
on both the outer edges of the top row.

More info about the theory behind the Elder problem:

Simpson, J., & Clement, P. (2003).
Theoretical analysis of the worthiness of Henry and Elder
problems as benchmark of density-dependent groundwater flow models.
`Advances in Water Resources, 1708` (02).
Retrieved from http://www.eng.auburn.edu/~clemept/publsihed_pdf/awrmat.pdf
"""

# %%
# We'll start with the usual imports

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import imod

# %%
# Discretization
# --------------
#
# We'll start off by creating a model discretization, since
# this is a simple conceptual model.
# The model is a 2D cross-section, hence ``nrow = 1``.

nrow = 1
ncol = 160
nlay = 82

dz = 1.875
dx = 3.75
dy = -dx

# %%
# setup tops and bottoms
top1D = xr.DataArray(
    np.arange(nlay * dz, 0.0, -dz), {"layer": np.arange(1, nlay + 1)}, ("layer")
)

bot = top1D - dz
top = nlay * dz


# %%
# Set up ibound, which sets where active cells are `(ibound = 1.0)`
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

fig, ax = plt.subplots()
bnd.plot(y="layer", yincrease=False, ax=ax)

# %%
# Boundary Conditions
# -------------------
#
# Set the constant heads by specifying a negative value in iboud,
# that is: ``bnd[index] = -1```

bnd[0, :, 0:40] = 0
bnd[0, :, 121:160] = 0
bnd[1, :, 0] = -1
bnd[1, :, 159] = -1

fig, ax = plt.subplots()
bnd.plot(y="layer", yincrease=False, ax=ax)

# %%
# Define the icbund, which sets which cells
# in the solute transport model are active, inactive or constant.
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

fig, ax = plt.subplots()
icbund.plot(y="layer", yincrease=False, ax=ax)

# %%
# Initial Conditions
# ------------------
#
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

fig, ax = plt.subplots()
sconc.plot(y="layer", yincrease=False, ax=ax)

# %%
# Build
# -----
#
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
m.create_time_discretization(additional_times=["2000-01-01T00:00", "2020-01-01T00:00"])

# %%
# Now we write the model

modeldir = imod.util.temporary_directory()
m.write(modeldir, resultdir_is_workdir=True)

# %%
# Run
# ---
#
# You can run the model using the comand prompt and the iMOD-WQ executable.
# This is part of the iMOD v5 release, which can be downloaded here:
# https://oss.deltares.nl/web/imod/download-imod5 .
# This only works on Windows.
#
# To run your model, open up a command prompt
# and run the following commands:
#
# .. code-block:: batch
#
#    cd c:\path\to\modeldir
#    c:\path\to\imod\folder\iMOD-WQ_V5_3_SVN359_X64R.exe Elder.run
#
# Note that the version name of your executable might differ.
#

# %%
# Visualise results
# -----------------
#
# After succesfully running the model, you can
# plot results as follows:
#
# .. code:: python
#
#    head = imod.idf.open(modeldir / "results/head/*.idf")
#
#    fig, ax = plt.subplots()
#    head.plot(yincrease=False, ax=ax)
#
#    conc = imod.idf.open(modeldir / "results/conc/*.idf")
#
#    fig, ax = plt.subplots()
#    conc.plot(levels=range(0, 35, 5), yincrease=False, ax=ax)
#

# %%

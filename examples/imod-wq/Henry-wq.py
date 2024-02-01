"""
Henry
=====

The classic 2D Henry problem demonstrates the development of a fresh-salt
interface.
"""

# %%
# We'll start with the usual imports
import numpy as np
import pandas as pd
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
ncol = 100
nlay = 50

dz = 1.0
dx = 1.0
dy = -dx

top1D = xr.DataArray(
    np.arange(nlay * dz, 0.0, -dz), {"layer": np.arange(1, 1 + nlay)}, ("layer")
)
bot = top1D - 1.0

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

# Boundary Conditions
# -------------------
#
# We define constant head here, after generating the tops, or we'd end up with negative top values
bnd[:, :, -1] = -1

# %%
# Create WEL package
#
# First we scale the pumping rate with discretization
qscaled = 0.03 * (dz * abs(dy))

# %%
# Fresh water injection with well
# Add the arguments as a list, so pandas doesn't complain about having to set
# an index.
weldata = pd.DataFrame()
weldata["x"] = [0.5]
weldata["y"] = [0.5]
weldata["q"] = [qscaled]

# %%
# Build
# -----
#
# Finally, we build the model.

m = imod.wq.SeawatModel("Henry")
m["bas"] = imod.wq.BasicFlow(ibound=bnd, top=50.0, bottom=bot, starting_head=1.0)
m["lpf"] = imod.wq.LayerPropertyFlow(
    k_horizontal=10.0, k_vertical=10.0, specific_storage=0.0
)
m["btn"] = imod.wq.BasicTransport(
    icbund=bnd, starting_concentration=35.0, porosity=0.35
)
m["adv"] = imod.wq.AdvectionTVD(courant=1.0)
m["dsp"] = imod.wq.Dispersion(longitudinal=0.1, diffusion_coefficient=1.0e-9)
m["vdf"] = imod.wq.VariableDensityFlow(density_concentration_slope=0.71)
m["wel"] = imod.wq.Well(
    id_name="well", x=weldata["x"], y=weldata["y"], rate=weldata["q"]
)
m["pcg"] = imod.wq.PreconditionedConjugateGradientSolver(
    max_iter=150, inner_iter=30, hclose=0.0001, rclose=1.0, relax=0.98, damp=1.0
)
m["gcg"] = imod.wq.GeneralizedConjugateGradientSolver(
    max_iter=150,
    inner_iter=30,
    cclose=1.0e-6,
    preconditioner="mic",
    lump_dispersion=True,
)
m["oc"] = imod.wq.OutputControl(save_head_idf=True, save_concentration_idf=True)
m.create_time_discretization(
    additional_times=pd.date_range("2000-01-01", "2001-01-01", freq="M")
)

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
#    c:\path\to\imod\folder\iMOD-WQ_V5_3_SVN359_X64R.exe Henry.run
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


# %%

"""
TWRI
====

This example has been converted from the `MODFLOW6 Example problems`_.  See the
`description`_ and the `notebook`_ which uses `FloPy`_ to setup the model.

This example is a modified version of the original MODFLOW example
("`Techniques of Water-Resources Investigation`_" (TWRI)) described in
(`McDonald & Harbaugh, 1988`_) and duplicated in (`Harbaugh & McDonald, 1996`_).
This problem is also is distributed with MODFLOW-2005 (`Harbaugh, 2005`_). The
problem has been modified from a quasi-3D problem, where confining beds are not
explicitly simulated, to an equivalent three-dimensional problem.

In overview, we'll set the following steps:

    * Create a structured grid for a rectangular geometry.
    * Create the xarray DataArrays containg the MODFLOW6 parameters.
    * Feed these arrays into the imod mf6 classes.
    * Write to modflow6 files.
    * Run the model.
    * Open the results back into DataArrays.
    * Visualize the results.

"""
# %%
# We'll start with the usual imports. As this is an simple (synthetic)
# structured model, we can make due with few packages.

import matplotlib.pyplot as plt
from example_models import create_twri_simulation

import imod
from imod.mf6.partitioned_simulation_postprocessing import merge_balances, merge_heads
from imod.typing.grid import zeros_like

simulation = create_twri_simulation()
original_simulation = create_twri_simulation()
# %%
# We'll create a new directory in which we will write and run the model.
gwf_model = simulation["GWF_1"]
original_model = original_simulation["GWF_1"]

gwf_model.pop("wel")
original_model.pop("wel")

active = gwf_model.domain.sel(layer=1)

coords = active.coords
submodel_labels = zeros_like(active)

"""
submodel_labels.values[0:7, 0:7] = 0
submodel_labels.values[0:7, 7:] = 1
submodel_labels.values[7:, 0:7] = 2
submodel_labels.values[7:, 7:]  = 3

submodel_labels.values[0:7,:] = 0
submodel_labels.values[7:,:] = 1
submodel_labels.values[0:8,5] = 0
"""
for i in range(15):
    for j in range(i):
        submodel_labels.values[i, j] = 1


fig, ax = plt.subplots()
submodel_labels.plot.contourf(ax=ax)

simulation = simulation.split(submodel_labels)

modeldir = imod.util.temporary_directory()

original_simulation_dir = modeldir / "original"
original_simulation.write(original_simulation_dir, binary=False)
original_simulation.run()
original_balance = imod.mf6.open_cbc(
    original_simulation_dir / "GWF_1/GWF_1.cbc",
    original_simulation_dir / "GWF_1/dis.dis.grb",
)
fig, ax = plt.subplots()
original_head = imod.mf6.open_hds(
    original_simulation_dir / "GWF_1/GWF_1.hds",
    original_simulation_dir / "GWF_1/dis.dis.grb",
)
original_head.isel(layer=0, time=0).plot.contourf()
ax.title.set_text("original head")


simulation.write(modeldir, binary=False)

# %%
# Run the model
# -------------
#
# .. note::
#
#   The following lines assume the ``mf6`` executable is available on your PATH.
#   :ref:`The Modflow 6 examples introduction <mf6-introduction>` shortly
#   describes how to add it to yours.

simulation.run()

# %%
# Open the results (head)
# -----------------------
#
fig, ax = plt.subplots()
head = merge_heads(modeldir, simulation)
head.isel(layer=0, time=0).plot.contourf()
ax.title.set_text("head")

# %%
# Open the results (balance)
# --------------------------
#
balances = merge_balances(modeldir, simulation)

for key in balances:
    if "gwf-gwf" in key:
        new = balances[key]
        fig, ax = plt.subplots()

        new.isel(layer=0, time=-1).plot.contourf(ax=ax)
        ax.title.set_text(key)
        print(key)
        print(new.isel(layer=1, time=-1).values)
pass

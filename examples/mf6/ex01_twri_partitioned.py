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
import numpy as np
from example_models import create_twri_simulation

import imod
from imod.mf6.partitioned_simulation_postprocessing import merge_balances, merge_heads
from imod.typing.grid import zeros_like
from imod.mf6.partition_generator import get_label_array
simulation = create_twri_simulation()

# %%
# We'll create a new directory in which we will write and run the model.
gwf_model = simulation["GWF_1"]
active = gwf_model.domain.sel(layer=1)
number_partitions = 9
submodel_labels = get_label_array(simulation, number_partitions)

fig, ax = plt.subplots()
submodel_labels.plot.contourf(ax=ax)

simulation = simulation.split(submodel_labels)

modeldir = imod.util.temporary_directory()
simulation.write(modeldir, binary=False)

# %%
# Run the model.
# --------------
#
# .. note::
#
#   The following lines assume the ``mf6`` executable is available on your PATH.
#   :ref:`The Modflow 6 examples introduction <mf6-introduction>` shortly
#   describes how to add it to yours.

simulation.run()

# %%
# Open the results (head).
# ------------------------
#
fig, ax = plt.subplots()
head = merge_heads(modeldir, simulation)
head.isel(layer=0, time=0).plot.contourf()
ax.title.set_text("head")

# %%
# Open the results (balance).
# --------------------------
#
balances = merge_balances(modeldir, simulation)

fig, ax = plt.subplots()
balances["flow-front-face"].isel(layer=0, time=-1).plot.contourf(ax=ax)
ax.title.set_text("flow-front-face")

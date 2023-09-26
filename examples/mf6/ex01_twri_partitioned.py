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

import numpy as np
import xarray as xr
from example_models import create_twri_simulation

import imod
from imod.typing.grid import zeros_like

simulation = create_twri_simulation()

# %%
# We'll create a new directory in which we will write and run the model.
gwf_model = simulation["GWF_1"]
active = gwf_model.domain.sel(layer=1)
number_partitions = 3
split_location = np.geomspace(active.y.min(), active.y.max(), number_partitions + 1)

coords = active.coords
submodel_labels = zeros_like(active)
for id in np.arange(1, number_partitions):
    submodel_labels.loc[
        (coords["y"] > split_location[id]) & (coords["y"] <= split_location[id + 1])
    ] = id

simulation = simulation.split(submodel_labels)

modeldir = imod.util.temporary_directory()
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
# Open the results
# ----------------
#
# We'll open the heads (.hds) file.

# head = imod.mf6.open_hds(
#     modeldir / "GWF_1/GWF_1.hds",
#     modeldir / "GWF_1/dis.dis.grb",
# )
# head.isel(layer=0, time=0).plot.contourf()


heads = []
for id in np.arange(0, number_partitions):
    head = imod.mf6.open_hds(
        modeldir / f"GWF_1_{id}/GWF_1_{id}.hds",
        modeldir / f"GWF_1_{id}/dis.dis.grb",
    )
    heads.append(head)

head = xr.merge(heads)
head["head"].isel(layer=0, time=0).plot.contourf()

# %%
# Visualize the results
# ---------------------


# %%
# .. _MODFLOW6 example problems: https://github.com/MODFLOW-USGS/modflow6-examples
# .. _description: https://modflow6-examples.readthedocs.io/en/master/_examples/ex-gwf-twri.html
# .. _notebook: https://github.com/MODFLOW-USGS/modflow6-examples/tree/master/notebooks/ex-gwf-twri.ipynb
# .. _Techniques of Water-Resources Investigation: https://pubs.usgs.gov/twri/twri7-c1/
# .. _McDonald & Harbaugh, 1988: https://pubs.er.usgs.gov/publication/twri06A1
# .. _Harbaugh & McDonald, 1996: https://pubs.er.usgs.gov/publication/ofr96485
# .. _Harbaugh, 2005: https://pubs.er.usgs.gov/publication/tm6A16
# .. _FloPy: https://github.com/modflowpy/flopy

# %%

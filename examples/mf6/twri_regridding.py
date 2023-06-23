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
import xarray as xr
from example_models import create_twri_simulation

from imod.mf6.regridding_utils import RegridderType

# %%
# now we create the twri simulation itself. It yields a simulation of a flow problem, with a grid of 3 layers and 15 cells in both x and y directions.
# To better illustrate the regridding, we replace the K field with a lognormal random K field. The original k-field is a constant per layer.
simulation = create_twri_simulation()

idomain = simulation["GWF_1"]["dis"]["idomain"]
heterogeneous_k = xr.zeros_like(idomain, dtype=np.double)
heterogeneous_k.values = np.random.lognormal(-2, 2, heterogeneous_k.shape)
simulation["GWF_1"]["npf"]["k"] = heterogeneous_k

# %%
# Let's plot the k-field. This is going to be the input for the regridder, and the regridded output should somewhat resemble it.
fig, ax = plt.subplots()
heterogeneous_k.sel(layer=1).plot(y="y", yincrease=False, ax=ax)

# %%
# now we create a new grid for this simulation. It has 3 layers,  45 rows and 20 columns.
# The length of the domain is slightly different from the input grid. That was 15*5000 = 75000 long in x and y
# but the new grid is 75015 long in x and  75020 long in y

nlay = 3
nrow = 45
ncol = 20
shape = (nlay, nrow, ncol)

dx = 3751.0
dy = -1667.0
xmin = 0.0
xmax = dx * ncol
ymin = 0.0
ymax = abs(dy) * nrow
dims = ("layer", "y", "x")

layer = np.array([1, 2, 3])
y = np.arange(ymax, ymin, dy) + 0.5 * dy
x = np.arange(xmin, xmax, dx) + 0.5 * dx
coords = {"layer": layer, "y": y, "x": x, "dx": dx, "dy": dy}
target_grid = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)

# %%
# a first way to regrid the twri model is to regrid the whole simulation object. This is the most straightforward method,
# and it uses default regridding methods for each input field. To see which ones are used, look at the _regrid_method
# class attribute of the relevant package. For example the _regrid_method attribute  of the NodePropertyFlow package
# specifies that field "k" uses an OVERLAP regridder in combination with the averaging function "geometric_mean".
new_simulation = simulation.regrid_like("regridded_twri", target_grid=target_grid)

# %%
# Let's plot the k-field. This is the regridded output, and it should should somewhat resemble the original k-field plotted earlier.
regridded_k_1 = new_simulation["GWF_1"]["npf"]["k"]
fig, ax = plt.subplots()

regridded_k_1.sel(layer=1).plot(y="y", yincrease=False, ax=ax)

# %%
# a second way to regrid  twri  is to regrid the groundwater flow model.

model = simulation["GWF_1"]
new_model = model.regrid_like(target_grid)

regridded_k_2 = new_model["npf"]["k"]
fig, ax = plt.subplots()
regridded_k_2.sel(layer=1).plot(y="y", yincrease=False, ax=ax)


# %%
# finally, we can regrid package per package. This allows us to choose the regridding method as well.
# in this example we'll regrid the npf package manually and the rest of the packages using default methods.

regridder_types = {"k": (RegridderType.CENTROIDLOCATOR, None)}
npf_regridded = model["npf"].regrid_like(
    target_grid=target_grid, regridder_types=regridder_types
)
new_model["npf"] = npf_regridded


regridded_k_3 = new_model["npf"]["k"]
fig, ax = plt.subplots()
regridded_k_3.sel(layer=1).plot(y="y", yincrease=False, ax=ax)
pass  # break here to see the plots

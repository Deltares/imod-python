"""
TWRI
====

This example has been converted from the `MODFLOW 6 Example problems`_.  See the
`description`_ and the `notebook`_ which uses `FloPy`_ to setup the model.

This example is a modified version of the original MODFLOW example
("`Techniques of Water-Resources Investigation`_" (TWRI)) described in
(`McDonald & Harbaugh, 1988`_) and duplicated in (`Harbaugh & McDonald, 1996`_).
This problem is also is distributed with MODFLOW-2005 (`Harbaugh, 2005`_). The
problem has been modified from a quasi-3D problem, where confining beds are not
explicitly simulated, to an equivalent three-dimensional problem.

In overview, we'll set the following steps:

    * Create a structured grid for a rectangular geometry.
    * Create the xarray DataArrays containg the MODFLOW 6 parameters.
    * Feed these arrays into the imod mf6 classes.
    * Write to modflow6 files.
    * Run the model.
    * Open the results back into DataArrays.
    * Visualize the results.

"""

# %%
# We'll start with the usual imports. As this is a simple (synthetic)
# structured model, we can make due with few packages.


# TODO
# General comments from Joeri:
# - iMOD Python has a ``.create_time_discretization`` method, which is called at
#   the end, after all packages have been assigned to the model. This creates a
#   TDIS package based on all the times specfied in the simulation. For this, it
#   is necessary that the TDIS can be specified at the end. I'm not sure if you
#   flopy4dev supports providing a TDIS at the end of the script.

# TODO
# Would it be possible to have a longname for every class in the dfns for generation?
# Would it be possible to use the longname for every argument in each package? Revise all longnames in dfns.

import flopy4.mf6.gwf
import flopy4.mf6.simulation
import flopy4.mf6.solution
import flopy4.mf6.utils
import numpy as np
import xarray as xr

import imod

# %%
# Create grid coordinates
# -----------------------
#
# The first steps consist of setting up the grid -- first the number of layer,
# rows, and columns. Cell sizes are constant throughout the model.

nlay = 3
nrow = 15
ncol = 15
shape = (nlay, nrow, ncol)

dx = 5000.0
dy = -5000.0
xmin = 0.0
xmax = dx * ncol
ymin = 0.0
ymax = abs(dy) * nrow
dims = ("layer", "y", "x")

layer = np.array([1, 2, 3])
y = np.arange(ymax, ymin, dy) + 0.5 * dy
x = np.arange(xmin, xmax, dx) + 0.5 * dx
coords = {"layer": layer, "y": y, "x": x}

# %%
# Create DataArrays
# -----------------
#
# Now that we have the grid coordinates setup, we can start defining model
# parameters. The model is characterized by:
#
# * a constant head boundary on the left
# * a single drain in the center left of the model
# * uniform recharge on the top layer
# * a number of wells scattered throughout the model.

idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)
bottom = xr.DataArray([-200.0, -300.0, -450.0], {"layer": layer}, ("layer",))

# Constant head
constant_head = xr.full_like(idomain, np.nan, dtype=float).sel(layer=[1, 2])
constant_head[..., 0] = 0.0

# Drainage
elevation = xr.full_like(idomain.sel(layer=1), np.nan, dtype=float)
conductance = xr.full_like(idomain.sel(layer=1), np.nan, dtype=float)
elevation[7, 1:10] = np.array([0.0, 0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0, 100.0])
conductance[7, 1:10] = 1.0

# Recharge
rch_rate = xr.full_like(idomain.sel(layer=1), 3.0e-8, dtype=float)

# Well
# fmt: off
wells_x = [52500.0, 27500.0, 57500.0, 37500.0, 47500.0, 57500.0, 67500.0, 37500.0,
           47500.0, 57500.0, 67500.0, 37500.0, 47500.0, 57500.0, 67500.0, ]
wells_y = [52500.0, 57500.0, 47500.0, 32500.0, 32500.0, 32500.0, 32500.0, 22500.0,
           22500.0, 22500.0, 22500.0, 12500.0, 12500.0, 12500.0, 12500.0, ]
screen_top = [-300.0, -200.0, -200.0, 200.0, 200.0, 200.0, 200.0, 200.0,
              200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, ]
screen_bottom = [-450.0, -300.0, -300.0, -200.0, -200.0, -200.0, -200.0, -200.0,
                 -200.0, -200.0, -200.0, -200.0, -200.0, -200.0, -200.0, ]
rate_wel = [-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
            -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, ]
# fmt: on

# Node properties
icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

# %%
# Write the model
# ---------------
#
# The first step is to define an empty model, the parameters and boundary
# conditions are added in the form of the familiar MODFLOW packages.

# TODO: It's unclear what is mandatory to pass to Dis.
# Now all arguments have defaults, whereas quite a lot of them are
# optional. You can either create a Dis from a ncol/nrow/nlay with cellsizes or
# by directly providing grids. We are now not sure which are optional and which are not.
# Maybe just support one way to create a Dis, and add factory functions next to it.
dis = flopy4.mf6.gwf.Dis(top=200.0, bottom=bottom, idomain=idomain)

# TODO: We don't want to work on (nper,nnodes).
# Instead, we want to create a grid so that the data can come from a raster.
# We generate nper as the final step of model setup, so that the user can specify heterogenuous periods per package.
# And then create_time_discretization will outline them into one time discretization.
chd = flopy4.mf6.gwf.Chd(
    # This constant_head is now wrong and specified as (nlay,nrow,ncol)
    head=constant_head,
    print_input=True,
    print_flows=True,
    save_flows=True,
)
drn = flopy4.mf6.gwf.Drn(
    # TODO: Same problem here. We want to use a different shape.
    elev=elevation,
    cond=conductance,
    print_input=True,
    print_flows=True,
    save_flows=True,
)
ic = flopy4.mf6.gwf.Ic(strt=0.0)
npf = flopy4.mf6.gwf.Npf(
    # TODO: All griddata parameters are expected to be (nlay,nrow,ncol). Now we must convert them to (nnodes,).
    icelltype=icelltype,
    k=k,
    k33=k33,
    cvoptions=flopy4.mf6.gwf.Npf.CvOptions(variablecv=True, dewatered=True),
    perched=True,
    save_flows=True,
)
# TODO: Do we want to add two extra Sto subclasses that are hand-written in flopy4?
# It would make it easier for users to separate storage coefficient and specific storage.
# class SpecificStorage(flopy4.mf6.gwf.Sto) and class StorageCoefficient(flopy4.mf6.gwf.Sto).
# Or change the dfn so that there is an ss and an sc parameter.
sto = flopy4.mf6.gwf.Sto(
    storagecoefficient=False,
    ss=1.0e-5,
    sy=0.15,
    transient={"*": False},
    iconvert=0,
)
oc = flopy4.mf6.gwf.Oc(save_head="all", save_budget="all")
# TODO: Also here (nper,nnodes) is not what we want.
rch = flopy4.mf6.gwf.Rch(recharge=rch_rate)

# TODO: This would be a typical imod function to generate the well rate from coordinates into the grid.
q = imod.generate_well_rate(
    dis=dis,
    k=k,
    wells_x=wells_x,
    wells_y=wells_y,
    screen_top=screen_top,
    screen_bottom=screen_bottom,
    rate=rate_wel,
)
wel = flopy4.mf6.gwf.Wel(q=q)
# TODO: It's unclear which parameters are optional and which are not.
# It would be best if this was clarified in the docstrings with attrs.
# Recommendation: Only support one way to set parent-child relations.
# I.e. the way it is written here.
gwf_model = flopy4.mf6.gwf.Gwf(
    name="GWF_1",
    dis=dis,
    chd=chd,
    ic=ic,
    npf=npf,
    sto=sto,
    oc=oc,
    rch=rch,
    wel=wel,
    drn=drn,
)

# Define solver settings
solver = flopy4.mf6.solution.Solution(
    print_option="summary",
    outer_dvclose=1.0e-4,
    outer_maximum=500,
    under_relaxation=None,
    inner_dvclose=1.0e-4,
    inner_rclose=0.001,
    inner_maximum=100,
    linear_acceleration="cg",
    scaling_method=None,
    reordering_method=None,
    relaxation_factor=0.97,
)

# Attach it to a simulation
# We'll create a new directory in which we will write and run the model.
# TODO: The model name feels duplicate, because the model already has a name.
# TODO: What should be the key for solutions?
modeldir = imod.util.temporary_directory()
simulation = flopy4.mf6.simulation.Simulation(
    name="ex01-twri",
    models={"GWF_1": gwf_model},
    solutions={"solver": solver},
    workspace=modeldir,
)


# Collect time discretization
# TODO: How are we going to implement this with flopy4?
# flopy4 expects the time discretization to be set up from the start.
# This will assign the TDIS package to the simulation.
simulation.create_time_discretization(
    additional_times=["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]
)

# %%
simulation.write(format="binary")

# %%
# Run the model
# -------------
#
# .. note::
#
#   The following lines assume the ``mf6`` executable is available on your PATH.
#   :ref:`The MODFLOW 6 examples introduction <mf6-introduction>` shortly
#   describes how to add it to yours.

simulation.run()

# %%
# Open the results
# ----------------
#
# We'll open the heads (.hds) file.

head = flopy4.mf6.utils.open_hds(
    modeldir / "GWF_1/GWF_1.hds",
    modeldir / "GWF_1/dis.dis.grb",
)

# %%
# Visualize the results
# ---------------------

head.isel(layer=0, time=0).plot.contourf()


# %%
# .. _MODFLOW 6 example problems: https://github.com/MODFLOW-USGS/modflow6-examples
# .. _description: https://modflow6-examples.readthedocs.io/en/master/_examples/ex-gwf-twri.html
# .. _notebook: https://github.com/MODFLOW-USGS/modflow6-examples/blob/develop/scripts/ex-gwf-twri.py
# .. _Techniques of Water-Resources Investigation: https://pubs.usgs.gov/twri/twri7-c1/
# .. _McDonald & Harbaugh, 1988: https://pubs.er.usgs.gov/publication/twri06A1
# .. _Harbaugh & McDonald, 1996: https://pubs.er.usgs.gov/publication/ofr96485
# .. _Harbaugh, 2005: https://pubs.er.usgs.gov/publication/tm6A16
# .. _FloPy: https://github.com/modflowpy/flopy

# %%

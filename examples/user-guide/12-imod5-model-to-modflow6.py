"""
Working with iMOD5 models in MODFLOW 6
======================================

This example shows how to work with iMOD5 models in MODFLOW 6. It demonstrates
how to convert an iMOD5 model to a MODFLOW 6 model using the `imod` package. The
example fetches an iMOD5 model, converts it to a MODFLOW 6 model, regrids it to
an unstructured grid, and compares differences in output between the structured
and unstructured grids.

"""

import imod

# %%
#
# Fetching an iMOD5 model
# -----------------------
#
# Let's start by fetching the example data
# from the `imod.data` module. This will download a project file and
# accompanying data files to a temporary directory.

tmpdir = imod.util.temporary_directory()

prj_dir = tmpdir / "prj"
prj_dir.mkdir(exist_ok=True, parents=True)

model_dir = imod.data.fetch_imod5_model(prj_dir)

# %%
#
# Let's view the model directory. It contains the project file and
# accompanying model contents.

from pprint import pprint

imod_dir_contents = list(model_dir.glob("*"))
pprint(imod_dir_contents)

# %%
#
# The directory contains a project file and a database folder.
# This database contains all the IDF, IPF, and GEN files that make up the
# spatial model input.
#
# Let's look at the projectfile contents. Read the projectfile as follows:

prj_path = model_dir / "iMOD5_model.prj"
prj_content = imod.prj.read_projectfile(prj_path)
pprint(prj_content)

# %%
#
# This contains all the projectfile contents in a dictionary, which is quite a
# lot of information. This is too much to go through in detail. We can also open
# all data that the projectfile points to, using the
# :doc:`/api/generated/io/imod.prj.open_projectfile_data` function.

imod5_data, period_data = imod.prj.open_projectfile_data(prj_path)
imod5_data

# %%
#
# This groups all data per package in the projectfile into a dictionary with
# DataArrays per variable.

imod5_data["riv-1"]["stage"]

# %%
#
# Let's plot the stage data of the first river package.

imod5_data["riv-1"]["stage"].isel(layer=0, drop=True).plot.imshow()

# %%
#
# Converting iMOD5 model to MODFLOW 6
# -----------------------------------
#
# This is nice enough, but we want to convert this iMOD5 model to a MODFLOW 6
# model. We can do this using
# :doc:`/api/generated/mf6/imod.mf6.Modflow6Simulation.from_imod5_data` method.
# Next to the iMOD5 data and period data, we also need to provide the times.
# These will be used to resample the asynchronous well timeseries data to these
# times. For instance, well 1 in the iMOD5 database can have rates specified on
# a daily basis, whereas well 2 is specified on a few days in the year. Say the
# user wants to run a model on a monthly basis, this will require resampling
# these rate timeseries to make them consistent with the simulation timesteps.
# Let's therefore first create a list of times which will be the simulation's
# timesteps, we can use pandas for this. "MS" stands for "month start",
# meaning the first day of each month.

import pandas as pd

times = pd.date_range(start="2020-01-01", periods=10, freq="MS")
times

# %%
#
# Now that we have a list of times, we can import the iMOD5 data into a MODFLOW
# 6 simulation. This might require some time, as it will convert all the iMOD5
# data to be compatible with MODFLOW . For example, the river systems with
# infiltration factors are transformed into a separate
# Drain and River package (if necessary) to get the same behavior as iMOD5's
# infiltration factors.

mf6_sim = imod.mf6.Modflow6Simulation.from_imod5_data(imod5_data, period_data, times)
mf6_sim

# %%
#
# Improving the solver settings
# -----------------------------
#
# At the moment the MODFLOW 6 simulation has quite loose solver settings. Most
# notably, the ``inner_dvclose`` is set to 0.01, which means that the solver
# allows a numerical error of 1 cm in the head values. This is quite loose.

mf6_sim["ims"]

# %%
#
# This is because by default an iMOD5 model is imported with a
# SolutionPresetModerate, which is quite loose. Let's set a stricter solver
# setting preset, by setting it to SolutionPresetSimple. This has a
# ``inner_dvclose`` of 0.001, which allows a numerical error of 1 mm in the head
# values.

mf6_sim["ims"] = imod.mf6.SolutionPresetSimple(["imported_model"])
mf6_sim["ims"]

# %%
#
# A note on performance
# ---------------------
#
# By default, the iMOD5 model rasters will not be directly loaded into memory,
# but instead will be lazily loaded. Read more about this in the
# :doc:`06-lazy-evaluation` documentation.
#
# By default the data is chunked per raster file, which is a chunk per layer,
# per timestep. Usually this is not optimal, as this creates many small chunks.
# The discretization (DIS) package and node property flow (NPF) package are used
# frequently during the validation process, so it saves us a lot of waiting time
# if we load these into memory.
#
gwf_model = mf6_sim["imported_model"]
gwf_model["dis"].dataset.load()
gwf_model["npf"].dataset.load()

# %%
#
# Writing the structured model: in fits and starts
# ------------------------------------------------
#
# Let's try to write this simulation. **spoiler alert**: this will fail, because
# we still have to configure some packages.

mf6_dir = tmpdir / "mf6_structured"

# Ignore this "with" statement, it is to catch the error and render the
# documentation without error.
with imod.util.print_if_error(ValueError):
    mf6_sim.write(mf6_dir)  # Attention: this will fail!

# %%
#
# We are still missing output control, as the projectfile does not contain this
# information. For this example, we'll only save the last head of each stress period.


gwf_model["oc"] = imod.mf6.OutputControl(
    save_head="last",
)

from imod.schemata import ValidationError

with imod.util.print_if_error(ValidationError):
    mf6_sim.write(mf6_dir)  # Attention: this will fail!

# %%
#
# Argh! The simulation still fails to write. In general, iMOD Python is a lot
# stricter with writing model data than iMOD5. iMOD Python forces users to
# conciously clean up their models, whereas iMOD5 cleaned data under the hood.
# The error message states that the nodata is not aligned with idomain. This
# means there are k values and ic values specified at inactive locations, or
# vice versa. Let's try if masking the data works. This will remove all inactive
# locations (idomain==0) from the data.

idomain = gwf_model["dis"]["idomain"]
mf6_sim.mask_all_models(idomain)

mf6_sim.write(mf6_dir)

# %%
#
# Running the structured model
# ----------------------------
#
# Let's run the simulation and open the head data.

mf6_sim.run()
head_structured = mf6_sim.open_head()
# Plot the head of the last stress period at layer 5.
head_structured.isel(time=-1).sel(layer=5).plot.imshow()

# %%
#
# Regridding the structured model to an unstructured grid
# -------------------------------------------------------
#
# Now that we have a MODFLOW 6 simulation, we can regrid it to an unstructured
# grid. Let's first load a triangular grid.

triangular_grid = imod.data.lhm_clip_triangular_grid()
triangular_grid.plot()

# %%
#
# That looks more exciting than the rectangular grid we had before. You can see
# there is refinement around some of the streams and especially around
# horizontal flow barriers. We haven't looked at horizontal flow barriers yet,
# so let's plot them on top of the triangular mesh.

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
triangular_grid.plot(ax=ax, color="lightgrey", edgecolor="black")
gwf_model["hfb-25"].line_data.plot(ax=ax, color="blue", linewidth=2)
gwf_model["hfb-26"].line_data.plot(ax=ax, color="blue", linewidth=2)

# %%
#
# However, this grid is triangular, which has the disadvantage that the connections
# between cell centers are not orthogonal to the cell edges, which can lead to
# mass balance errors. xugrid has a method to convert this triangular grid
# to a Voronoi grid, which has orthogonal connections between cell centers and
# edges.

voronoi_grid = triangular_grid.tesselate_centroidal_voronoi()
voronoi_grid.plot()

# %%
#
# Now that we have a Voronoi grid, we can regrid the MODFLOW 6 simulation to this
# grid.

# Workaround for bug where iMOD Python doesn't recognize Ugrid2D, only UgridDataArray
import numpy as np
import xarray as xr
import xugrid as xu

data = xr.DataArray(
    np.ones(voronoi_grid.sizes["mesh2d_nFaces"]), dims=["mesh2d_nFaces"]
)
voronoi_uda = xu.UgridDataArray(data, voronoi_grid)

mf6_unstructured = mf6_sim.regrid_like(
    "unstructured_example", voronoi_uda, validate=False
)
mf6_unstructured

# %%
#
# Let's take a gander at how the river data is regridded.

mf6_unstructured["imported_model"]["riv-1riv"]["stage"].isel(layer=0).ugrid.plot()

# %%
#
# Writing the unstructured model: in more fits and starts
# -------------------------------------------------------
#
# Let's try to write this to a temporary directory. **Spoiler alert**: Like
# before, this will fail.

mf6_dir = tmpdir / "mf6_unstructured"

# Ignore this "with" statement, it is to catch the error and render the
# documentation without error.
with imod.util.print_if_error(ValidationError):
    mf6_unstructured.write(mf6_dir)  # Attention: this will fail!

# %%
#
# The error message states that the iMOD5 model has a river package that has its
# river bottom elevation below the model bottom. The averaging when regridding
# can cause this: The model bottom has a continuous surface, whereas the rivers
# usually are located in a local valley. Upscaling both with a mean causes the
# river bottom elevation to have the tendency to be lower than the model bottom.
# We therefore need to reallocate the river data to the new model layer
# schematization. There currently is no direct method to do this, but we can
# reallocate the river dataset with the following function.


def reallocate_river(river, dis, npf, allocation_option, distributing_option):
    """
    Reallocates river data across layers in place. Aggregate river data to
    planar data first, by taking the mean across layers for the stage and bottom
    elevation, and the sum for the conductance. Consequently allocate and
    distribute the planar data to the provided model layer schematization.

    Parameters
    ----------
    river_ds : River
        The river package to reallocate.
    dis : StructuredDiscretization | VerticesDiscretization
        The discretization of the model to which the river data should be
        reallocated.
    npf : NodePropertyFlow
        The node property flow package of the model to which the river
        conductance should be distributed (if applicable).
    allocation_option : ALLOCATION_OPTION
        The allocation option to use for the reallocation.
    distributing_option : DISTRIBUTING_OPTION
        The distributing option to use for the reallocation.
    """
    river_ds = river.dataset
    aggr_dict = {
        "stage": np.nanmean,
        "conductance": np.nansum,
        "bottom_elevation": np.nanmean,
    }
    planar_data = {
        key: river_ds[key].reduce(func, dim="layer") for key, func in aggr_dict.items()
    }
    riv_dict, _ = imod.mf6.River.allocate_and_distribute_planar_data(
        planar_data, dis, npf, allocation_option, distributing_option
    )
    # .update for some reason requires the dimensions to be specified.
    riv_dict = {key: (da.dims, da) for key, da in riv_dict.items()}
    river_ds.update(riv_dict)


def reallocate_drain(drain, dis, npf, allocation_option, distributing_option):
    """
    Reallocates river data in place. Aggregate river data to planar data first,
    by taking the mean across layers for the stage and bottom elevation, and the
    sum for the conductance. Consequently allocate and distribute the planar
    data to the provided model layer schematization.

    Parameters
    ----------
    drain : Drainage
        The river dataset to reallocate.
    dis : StructuredDiscretization | VerticesDiscretization
        The discretization of the model to which the river data should be
        reallocated.
    npf : NodePropertyFlow
        The node property flow package of the model to which the river
        conductance should be distributed (if applicable).
    allocation_option : ALLOCATION_OPTION
        The allocation option to use for the reallocation.
    distributing_option : DISTRIBUTING_OPTION
        The distributing option to use for the reallocation.
    """
    drain_ds = drain.dataset
    aggr_dict = {"elevation": np.nanmean, "conductance": np.nansum}
    planar_data = {
        key: drain_ds[key].reduce(func, dim="layer") for key, func in aggr_dict.items()
    }
    drn_dict = imod.mf6.Drainage.allocate_and_distribute_planar_data(
        planar_data, dis, npf, allocation_option, distributing_option
    )
    # .update for some reason requires the dimensions to be specified.
    drn_dict = {key: (da.dims, da) for key, da in drn_dict.items()}
    drain_ds.update(drn_dict)


from imod.prepare import ALLOCATION_OPTION, DISTRIBUTING_OPTION

gwf_unstructured = mf6_unstructured["imported_model"]
dis = gwf_unstructured["dis"]
npf = gwf_unstructured["npf"]

riv_args = (
    dis,
    npf,
    ALLOCATION_OPTION.stage_to_riv_bot,
    DISTRIBUTING_OPTION.by_corrected_transmissivity,
)
drn_args = (
    dis,
    npf,
    ALLOCATION_OPTION.at_elevation,
    DISTRIBUTING_OPTION.by_corrected_transmissivity,
)
reallocate_river(gwf_unstructured["riv-1riv"], *riv_args)
reallocate_river(gwf_unstructured["riv-2riv"], *riv_args)
reallocate_drain(gwf_unstructured["riv-1drn"], *drn_args)
reallocate_drain(gwf_unstructured["riv-2drn"], *drn_args)

gwf_unstructured["riv-1riv"].cleanup(dis)
gwf_unstructured["riv-2riv"].cleanup(dis)
gwf_unstructured["riv-1drn"].cleanup(dis)
gwf_unstructured["riv-2drn"].cleanup(dis)

# %%
#
# Finally, we need to set the HFB validation settings to less strict. Otherwise,
# we'll get errors about hfb's being connected to inactive cells. Normally, you
# would set this when creating a new Modflow6Simulation, but since we created
# one from an iMOD5 model, these settings cannot be set upon creation. We can
# however set the validation settings directly by changing this attribute:

from imod.mf6 import ValidationSettings

mf6_unstructured._validation_context = ValidationSettings(strict_hfb_validation=False)

# %%
#
# We'll now be able to finally write the unstructured model.

mf6_unstructured.write(mf6_dir)

# %%
#
# Running the unstructured model
# ------------------------------
#
# Let's run the unstructured model and open the head data.

mf6_unstructured.run()
head_unstructured = mf6_unstructured.open_head()
# Plot the head of the last stress period at layer 5.
head_unstructured.isel(time=-1).sel(layer=5).ugrid.plot()

# %%
#
# Comparing differences in output
# -------------------------------
#
# Let's upscale the structured head data to the unstructured grid. This is done
# using the `OverlapRegridder from the xugrid package
# <https://deltares.github.io/xugrid/examples/regridder_overview.html#overlapregridder>_,

import xugrid as xu

regridder = xu.OverlapRegridder(head_structured, head_unstructured.ugrid.grid)
head_structured_upscaled = regridder.regrid(head_structured)

# %%
#
# Compute the difference between the upscaled structured head and the
# unstructured head. A zero difference means the regridding didn't result in any
# differences. We can see around the western fault that the regridding caused
# differences.

diff = head_structured_upscaled - head_unstructured
diff.isel(time=-1).mean(dim="layer").ugrid.plot()

# %%
#
# Let's also plot the standard deviation of the difference. This shows that
# variations in difference are also mostly around the western fault.

diff.isel(time=-1).std(dim="layer").ugrid.plot()

# %%
#
# EXERCISE: Download this file as a script or Jupyter notebook, remove all HFB
# packages and re-run the example. Investigate if differences are still as large
# as they were.

# %%

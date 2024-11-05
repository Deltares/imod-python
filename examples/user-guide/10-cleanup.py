"""
More often than not, data contained in databases is not entirely consistent,
causing errors. It therefore is useful to have some utilities at hand to clean
up data. We included some convenience methods to help cleaning up inconsistent
datasets.
"""
import numpy as np

import imod

# %%
# There is a separate example contained in
# :doc:`hondsrug </examples/mf6/hondsrug>`
# that you should look at if you are interested in the model building
tmpdir = imod.util.temporary_directory()

gwf_simulation = imod.data.hondsrug_simulation(tmpdir / "hondsrug_saved")

# %%

def get_colleagues_data(gwf_model):
    import xarray as xr
    dis_ds = gwf_model["dis"].dataset
    riv_ds_old = gwf_model["riv"].dataset
    # Existing RIV package has only layer coord with [1, 3, 5, 6]. This causes
    # problems with some methods, at least with cleanup. Therefore align data
    # here for this example. Not sure if we want to support river packages with
    # limited layer coords.
    riv_ds, _ = xr.align(riv_ds_old, dis_ds, join="outer")
    x = riv_ds.coords["x"]
    y = riv_ds.coords["y"]
    riv_bot_da = riv_ds["bottom_elevation"]
    riv_ds["stage"] += 0.05
    riv_ds["stage"] = riv_ds["stage"].where(x > 239500)
    riv_ds["conductance"] = riv_ds["conductance"].fillna(0.0)
    x_preserve = (x < 244200) | (x > 246000)
    y_preserve = (y < 560000) | (y > 561000)
    riv_ds["bottom_elevation"] = riv_bot_da.where(
        x_preserve | y_preserve, 
    riv_bot_da + 0.15)
    return riv_ds


class PrintErrorInsteadOfRaise:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, tb):
        if exc_type:
            print(f"{exc_val}")
            return True  # swallow the exception

# %%
#
# Update existing model with new data
# -----------------------------------
#
# Your dear colleague has brought you some new data for tile drainage, which
# should be much better than the previous dataset "riv" included in the
# database.

gwf_model = gwf_simulation["GWF"]
new_riv_ds = get_colleagues_data(gwf_model)

# %%
# Let's do a brief visual check if the colleague's data seems alright:

imod.visualize.plot_map(
    new_riv_ds["stage"].max(dim="layer"), "viridis", np.linspace(-1, 19, 9)
)

# %% 
# 
# Hmmmm, the western side of the river stage grid seems suspiciously inactive...
# We have to contact our colleague later. For now, let's work with what we have
# and update the model.

old_riv = gwf_model.pop("riv")

gwf_model["new_riv"] = imod.mf6.River(**new_riv_ds)

# %%
# 
# Lets's write the simulation with our updated model!

tmp_dir = imod.util.temporary_directory()

with PrintErrorInsteadOfRaise():
    gwf_simulation.write(tmp_dir)

# %%
#
# Oh no! Our river package has a completely inconsistent dataset.
# The model validation raises the following issues:
# 
# * The bottom elevation exceeds stage in some cells
# * NoData cells are not aligned between stage and conductance
# * NoData cells are not aligned between stage and bottom_elevation
# * There are conductance values with value <= 0.0

# %%
# 
# *Exercise*: Use the function ``imod.visualize.plot_map`` to visually inspect
# the errors in your colleagues dataset. You can alter the variable name to
# inspect different variables.
# 

imod.visualize.plot_map(
    new_riv_ds["stage"].max(dim="layer"), "viridis", np.linspace(-1, 19, 9)
)

# %% 
# 
# We can also check where stage exceeds river bottom elevation using basic
# xarray functionality.

stage_above_riv_bot = new_riv_ds["stage"] < new_riv_ds["bottom_elevation"]

imod.visualize.plot_map(
    stage_above_riv_bot.max(dim="layer"), "viridis", [0, 1]
)

# %%
# 
# So luckily the area affected by this error is only small. Let's investigate
# the size of the error here.
# 

diff = new_riv_ds["bottom_elevation"] - new_riv_ds["stage"]

max_diff = diff.max()

max_diff

# %%
#
# That is a relatively subtle error, which is probably why it slipped your
# colleague's attention. Let's plot the troublesome area:

imod.visualize.plot_map(
    diff.where(stage_above_riv_bot).max(dim="layer"), "viridis", np.linspace(0, max_diff.values, 9)
)

# %%
# 
# That is only a small area. Plotting these errors can help you analyze the size
# of problems and communicate with your colleague where there are problems.
#
# Data cleanup
# ------------
# 
# The data quality seems acceptable for now to carry on modelling, so we can
# already set up a model for the project. We need to communicate with colleagues
# later that there are mistakes, so that the errors can be corrected.
# 
# iMOD Python has a set of utilities to help you with cleaning erronous data.
# Note that these cleanup utilities can only fix a limited set of common
# mistakes, in ways you might not need/like. It therefore is always wise to
# verify if you think the fixes are correct in this case.
#
# Let's clean up the problematic river data using the
# :meth:`imod.mf6.River.cleanup` method. Note that this method changes the data
# inplace. This means that the package's will be updated, without returning a
# new copy of the package. This means that do comparisons of the
# dataset before and after the fix, we can copy the dirty dataset first.

dirty_ds = gwf_model["new_riv"].dataset.copy()

# %% 
# 
# We can now clean up the package. To clean up the River package, the method
# requires the model discretization, contained in the
# :class:`imod.mf6.StructuredDiscretization` package. In this model, we named
# this package "dis".

dis_pkg = gwf_model["dis"]
gwf_model["new_riv"].cleanup(dis=dis_pkg)

cleaned_ds = gwf_model["new_riv"].dataset

# %%
#
# According to the method's explanation (the method calls
# :func:`imod.prepare.cleanup_riv`), the bottom elevation should be altered by
# the cleanup function.
#
# Let's first see if the stages were altered, we'll calculate the difference
# between the original stage and cleaned version.

diff_stage = dirty_ds["stage"] - cleaned_ds["stage"]

imod.visualize.plot_map(
    diff_stage.max(dim="layer"), "viridis", np.linspace(0, max_diff.values, 9)
)

# %%
#
# The stages are indeed not altered.
#
# Let's see if the bottom elevations are lowered, we'll calculate the difference
# between the original bottom elevation and cleaned version.

diff_riv_bot = dirty_ds["bottom_elevation"] - cleaned_ds["bottom_elevation"]

imod.visualize.plot_map(
    diff_riv_bot.max(dim="layer"), "viridis", np.linspace(0, max_diff.values, 9)
)
# %%
#
# You can see the bottom elevation was lowered by the cleanup method.
# Furthermore the area in the west where no active stages were defined are also
# deactivated.
#
# Writing the cleaned model
# -------------------------
#
# The river package has been Let's see if we can write the model.

gwf_simulation.write(tmp_dir)

# %%
#
# Great! The model was succesfully written!
#
# Cleaning data without a MODFLOW6 simulation
# -------------------------------------------
#
# There might be situations where you do not have a MODFLOW6 simulation or River
# package at hand, and you still want to clean up your river grids. For this you
# can use the :func:`imod.prepare.cleanup_riv` function. For this you need to
# separately provide your grids.
#

idomain = dis_pkg["idomain"]
bottom = dis_pkg["bottom"]
stage = dirty_ds["stage"]
conductance = dirty_ds["conductance"]
bottom_elevation = dirty_ds["bottom_elevation"]

riv_cleaned_dict = imod.prepare.cleanup_riv(
    idomain, bottom, stage, conductance, bottom_elevation
)

riv_cleaned_dict

# %%
#
# This returns a dictionary which you can use however you want, and furthermore
# feed to the River package.

riv_pkg = imod.mf6.River(**riv_cleaned_dict)

riv_pkg

# %%
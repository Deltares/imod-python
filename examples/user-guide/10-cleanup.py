"""
Data cleanup
============

More often than not, data contained in databases is not entirely consistent,
causing errors. It therefore is useful to have some utilities at hand to clean
up data. We included some convenience methods to help clean up inconsistent
datasets.
"""

# sphinx_gallery_thumbnail_number = -1

# %%
# We'll start with the usual imports
import matplotlib.pyplot as plt
import numpy as np

import imod

# %%
# These imports can be ignored
from imod.schemata import ValidationError
from imod.util import print_if_error

# %%
# There is a separate example contained in
# :doc:`hondsrug </examples/mf6/hondsrug>`
# that you should look at if you are interested in the model building
tmpdir = imod.util.temporary_directory()

gwf_simulation = imod.data.hondsrug_simulation(tmpdir / "hondsrug_saved")


# %%
#
# Update existing model with new data
# -----------------------------------
#
# Your dear colleague has brought you some new data for the river package, which
# should be much better than the previous dataset "riv" included in the
# database.

gwf_model = gwf_simulation["GWF"]
new_riv_ds = imod.data.colleagues_river_data(tmpdir / "hondsrug_saved")

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

# Ignore the "with" statement, it is to succesfully render the Jupyter notebook
# online.
with print_if_error(ValidationError):
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

imod.visualize.plot_map(stage_above_riv_bot.max(dim="layer"), "viridis", [0, 1])

# %%
#
# So luckily the area affected by this error is only small. Let's investigate
# the size of the error here.
#

diff = new_riv_ds["bottom_elevation"] - new_riv_ds["stage"]

max_diff = diff.max().values

max_diff

# %%
#
# That is a relatively subtle error, which is probably why it slipped your
# colleague's attention. Let's plot the troublesome area:

imod.visualize.plot_map(
    diff.where(stage_above_riv_bot).max(dim="layer"),
    "viridis",
    np.linspace(0, max_diff, 9),
)

# %%
#
# That is only a small area. Plotting these errors can help you analyze the size
# of problems and communicate with your colleague where there are problems.
#
# The ``cleanup`` method
# ----------------------
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
# inplace. This means that the package will be updated, without returning a new
# copy of the package. This means that we need to copy the dirty dataset first,
# before we can do comparisons of the dataset before and after the fix.

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

fig, ax = imod.visualize.plot_map(
    diff_stage.max(dim="layer"), "viridis", np.linspace(0, max_diff, 9)
)
ax.set_title("stage lowered by cleanup (m)")
plt.show()

# %%
#
# The stages are indeed not altered.
#
# Let's see if the bottom elevations are lowered, we'll calculate the difference
# between the original bottom elevation and cleaned version.

diff_riv_bot = dirty_ds["bottom_elevation"] - cleaned_ds["bottom_elevation"]

fig, ax = imod.visualize.plot_map(
    diff_riv_bot.max(dim="layer"), "viridis", np.linspace(0, max_diff, 9)
)
ax.set_title("river bottom lowered by cleanup (m)")
plt.show()

# %%
#
# You can see the bottom elevation was lowered by the cleanup method.
# Furthermore the area in the west where no active stages were defined are also
# deactivated.
#
# We advice to always verify if the data is cleaned in a manner that fits your
# use-case. For example, you might need to raise the stages to the river bottom
# elevation, instead of lowering the latter to the former like the ``cleanup``
# method just did. In such case, you need to manually fix the data.
#
# The ``cleanup`` method helps getting your data through the model validation,
# but is not guaranteed to be the "best" way to clean up your data!
#
# Writing the cleaned model
# -------------------------
#
# Now that the river package has been cleaned, let's see if we can write the
# model.

gwf_simulation.write(tmp_dir)

# %%
#
# Great! The model was succesfully written!
#
# Cleaning data without a MODFLOW6 simulation
# -------------------------------------------
#
# There might be situations where you do not have a MODFLOW6 simulation or River
# package at hand, and you still want to clean up your river grids. In this
# case, you can use the :func:`imod.prepare.cleanup_riv` function. This function
# requires you to to separately provide your grids and returns a dictionary of
# grids.
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

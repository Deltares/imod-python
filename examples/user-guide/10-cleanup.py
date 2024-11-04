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
    from copy import deepcopy
    riv_ds = deepcopy(gwf_model["riv"].dataset)
    x = riv_ds.coords["x"]
    riv_bot_da = riv_ds["bottom_elevation"]
    riv_ds["stage"] += 0.05
    riv_ds["stage"] = riv_ds["stage"].where(x > 239500)
    riv_ds["conductance"] = riv_ds["conductance"].fillna(0.0)
    riv_ds["bottom_elevation"] = riv_bot_da.where(
        (x > 244000) & (x < 246000), 
    riv_bot_da + 0.15)
    return riv_ds



# %%
#
# Your dear colleague has brought you some new data for tile drainage, which
# should be much better than the previous dataset "riv" included in the
# database.

gwf_model = gwf_simulation["GWF"]
new_riv_ds = get_colleagues_data(gwf_model)

# %%
# Let's do a brief visual check if the colleague's data seems alright:

# Plot
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

gwf_simulation.write(tmp_dir)

# %%
#
# Oh no! Our river package has a completely inconsistent dataset.
# The model validation raises the following issues:
# 
# * The bottom elevation exceeds stage in some cells
# * NoData cells are not aligned between stage and conductance
# * NoData cells are not aligned between stage and bottom_elevation
# * There are conductance values with value < 0.0
# 
# *Exercise*: Use the function ``imod.visualize.plot_map`` to visually inspect
# the errors in your colleagues dataset.
# 

# %%






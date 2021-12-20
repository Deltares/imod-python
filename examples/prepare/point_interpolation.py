"""
Interpolating points to a grid
******************************

"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import imod

# Heads interpolation
# --------------------
#
# The starting heads to be used for a model could be based on the
# interpolation of x-y head
# measurements. In order to have better interpolation results,
# an area larger than the model domain should be considered.
# The larger area to be used as reference is loaded
# and used to create a base array filled with NaNs,
# which will be later modified to include the heads interpolation.
#
# Using a larger area for the heads interpolation

# TODO: Add data to pooch, and load in that manner.
wdir = Path(
    r"c:\Users\engelen\projects_wdir\imod-python\examples\data_betsy\drenthe_input"
)
# TODO: Add data to pooch, and load in that manner.
idomain_larger = xr.open_dataarray(wdir / "idomain_larger.nc")
like_2d_larger = xr.full_like(idomain_larger, np.nan).squeeze(drop="layer")

like_2d_larger

# %%
# Head measurements information has been obtained from the Dinoloket website for the
# case study area. This data consists on a .csv file
# (read using Pandas
# `pd.read_csv <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv>`_
# )
# that contains the following columns:
# `id`, `time`, `head`, `filt_top`, `filt_bot`, `elevation`, `x` and `y`.
# For this example, all the measurements at different depths and
# along time for a same id were averaged, to have one reference head value in each point.

heads = pd.read_csv(wdir / "dino_validatie_data.csv")
mean_heads = heads.groupby("id").mean()  # Calculating mean values by id
x = mean_heads["x"]
y = mean_heads["y"]

mean_heads.head(5)

# %%
# The previous head information needs to be assigned to the model grid.
# imod-python has a tool called
# `imod.select.points_set_values <https://imod.xyz/api/select.html#imod.select.points_set_values>`_,
# which assigns values based on x-y coordinates to a previously defined array.
# In this case, the array is the starting_heads_larger,
# the values are the mean calculated heads and the x and y are the coordinates corresponding to the heads.

starting_heads_larger = imod.select.points_set_values(
    like_2d_larger, values=mean_heads["head"].to_list(), x=x.to_list(), y=y.to_list()
)

# Plotting the points
starting_heads_larger.plot.imshow()

# %%
# The previous information is still only available at certain points,
# so it needs to be interpolated. The imod-python tool
# `imod.prepare.laplace_interpolate <https://imod.xyz/api/prepare.html#imod.prepare.laplace_interpolate>`_
# will be used to do an interpolation of the previously indicated head values.
# It is possible to assign interpolation parameters such as
# the number of iterations and the closing criteria.

interpolated_head_larger = imod.prepare.laplace_interpolate(
    starting_heads_larger, close=0.001, mxiter=150, iter1=100
)

# Plotting the interpolation results
interpolated_head_larger.plot.imshow()

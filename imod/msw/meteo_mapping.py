import numpy as np
import pandas as pd
import xarray as xr

from imod.prepare import common


def grid_mapping(svat_grid: xr.DataArray, meteo_grid: xr.DataArray) -> pd.DataFrame:
    flip_svat_x = svat_grid.indexes["x"].is_monotonic_decreasing
    flip_svat_y = svat_grid.indexes["y"].is_monotonic_decreasing
    flip_meteo_x = meteo_grid.indexes["x"].is_monotonic_decreasing
    flip_meteo_y = meteo_grid.indexes["y"].is_monotonic_decreasing
    nrow = meteo_grid["y"].size
    ncol = meteo_grid["y"].size

    # Ensure all are increasing
    pass

    # Convert to cell boundaries for the meteo grid
    meteo_x = common._coord(meteo_grid, "x")
    meteo_y = common._coord(meteo_grid, "y")

    # Maybe side="left" or side="right" is appropriate...
    index_x = np.searchsorted(svat_grid["x"].values, meteo_x)
    index_y = np.searchsorted(svat_grid["y"].values, meteo_x)

    # Find out of bounds members
    if (index_x == 0).any() or (index_x >= ncol).any():
        raise ValueError("out of bounds for x")
    if (index_y == 0).any() or (index_y >= nrow).any():
        raise ValueError("out of bounds for y")
    rows, columns = np.meshgrid(index_y, index_x, indexing="ij")
    if flip_meteo_x ^ flip_svat_x:  # or something?
        rows = (nrow - 1) - rows

    # Repeat for multiple stacked SVATs I guess?
    return pd.DataFrame({"svat": svat, "row": rows, "column": columns})
